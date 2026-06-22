#!/usr/bin/env python3
"""Dump every intermediate activation from the reference model.

The verification harness, and the single most valuable tool in the project. It
runs HuggingFace GPT-2 on one FIXED prompt with greedy decoding and writes every
intermediate tensor to disk: the embeddings, and for each block the post-LN_1,
post-attention, post-LN_2, post-MLP, and post-block hidden states, then the
final-norm output and the logits.

The C++ engine dumps the same tensors for the same input ids, and
``tests/check_layers.py`` diffs the two and reports the FIRST tensor that
diverges -- turning a three-day "stare at twelve layers" bug into a
twenty-minute "layer 4's attention output is wrong" fix.

Outputs (into --out-dir):
    order.txt          forward-order list of tensor names (one per line)
    input_ids.npy      int32 token ids for the prompt (feed these to the engine
                       directly during bringup to decouple tokenizer from model)
    <name>.npy         each captured activation, squeezed to 2-D [seq, dim]

GPT-2 uses the tanh ("gelu_new") GELU; the C++ engine must use the same approx,
or these dumps will disagree at the 1e-3 level for no real reason.

Usage:
    pip install -r requirements.txt
    python tools/reference.py --model gpt2 --out-dir tests/reference_dumps
    python tools/reference.py --arch llama   # TinyLlama -> tests/reference_dumps_llama

The same canonical tensor names are captured for both architectures (00_embed,
<tag>.ln_1/.attn/.ln_2/.mlp/.block, zz_ln_f, zz_logits), so the diff harness is
arch-agnostic. For Llama, ln_1/ln_2 are the RMSNorms (input / post-attention) and
00_embed is the token embedding alone (RoPE is applied later, inside attention).
"""

import argparse
import os
import sys

# Freeze ONE prompt for the whole project. Do not change it casually -- the
# committed reference dumps depend on it.
FIXED_PROMPT = "The quick brown fox"


def _hooks_gpt2(model, grab, handles) -> None:
    tf = model.transformer
    # Embedding output (token + position, after the dropout which is a no-op in eval).
    handles.append(tf.drop.register_forward_hook(grab("00_embed")))
    for i, block in enumerate(tf.h):
        tag = f"{i + 1:02d}_L{i:02d}"
        handles.append(block.ln_1.register_forward_hook(grab(f"{tag}.ln_1")))
        handles.append(block.attn.register_forward_hook(grab(f"{tag}.attn")))
        handles.append(block.ln_2.register_forward_hook(grab(f"{tag}.ln_2")))
        handles.append(block.mlp.register_forward_hook(grab(f"{tag}.mlp")))
        handles.append(block.register_forward_hook(grab(f"{tag}.block")))
    handles.append(tf.ln_f.register_forward_hook(grab("zz_ln_f")))


def _hooks_llama(model, grab, handles) -> None:
    m = model.model
    # Token embedding alone (no positions added; RoPE is applied inside attention).
    handles.append(m.embed_tokens.register_forward_hook(grab("00_embed")))
    for i, layer in enumerate(m.layers):
        tag = f"{i + 1:02d}_L{i:02d}"
        handles.append(layer.input_layernorm.register_forward_hook(grab(f"{tag}.ln_1")))
        handles.append(layer.self_attn.register_forward_hook(grab(f"{tag}.attn")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(grab(f"{tag}.ln_2")))
        handles.append(layer.mlp.register_forward_hook(grab(f"{tag}.mlp")))
        handles.append(layer.register_forward_hook(grab(f"{tag}.block")))
    handles.append(m.norm.register_forward_hook(grab("zz_ln_f")))


# arch -> (default model id, default out-dir, hook registrar)
_ARCH = {
    "gpt2": ("gpt2", "tests/reference_dumps", _hooks_gpt2),
    "llama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tests/reference_dumps_llama", _hooks_llama),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpt2", choices=["gpt2", "llama"])
    parser.add_argument("--model", default=None, help="HF model id (default: per --arch)")
    parser.add_argument("--out-dir", default=None, help="dump directory (default: per --arch)")
    parser.add_argument(
        "--prompt", default=FIXED_PROMPT, help="prompt to run (defaults to the frozen one)"
    )
    args = parser.parse_args()

    default_model, default_out, hook_fn = _ARCH[args.arch]
    model_id = args.model or default_model
    out_dir = args.out_dir or default_out

    try:
        import numpy as np
        import torch

        if args.arch == "gpt2":
            from transformers import GPT2LMHeadModel as LM
            from transformers import GPT2Tokenizer as Tok
        else:
            from transformers import AutoModelForCausalLM as LM
            from transformers import AutoTokenizer as Tok
    except ImportError:
        print(
            "torch/transformers not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = Tok.from_pretrained(model_id)
    # Force fp32 so the reference is a clean fp32 oracle. TinyLlama's checkpoint is
    # bf16; left as-is, the reference activations carry ~1% bf16 rounding error and
    # the layer diff never approaches 1e-4. The engine loads the same weights upcast
    # to fp32, so an fp32 reference is the correct apples-to-apples comparison.
    load_kwargs = {} if args.arch == "gpt2" else {"dtype": torch.float32}
    model = LM.from_pretrained(model_id, **load_kwargs).eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids[0].numpy().astype(np.int32))

    captured: dict[str, np.ndarray] = {}
    order: list[str] = []
    handles = []

    def grab(name: str):
        order.append(name)

        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().cpu().squeeze(0).float().numpy()

        return hook

    hook_fn(model, grab, handles)

    with torch.no_grad():
        out = model(input_ids)
    captured["zz_logits"] = out.logits.detach().cpu().squeeze(0).float().numpy()
    order.append("zz_logits")

    for h in handles:
        h.remove()

    for name in order:
        np.save(os.path.join(out_dir, f"{name}.npy"), captured[name])
    with open(os.path.join(out_dir, "order.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(order) + "\n")

    seq = input_ids.shape[1]
    print(
        f"[reference] arch={args.arch} prompt={args.prompt!r} ({seq} tokens) -> "
        f"{len(order)} tensors in {out_dir}"
    )
    print(f"[reference] logits shape: {tuple(captured['zz_logits'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
