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
"""

import argparse
import os
import sys

# Freeze ONE prompt for the whole project. Do not change it casually -- the
# committed reference dumps depend on it.
FIXED_PROMPT = "The quick brown fox"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument(
        "--out-dir", default="tests/reference_dumps", help="directory to write dumps into"
    )
    parser.add_argument(
        "--prompt", default=FIXED_PROMPT, help="prompt to run (defaults to the frozen one)"
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print(
            "torch/transformers not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    np.save(os.path.join(args.out_dir, "input_ids.npy"), input_ids[0].numpy().astype(np.int32))

    captured: dict[str, np.ndarray] = {}
    order: list[str] = []
    handles = []

    def grab(name: str):
        order.append(name)

        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().cpu().squeeze(0).float().numpy()

        return hook

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

    with torch.no_grad():
        out = model(input_ids)
    captured["zz_logits"] = out.logits.detach().cpu().squeeze(0).float().numpy()
    order.append("zz_logits")

    for h in handles:
        h.remove()

    for name in order:
        np.save(os.path.join(args.out_dir, f"{name}.npy"), captured[name])
    with open(os.path.join(args.out_dir, "order.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(order) + "\n")

    seq = input_ids.shape[1]
    print(
        f"[reference] prompt={args.prompt!r} ({seq} tokens) -> "
        f"{len(order)} tensors in {args.out_dir}"
    )
    print(f"[reference] logits shape: {tuple(captured['zz_logits'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
