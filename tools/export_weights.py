#!/usr/bin/env python3
"""Export GPT-2 weights to the flat binary the C++ engine reads.

Pulls GPT-2 (124M by default) from HuggingFace ``transformers`` and writes a
single file: the header from ``format_spec`` followed by raw fp32 tensors in the
canonical order. The C++ reader and this writer agree only through
``tools/format_spec.py`` -- keep them in sync there.

The one trap: GPT-2's linear weights are stored as ``Conv1D`` with a transposed
[in, out] layout. They are transposed here to [out, in] so every matmul
downstream is correct. nanoGPT's loader is the reference for the exact set.

Usage:
    pip install -r requirements.txt
    python tools/export_weights.py --model gpt2 --out models/gpt2-124m.bin
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_spec as fs  # noqa: E402  (after sys.path tweak)

# HF Conv1D stores weight as [in, out]; we want [out, in]. These are the keys
# (relative to each block, plus a layer index) whose .weight must be transposed.
_CONV1D_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _to_numpy(t) -> np.ndarray:
    return t.detach().cpu().to(dtype=__import__("torch").float32).numpy()


def build_tensor_dict(model) -> dict:
    """Map canonical tensor names -> contiguous fp32 numpy arrays.

    Handles the Conv1D transpose and ties the LM head to wte (written once).
    """
    sd = model.state_dict()
    cfg = model.config
    n_layers = cfg.n_layer
    out: dict[str, np.ndarray] = {}

    out["wte"] = _to_numpy(sd["transformer.wte.weight"])
    out["wpe"] = _to_numpy(sd["transformer.wpe.weight"])

    for i in range(n_layers):
        src = f"transformer.h.{i}."
        dst = f"h.{i}."
        # LayerNorms and biases: copied straight through.
        out[dst + "ln_1.w"] = _to_numpy(sd[src + "ln_1.weight"])
        out[dst + "ln_1.b"] = _to_numpy(sd[src + "ln_1.bias"])
        out[dst + "ln_2.w"] = _to_numpy(sd[src + "ln_2.weight"])
        out[dst + "ln_2.b"] = _to_numpy(sd[src + "ln_2.bias"])
        out[dst + "attn.c_attn.b"] = _to_numpy(sd[src + "attn.c_attn.bias"])
        out[dst + "attn.c_proj.b"] = _to_numpy(sd[src + "attn.c_proj.bias"])
        out[dst + "mlp.c_fc.b"] = _to_numpy(sd[src + "mlp.c_fc.bias"])
        out[dst + "mlp.c_proj.b"] = _to_numpy(sd[src + "mlp.c_proj.bias"])
        # Conv1D weights: transpose [in, out] -> [out, in].
        out[dst + "attn.c_attn.w"] = _to_numpy(sd[src + "attn.c_attn.weight"]).T
        out[dst + "attn.c_proj.w"] = _to_numpy(sd[src + "attn.c_proj.weight"]).T
        out[dst + "mlp.c_fc.w"] = _to_numpy(sd[src + "mlp.c_fc.weight"]).T
        out[dst + "mlp.c_proj.w"] = _to_numpy(sd[src + "mlp.c_proj.weight"]).T

    out["ln_f.w"] = _to_numpy(sd["transformer.ln_f.weight"])
    out["ln_f.b"] = _to_numpy(sd["transformer.ln_f.bias"])
    return out


def write_binary(path: str, cfg: dict, tensors: dict) -> int:
    """Write header + tensors in canonical order. Returns total floats written."""
    specs = fs.tensor_specs(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    total = 0
    with open(path, "wb") as f:
        f.write(fs.pack_model_header(cfg))
        offset_floats = 0
        for name, shape in specs:
            arr = tensors[name]
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(
                    f"{name}: shape {tuple(arr.shape)} != expected {tuple(shape)}"
                )
            arr = np.ascontiguousarray(arr, dtype="<f4")
            f.write(arr.tobytes())
            count = arr.size
            print(f"  {offset_floats:>11d}  {name:<22s} {tuple(shape)}")
            offset_floats += count
            total += count
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument(
        "--out", default="models/gpt2-124m.bin", help="output path for the flat binary"
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel
    except ImportError:
        print(
            "torch/transformers not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"[export] loading {args.model} from HuggingFace ...")
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    hf = model.config
    cfg = {
        "n_layers": hf.n_layer,
        "n_heads": hf.n_head,
        "d_model": hf.n_embd,
        "head_dim": hf.n_embd // hf.n_head,
        "n_ctx": hf.n_positions,
        "vocab_size": hf.vocab_size,
        "d_mlp": hf.n_inner if getattr(hf, "n_inner", None) else 4 * hf.n_embd,
        "ln_eps": float(hf.layer_norm_epsilon),
    }
    print(
        f"[export] config: {cfg['n_layers']}L {cfg['n_heads']}H "
        f"d_model={cfg['d_model']} d_mlp={cfg['d_mlp']} "
        f"vocab={cfg['vocab_size']} ctx={cfg['n_ctx']} eps={cfg['ln_eps']}"
    )

    tensors = build_tensor_dict(model)
    print(f"[export] writing {args.out} ...")
    total = write_binary(args.out, cfg, tensors)
    size_mb = (40 + total * 4) / 1e6
    print(f"[export] done: {total} floats, {size_mb:.1f} MB at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
