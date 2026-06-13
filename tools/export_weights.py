#!/usr/bin/env python3
"""Export model weights to a flat binary the C++ engine can read.

Pulls GPT-2 (124M) from HuggingFace ``transformers`` and writes a single file:
a small header (magic + config) followed by raw fp32 tensors in a fixed order.
The C reader and this writer are the only two things that need to agree on the
layout, so keep it dead simple.

NOTE (skeleton): the tensor-writing loop is intentionally a TODO. GPT-2's linear
weights are stored as ``Conv1D`` with a transposed layout -- transpose them on
export or every matmul downstream is silently wrong. nanoGPT's loading code is
the reference for the exact ordering.
"""

import argparse
import sys

MAGIC = 0x47505432  # b"GPT2", written little-endian as the first 4 bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument(
        "--out", default="models/gpt2-124m.bin", help="output path for the flat binary"
    )
    args = parser.parse_args()

    try:
        from transformers import GPT2LMHeadModel  # noqa: F401
    except ImportError:
        print(
            "transformers is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    # TODO:
    #   1. model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    #   2. write header: MAGIC, n_layers, n_heads, d_model, n_ctx, vocab, d_mlp
    #   3. write each tensor as fp32 in a fixed order, transposing the Conv1D
    #      (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) weights on the way
    #      out. The token embedding (wte) is tied with the LM head, so it is
    #      written once.
    print(f"[export] would write {args.model} -> {args.out} (not yet implemented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
