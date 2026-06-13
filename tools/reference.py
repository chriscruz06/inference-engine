#!/usr/bin/env python3
"""Dump every intermediate activation from the reference model.

This is the verification harness, and the single most valuable tool in the
project. It runs the reference implementation (HuggingFace ``transformers``) on
one FIXED prompt with greedy decoding and writes every intermediate tensor to
disk: the embeddings, each layer's post-LN / post-attention / post-MLP outputs,
and the final logits.

The C++ engine dumps the same tensors for the same input, and
``tests/check_layers.py`` diffs the two and reports the *first* layer where they
diverge. That turns a three-day "stare at twelve layers of code" bug into a
twenty-minute "layer 4's attention output is wrong, look there" fix.

NOTE (skeleton): activation capture via forward hooks is the TODO below.
"""

import argparse
import sys

# Freeze ONE prompt for the whole project. Do not change it casually -- the
# committed reference dumps depend on it.
FIXED_PROMPT = "The quick brown fox"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument(
        "--out-dir",
        default="tests/reference_dumps",
        help="directory to write the activation dumps into",
    )
    parser.add_argument(
        "--prompt", default=FIXED_PROMPT, help="prompt to run (defaults to the frozen one)"
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: F401
    except ImportError:
        print(
            "torch/transformers not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    # TODO:
    #   1. load model + tokenizer; model.eval(); run under torch.no_grad()
    #   2. register forward hooks on each LayerNorm / attention / MLP submodule
    #   3. run `args.prompt`, collect the captured activations
    #   4. write each tensor to args.out_dir (e.g. .npy) under a stable name
    #      that encodes forward order, plus the final logits
    print(
        f"[reference] would dump activations for {args.prompt!r} -> "
        f"{args.out_dir} (not yet implemented)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
