#!/usr/bin/env python3
"""Generate tiny model + tokenizer fixtures for the C++ loader test.

numpy-only (no torch, no network), so it runs anywhere. It writes a small model
binary and tokenizer binary in the real format, which the C++ ``test_loader
--fixture <dir>`` then loads and checks. This is the cross-language proof that
the Python writer and the C++ reader agree on the byte layout.

The tiny config and the 4-token vocab below are mirrored as constants in
``tests/test_loader.cpp`` -- if you change one, change the other.

Model tensor values are set to value == flat float index (0, 1, 2, ...), an
exactly-representable pattern that lets the C++ side assert every float landed
at the right offset.

Usage:
    python tests/make_fixtures.py --out-dir build/fixtures
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import format_spec as fs  # noqa: E402

# Mirror of the constants in tests/test_loader.cpp.
TINY_CFG = {
    "n_layers": 2,
    "n_heads": 2,
    "d_model": 8,
    "head_dim": 4,
    "n_ctx": 16,
    "vocab_size": 32,
    "d_mlp": 16,
    "ln_eps": 1e-5,
}
# 'Ġ' (U+0120) is GPT-2's mapped space byte; it exercises the multi-byte
# codepoint -> raw byte reversal in decode. ids [0,1,2,3] decode to "Hi world!".
TINY_VOCAB = ["H", "i", "\u0120world", "!"]
EXPECTED_DECODE = "Hi world!"


def write_model(path: str) -> int:
    specs = fs.tensor_specs(TINY_CFG)
    total = sum(int(np.prod(shape)) for _, shape in specs)
    flat = np.arange(total, dtype="<f4")
    with open(path, "wb") as f:
        f.write(fs.pack_model_header(TINY_CFG))
        cursor = 0
        for _name, shape in specs:
            n = int(np.prod(shape))
            f.write(flat[cursor : cursor + n].tobytes())
            cursor += n
    return total


def write_tokenizer(path: str) -> None:
    with open(path, "wb") as f:
        f.write(fs.pack_tokenizer_header(len(TINY_VOCAB), 0))
        for token in TINY_VOCAB:
            fs.write_length_prefixed(f, token)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/fixtures", help="where to write fixtures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = os.path.join(args.out_dir, "model.bin")
    tok_path = os.path.join(args.out_dir, "tokenizer.bin")
    total = write_model(model_path)
    write_tokenizer(tok_path)
    print(f"[fixtures] model.bin: {total} floats -> {model_path}")
    print(f"[fixtures] tokenizer.bin: {len(TINY_VOCAB)} tokens -> {tok_path}")
    print(f"[fixtures] expected decode of [0,1,2,3]: {EXPECTED_DECODE!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
