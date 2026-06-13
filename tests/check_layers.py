#!/usr/bin/env python3
"""Diff the engine's activation dumps against the reference dumps.

Reports, per tensor, the max absolute error, and -- crucially -- the FIRST
tensor in forward order whose error exceeds a threshold. That tensor is the
layer to go look at.

NOTE (skeleton): loading + comparison is the TODO below. Logits should match the
reference to roughly 1e-4 max abs error; if the greedy-decoded token matches the
reference at every step, the forward pass is correct.
"""

import argparse
import sys

THRESHOLD = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default="tests/reference_dumps",
        help="directory of reference dumps (from tools/reference.py)",
    )
    parser.add_argument(
        "--engine",
        default="tests/engine_dumps",
        help="directory of dumps emitted by the C++ engine",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD, help="max abs error to flag as a divergence"
    )
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("numpy not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    # TODO:
    #   1. list the tensors present in both directories, sorted in forward order
    #   2. for each, load both arrays, assert matching shape, compute max abs err
    #   3. print a table (tensor name, max abs err); flag and return non-zero on
    #      the first tensor exceeding args.threshold
    print(
        f"[check] would diff {args.engine} against {args.reference} "
        f"(threshold {args.threshold}, not yet implemented)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
