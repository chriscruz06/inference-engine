#!/usr/bin/env python3
"""Check that engine and reference agree on the argmax token at every position.

The real correctness test for the forward pass: per-tensor max-abs-error
accumulates across layers (see check_layers.py), so zz_logits can exceed the
1e-4 threshold and still be correct. What matters is that greedy decoding picks
the SAME token. If argmax matches at every position, the forward pass is right.

Usage:
    python tests/check_argmax.py --reference tests/reference_dumps --engine tests/engine_dumps
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="tests/reference_dumps")
    parser.add_argument("--engine", default="tests/engine_dumps")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    ref_path = os.path.join(args.reference, "zz_logits.npy")
    eng_path = os.path.join(args.engine, "zz_logits.npy")
    if not os.path.exists(ref_path):
        print(f"no zz_logits.npy in {args.reference}; run tools/reference.py first.",
              file=sys.stderr)
        return 2
    if not os.path.exists(eng_path):
        print(f"no zz_logits.npy in {args.engine}; run the engine with IE_DUMP_DIR set.",
              file=sys.stderr)
        return 2

    r = np.load(ref_path)
    e = np.load(eng_path)
    if r.shape != e.shape:
        print(f"shape mismatch: reference {r.shape} vs engine {e.shape}", file=sys.stderr)
        return 1

    r_arg = r.argmax(-1)
    e_arg = e.argmax(-1)
    match = r_arg == e_arg

    print(f"max abs err (logits): {float(np.abs(r - e).max()):.3e}")
    print(f"positions matching:   {int(match.sum())}/{len(match)}")
    for i, (ra, ea, ok) in enumerate(zip(r_arg, e_arg, match)):
        print(f"  pos {i}: ref {int(ra):>6d}  engine {int(ea):>6d}  {'ok' if ok else 'MISMATCH'}")

    if match.all():
        print("all positions match. forward pass is correct.")
        return 0
    print("argmax mismatch: the forward pass has a real bug.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())