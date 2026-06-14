#!/usr/bin/env python3
"""Diff the engine's activation dumps against the reference dumps.

Reports, per tensor, the max absolute error, and -- crucially -- the FIRST
tensor in forward order whose error exceeds the threshold. That tensor is the
one to go look at.

Forward order comes from the reference dir's ``order.txt`` (written by
``tools/reference.py``). For each name it loads ``<reference>/<name>.npy`` and
``<engine>/<name>.npy``. Tensors the engine has not dumped yet are reported as
"(missing)" and skipped, so this is useful even mid-bringup.

Logits should match to roughly 1e-4 max abs error; if the greedy-decoded token
matches the reference at every step, the forward pass is correct.

Usage:
    python tests/check_layers.py --reference tests/reference_dumps --engine tests/engine_dumps
"""

import argparse
import os
import sys

THRESHOLD = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", default="tests/reference_dumps", help="dir of reference dumps"
    )
    parser.add_argument("--engine", default="tests/engine_dumps", help="dir of engine dumps")
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD, help="max abs error to flag a divergence"
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    order_path = os.path.join(args.reference, "order.txt")
    if not os.path.exists(order_path):
        print(f"no order.txt in {args.reference}; run tools/reference.py first.", file=sys.stderr)
        return 2
    with open(order_path, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    print(f"{'tensor':<22s} {'shape':<16s} {'max_abs_err':>12s}  status")
    print("-" * 64)

    first_bad = None
    n_compared = 0
    n_missing = 0
    for name in names:
        ref_file = os.path.join(args.reference, f"{name}.npy")
        eng_file = os.path.join(args.engine, f"{name}.npy")
        if not os.path.exists(ref_file):
            continue
        ref = np.load(ref_file)
        if not os.path.exists(eng_file):
            print(f"{name:<22s} {str(ref.shape):<16s} {'-':>12s}  (missing)")
            n_missing += 1
            continue
        eng = np.load(eng_file)
        if ref.shape != eng.shape:
            print(f"{name:<22s} {str(ref.shape):<16s} {'-':>12s}  SHAPE {eng.shape}")
            first_bad = first_bad or name
            continue
        err = float(np.max(np.abs(ref.astype(np.float64) - eng.astype(np.float64))))
        n_compared += 1
        flag = "ok" if err <= args.threshold else "DIVERGES"
        if err > args.threshold and first_bad is None:
            first_bad = name
        print(f"{name:<22s} {str(ref.shape):<16s} {err:>12.3e}  {flag}")

    print("-" * 64)
    if n_compared == 0:
        print(f"nothing compared yet ({n_missing} reference tensors not dumped by the engine).")
        return 0
    if first_bad is not None:
        print(f"FIRST DIVERGENCE: {first_bad}  (threshold {args.threshold:g})")
        print("Look at the code producing that tensor; everything before it matches.")
        return 1
    print(f"all {n_compared} compared tensors within {args.threshold:g}. forward pass matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
