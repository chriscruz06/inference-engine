#!/usr/bin/env python3
"""Write golden (string, GPT-2 token id) pairs for tests/test_encode.cpp.

Runs the HuggingFace GPT-2 tokenizer over a battery of strings and writes each as
one line: the space-separated ids, a tab, then the string (with backslash escapes
for tab, newline, and carriage-return). tests/test_encode then loads the engine's
own tokenizer.bin and asserts its encode() produces the same ids:

    test_encode --tokenizer models/tokenizer.bin --fixtures tests/encode_fixtures.txt

This is the encode analogue of tests/check_argmax.py: the engine's BPE encoder is
checked against the reference, not against itself. Run it once (it needs
transformers; gpt2 is already cached after the weight export). The C++ test needs
no Python or network at run time.

The battery is limited to cases that SHOULD match HF exactly: ASCII words,
whitespace runs, the full contraction set, punctuation, digits, and non-ASCII
letters (accented Latin, Cyrillic, CJK), which this engine classes as letters just
as HF's Unicode letter property does. Emoji and other non-ASCII symbols are left
out on purpose: the engine approximates their Unicode class, so its chunking (and
thus its ids) can differ from HF. Reversibility on those is covered instead by the
C++ round-trip test, where decode(encode(s)) == s holds regardless of chunking.

If a string here ever mismatches on your transformers version, move it out of this
list (the round-trip test still exercises it); do not loosen the test.

Usage:
    python tests/make_encode_fixtures.py --out tests/encode_fixtures.txt
"""

import argparse
import os
import sys

# Strings expected to encode identically to HuggingFace GPT-2.
STRINGS = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    " leading space",
    "trailing space ",
    "double  space  between",
    "   three leading",
    "tabs\tand\nnewlines\r\nhere",
    "don't can't I'm we'll they've he'd you're it's",
    "MixedCase CamelCase snake_case",
    "numbers 0 1 23 456 7890 and 3.14",
    'punctuation: (parens), [brackets]; "quotes" -- dashes!',
    "a b c d e f g",
    "supercalifragilisticexpialidocious",
    "café naïve façade résumé",
    "Москва Санкт-Петербург",
    "日本語のテキストです",
    "汉字 测试",
]


def escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument("--out", default="tests/encode_fixtures.txt", help="output path")
    args = parser.parse_args()

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        print(
            "transformers not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    tok = GPT2Tokenizer.from_pretrained(args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in STRINGS:
            ids = tok.encode(s)  # GPT-2 adds no special tokens
            f.write(" ".join(str(i) for i in ids) + "\t" + escape(s) + "\n")
    print(f"[encode-fixtures] wrote {len(STRINGS)} strings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())