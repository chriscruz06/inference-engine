#!/usr/bin/env python3
"""Convert GPT-2's vocab + merge rules into a flat tokenizer.bin.

The C++ engine has no JSON parser by design, so the fiddly parsing happens here
once and the result is written in the dead-simple ``format_spec`` tokenizer
layout that C++ can read with a few ``read`` calls.

Two ways to get the data:
  --from-hf gpt2     download (or reuse the cache of) the model repo's
                     vocab.json + merges.txt and parse those. Stable across
                     transformers versions, since it never touches the
                     tokenizer object's shifting internals.
  --encoder/--vocab  point at a local vocab.json/encoder.json + merges.txt pair.

Usage:
    python tools/convert_tokenizer.py --from-hf gpt2 --out models/tokenizer.bin
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_spec as fs  # noqa: E402


def parse_merges(lines):
    """Each non-comment, non-empty line is 'a b'; rank == line order."""
    merges = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        merges.append((parts[0], parts[1]))
    return merges


def load_from_files(encoder_path: str, vocab_path: str):
    """Return (encoder dict, merges list) from a local vocab.json + merges.txt."""
    with open(encoder_path, encoding="utf-8") as f:
        encoder = json.load(f)  # token string -> id
    with open(vocab_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    return encoder, parse_merges(lines)


def load_from_hf(model_id: str):
    """Return (encoder dict, merges list) from the model repo's canonical files.

    Pulls vocab.json + merges.txt straight from the hub (already cached after
    the first download) and parses them. This sidesteps every tokenizer-internals
    and save_vocabulary API change across transformers versions.
    """
    from huggingface_hub import hf_hub_download

    vocab_path = hf_hub_download(repo_id=model_id, filename="vocab.json")
    merges_path = hf_hub_download(repo_id=model_id, filename="merges.txt")
    return load_from_files(vocab_path, merges_path)


def write_tokenizer(path: str, encoder: dict, merges) -> None:
    n_vocab = len(encoder)
    # encoder maps token -> id; invert to an id-ordered list.
    id_to_token = [None] * n_vocab
    for token, idx in encoder.items():
        if idx < 0 or idx >= n_vocab:
            raise ValueError(f"id {idx} out of range for vocab size {n_vocab}")
        id_to_token[idx] = token
    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        raise ValueError(f"vocab has gaps at ids {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(fs.pack_tokenizer_header(n_vocab, len(merges)))
        for token in id_to_token:
            fs.write_length_prefixed(f, token)
        for a, b in merges:
            fs.write_length_prefixed(f, a)
            fs.write_length_prefixed(f, b)
    print(f"[tokenizer] wrote {path}: {n_vocab} tokens, {len(merges)} merges")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", help="path to vocab.json / encoder.json")
    parser.add_argument("--vocab", help="path to merges.txt / vocab.bpe")
    parser.add_argument("--from-hf", help="HF model id to pull vocab.json + merges.txt from")
    parser.add_argument("--out", default="models/tokenizer.bin", help="output path")
    args = parser.parse_args()

    if args.from_hf:
        try:
            encoder, merges = load_from_hf(args.from_hf)
        except ImportError:
            print(
                "huggingface_hub/transformers not available; use --encoder/--vocab instead.",
                file=sys.stderr,
            )
            return 1
    elif args.encoder and args.vocab:
        encoder, merges = load_from_files(args.encoder, args.vocab)
    else:
        print("provide either --from-hf, or both --encoder and --vocab", file=sys.stderr)
        return 2

    write_tokenizer(args.out, encoder, merges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
