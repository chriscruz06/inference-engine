"""Canonical on-disk formats for the model and tokenizer binaries.

Both the Python writers (``export_weights.py``, ``convert_tokenizer.py``,
``tests/make_fixtures.py``) and the C++ readers (``src/model.cpp``,
``src/tokenizer.cpp``) must agree on these byte layouts. Keeping the spec in one
place is how the writer and reader avoid silent drift. If you change anything
here, change ``src/model.cpp`` / ``src/tokenizer.cpp`` to match.

------------------------------------------------------------------------------
MODEL BINARY  (.bin produced by export_weights.py)
------------------------------------------------------------------------------
All little-endian.

  bytes 0..3   : magic, the 4 ASCII bytes "GPT2"
  int32        : version (currently 1)
  int32        : n_layers
  int32        : n_heads
  int32        : d_model
  int32        : head_dim          (== d_model / n_heads)
  int32        : n_ctx
  int32        : vocab_size
  int32        : d_mlp
  float32      : ln_eps

Then raw float32 tensors, back to back, in the exact order returned by
``tensor_specs(cfg)`` below. No per-tensor headers: every shape is a function of
the config, so the reader recomputes sizes in the same order.

Linear weights are stored as [out_features, in_features] (row-major), so each
output row's weights are contiguous -- the layout the SIMD GEMM wants later.
GPT-2's HuggingFace weights are Conv1D with the transposed [in, out] layout, so
export_weights.py transposes the four projection matrices on the way out.

------------------------------------------------------------------------------
TOKENIZER BINARY  (.bin produced by convert_tokenizer.py)
------------------------------------------------------------------------------
All little-endian.

  bytes 0..3   : magic, the 4 ASCII bytes "TOK1"
  int32        : version (currently 1)
  int32        : n_vocab
  int32        : n_merges
  n_vocab times, in id order (id 0, 1, 2, ...):
      int32    : byte length L
      L bytes  : the token string, UTF-8, in GPT-2's byte->unicode space
  n_merges times, in rank order (rank 0 first):
      int32 La : a-side length
      La bytes : a-side string (UTF-8, byte->unicode space)
      int32 Lb : b-side length
      Lb bytes : b-side string

decode only needs the vocab strings plus the byte<->unicode map (rebuilt in C++
from the same rule as Python's bytes_to_unicode). The merges are for encode.
"""

import struct

MODEL_MAGIC = b"GPT2"
MODEL_VERSION = 1

TOKENIZER_MAGIC = b"TOK1"
TOKENIZER_VERSION = 1


def pack_model_header(cfg: dict) -> bytes:
    """Pack the 40-byte model header from a config dict."""
    return MODEL_MAGIC + struct.pack(
        "<iiiiiiiif",
        MODEL_VERSION,
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["d_model"],
        cfg["head_dim"],
        cfg["n_ctx"],
        cfg["vocab_size"],
        cfg["d_mlp"],
        cfg["ln_eps"],
    )


def tensor_specs(cfg: dict):
    """Ordered list of (name, shape) for every tensor in the model binary.

    This order IS the file order and the C++ read order. Do not reorder.
    """
    v, d, m, lyr, ctx = (
        cfg["vocab_size"],
        cfg["d_model"],
        cfg["d_mlp"],
        cfg["n_layers"],
        cfg["n_ctx"],
    )
    specs = [("wte", (v, d)), ("wpe", (ctx, d))]
    for i in range(lyr):
        p = f"h.{i}."
        specs += [
            (p + "ln_1.w", (d,)),
            (p + "ln_1.b", (d,)),
            (p + "attn.c_attn.w", (3 * d, d)),
            (p + "attn.c_attn.b", (3 * d,)),
            (p + "attn.c_proj.w", (d, d)),
            (p + "attn.c_proj.b", (d,)),
            (p + "ln_2.w", (d,)),
            (p + "ln_2.b", (d,)),
            (p + "mlp.c_fc.w", (m, d)),
            (p + "mlp.c_fc.b", (m,)),
            (p + "mlp.c_proj.w", (d, m)),
            (p + "mlp.c_proj.b", (d,)),
        ]
    specs += [("ln_f.w", (d,)), ("ln_f.b", (d,))]
    return specs


def pack_tokenizer_header(n_vocab: int, n_merges: int) -> bytes:
    return TOKENIZER_MAGIC + struct.pack("<iii", TOKENIZER_VERSION, n_vocab, n_merges)


def write_length_prefixed(f, s: str) -> None:
    """Write a string as int32 length + UTF-8 bytes."""
    data = s.encode("utf-8")
    f.write(struct.pack("<i", len(data)))
    f.write(data)


def bytes_to_unicode() -> dict:
    """GPT-2's reversible byte -> unicode-codepoint map.

    Maps each of the 256 byte values to a printable unicode codepoint, so token
    strings never contain control characters or whitespace. This is the exact
    function from the GPT-2 / nanoGPT tokenizer; the C++ side rebuilds the same
    map to reverse it during decode.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
