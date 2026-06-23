#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ie {

// Tokenizer for the flat tokenizer.bin produced by tools/convert_tokenizer.py, so
// there is no JSON parsing here. Two formats are supported, picked by magic:
//   "TOK1" -- GPT-2 byte-level BPE (decode reverses the byte<->unicode map).
//   "TOK2" -- Llama SentencePiece, decode-only: each id stores its precomputed
//             emit bytes, so decode is a concatenation (+ a leading-space strip).
class Tokenizer {
   public:
    // Load a tokenizer.bin (magic "TOK1" or "TOK2"). Returns false (with a
    // message on stderr) on any error.
    bool load(const std::string& tokenizer_bin);

    // Text -> token ids (GPT-2 byte-level BPE): pre-tokenize with the GPT-2
    // pattern, map each chunk's bytes through byte->unicode, then run the
    // rank-ordered merge loop and look each merged symbol up in the vocab. TOK1
    // only; SentencePiece (TOK2) encode is not implemented and returns empty.
    std::vector<int> encode(const std::string& text) const;

    // Token ids -> text. TOK1: reverse GPT-2's byte->unicode map back to UTF-8.
    // TOK2: concatenate the per-id emit bytes, dropping a single leading space
    // when the sequence began at BOS (matches HF LlamaTokenizer.decode).
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool loaded() const { return !id_to_token_.empty(); }
    // SentencePiece end-of-sequence id, or -1 (TOK1 / not loaded). Lets the
    // generation loop stop when the model emits EOS.
    int eos_id() const { return eos_id_; }

   private:
    void build_byte_maps();
    // BPE merge loop: repeatedly merge the lowest-rank adjacent pair in `word`
    // (a list of byte->unicode symbols) until no ranked pair remains. In place.
    void bpe(std::vector<std::string>& word) const;

    std::vector<std::string>
        id_to_token_;  // id -> token string (TOK1: mapped space; TOK2: emit bytes)
    std::vector<std::pair<std::string, std::string>> merges_;  // rank-ordered (for encode)

    bool sp_ = false;                // true when loaded from a SentencePiece (TOK2) tokenizer
    int bos_id_ = -1, eos_id_ = -1;  // SentencePiece special ids (TOK2 only)

    std::array<uint32_t, 256> byte_to_cp_{};            // byte -> unicode codepoint (TOK1)
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;  // and the reverse (TOK1)

    // Encode-side tables (TOK1), built in load().
    std::array<std::string, 256> byte_to_sym_{};        // byte -> its mapped UTF-8 symbol
    std::unordered_map<std::string, int> token_to_id_;  // vocab string -> id
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;  // merge pair -> rank
};

}  // namespace ie
