#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ie {

// Byte-level BPE tokenizer (GPT-2 style). Reads the flat tokenizer.bin produced
// by tools/convert_tokenizer.py, so there is no JSON parsing here.
class Tokenizer {
public:
    // Load the vocab strings and merge rules from a tokenizer.bin (magic
    // "TOK1"). Returns false (with a message on stderr) on any error.
    bool load(const std::string& tokenizer_bin);

    // Text -> token ids. Build this SECOND; left as a stub for now. During
    // early forward-pass bringup you feed reference token ids directly and skip
    // the encoder entirely (see tools/reference.py's input_ids.npy).
    std::vector<int> encode(const std::string& text) const;

    // Token ids -> text. Each token string lives in GPT-2's byte->unicode
    // space; decode reverses that map back to the original UTF-8 bytes. This is
    // the nearly-trivial half, built first so you can read model output early.
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool loaded() const { return !id_to_token_.empty(); }

private:
    void build_byte_maps();

    std::vector<std::string> id_to_token_;  // id -> token string (UTF-8, mapped space)
    std::vector<std::pair<std::string, std::string>> merges_;  // rank-ordered (for encode)

    std::array<uint32_t, 256> byte_to_cp_{};         // byte -> unicode codepoint
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;  // and the reverse
};

}  // namespace ie
