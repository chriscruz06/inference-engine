#pragma once

#include <string>
#include <vector>

namespace ie {

// Byte-level BPE tokenizer (GPT-2 style: encoder.json + vocab.bpe).
class Tokenizer {
public:
    // Load token->id table and merge rules. (Skeleton: not yet implemented.)
    bool load(const std::string& encoder_json, const std::string& vocab_bpe);

    // Text -> token ids. (Skeleton.) Build this second.
    std::vector<int> encode(const std::string& text) const;

    // Token ids -> text. Build this FIRST: it is nearly trivial and lets you
    // read model output before `encode` exists. During early forward-pass
    // bringup you can feed token IDs directly and skip the encoder entirely.
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    std::vector<std::string> id_to_token_;
    // TODO: token->id map, byte<->unicode tables, merge-rank table.
};

}  // namespace ie
