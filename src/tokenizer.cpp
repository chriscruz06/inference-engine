#include "tokenizer.hpp"

#include <cstdio>

namespace ie {

bool Tokenizer::load(const std::string& encoder_json, const std::string& vocab_bpe) {
    // TODO: parse encoder.json (token->id) and vocab.bpe (merge rules), and
    // build the byte->unicode mapping (the fiddly part of GPT-2 BPE).
    (void)encoder_json;
    (void)vocab_bpe;
    std::fprintf(stderr, "[tokenizer] load: not yet implemented\n");
    return false;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // TODO: byte-level BPE merge loop using the merge-rank table.
    (void)text;
    return {};
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    // TODO: id -> token string -> bytes (reverse the byte->unicode map).
    (void)ids;
    return {};
}

}  // namespace ie
