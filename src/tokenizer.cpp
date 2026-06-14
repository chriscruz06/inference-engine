#include "tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace ie {
namespace {

bool read_exact(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

bool read_i32(std::ifstream& in, int32_t& v) { return read_exact(in, &v, sizeof(v)); }

// Read an int32 length followed by that many raw bytes into `out`.
bool read_string(std::ifstream& in, std::string& out) {
    int32_t len = 0;
    if (!read_i32(in, len) || len < 0) return false;
    out.resize(static_cast<std::size_t>(len));
    return len == 0 || read_exact(in, out.data(), static_cast<std::size_t>(len));
}

bool fail(const char* msg) {
    std::fprintf(stderr, "[tokenizer] %s\n", msg);
    return false;
}

// Decode one UTF-8 codepoint from s starting at i. Returns bytes consumed and
// writes the codepoint to cp. On a malformed lead byte, consumes 1 byte and
// returns the raw byte as the codepoint (defensive; valid input never hits it).
std::size_t utf8_next(const std::string& s, std::size_t i, uint32_t& cp) {
    const auto c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0;
        return 1;
    }
    std::size_t n;
    if ((c0 & 0xE0) == 0xC0) {
        cp = c0 & 0x1F;
        n = 2;
    } else if ((c0 & 0xF0) == 0xE0) {
        cp = c0 & 0x0F;
        n = 3;
    } else if ((c0 & 0xF8) == 0xF0) {
        cp = c0 & 0x07;
        n = 4;
    } else {
        cp = c0;
        return 1;
    }
    if (i + n > s.size()) {
        cp = c0;
        return 1;
    }
    for (std::size_t k = 1; k < n; ++k) {
        const auto ck = static_cast<unsigned char>(s[i + k]);
        if ((ck & 0xC0) != 0x80) {
            cp = c0;
            return 1;
        }
        cp = (cp << 6) | (ck & 0x3F);
    }
    return n;
}

}  // namespace

void Tokenizer::build_byte_maps() {
    // The exact GPT-2 bytes_to_unicode rule: printable ASCII and two Latin-1
    // ranges map to themselves; the remaining bytes map to 256, 257, ...
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    cp_to_byte_.clear();
    cp_to_byte_.reserve(256);
    for (std::size_t i = 0; i < bs.size(); ++i) {
        byte_to_cp_[static_cast<std::size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
        cp_to_byte_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

bool Tokenizer::load(const std::string& tokenizer_bin) {
    std::ifstream in(tokenizer_bin, std::ios::binary);
    if (!in) return fail("could not open tokenizer file");

    char magic[4];
    if (!read_exact(in, magic, 4)) return fail("truncated header (magic)");
    if (std::memcmp(magic, "TOK1", 4) != 0) return fail("bad magic (expected \"TOK1\")");

    int32_t version = 0, n_vocab = 0, n_merges = 0;
    if (!read_i32(in, version) || !read_i32(in, n_vocab) || !read_i32(in, n_merges)) {
        return fail("truncated header (fields)");
    }
    if (version != 1) return fail("unsupported tokenizer version");
    if (n_vocab <= 0) return fail("non-positive vocab size");

    id_to_token_.assign(static_cast<std::size_t>(n_vocab), std::string());
    for (int i = 0; i < n_vocab; ++i) {
        if (!read_string(in, id_to_token_[static_cast<std::size_t>(i)])) {
            return fail("truncated vocab strings");
        }
    }
    merges_.assign(static_cast<std::size_t>(std::max(0, n_merges)),
                   std::pair<std::string, std::string>());
    for (int i = 0; i < n_merges; ++i) {
        auto& m = merges_[static_cast<std::size_t>(i)];
        if (!read_string(in, m.first) || !read_string(in, m.second)) {
            return fail("truncated merge rules");
        }
    }

    build_byte_maps();
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // TODO (Phase 2+): GPT-2 pre-tokenization regex, then the BPE merge loop
    // over merges_ ranks. Until then, feed reference token ids directly.
    (void)text;
    std::fprintf(stderr, "[tokenizer] encode: not yet implemented (decode-first)\n");
    return {};
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
        std::size_t i = 0;
        while (i < tok.size()) {
            uint32_t cp = 0;
            i += utf8_next(tok, i, cp);
            auto it = cp_to_byte_.find(cp);
            if (it != cp_to_byte_.end()) out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

}  // namespace ie
