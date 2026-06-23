#include "tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace ie {
namespace {

bool read_exact(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

bool read_i32(std::ifstream& in, int32_t& v) {
    return read_exact(in, &v, sizeof(v));
}

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

// Encode one codepoint as UTF-8. GPT-2's mapped codepoints are all < 0x800 (two
// bytes), but three-byte output is handled for safety.
std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

// Character classes for the GPT-2 pre-tokenization pattern. \p{L}/\p{N} are
// Unicode properties; ASCII is classified exactly and non-ASCII (non-space) is
// approximated as a letter, so accented words and CJK group as one chunk
// (matching GPT-2 for the common case; rare non-ASCII symbols/digits may split
// differently). The byte-level BPE still encodes any bytes correctly.
bool is_space_cp(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v')
        return true;
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    return cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80) return false;
    return !is_space_cp(cp);  // non-ASCII non-space -> treat as a letter
}
bool is_number_cp(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

// 0 = letter, 1 = number, 2 = other (for a content codepoint).
int class_of(uint32_t cp) {
    if (is_letter_cp(cp)) return 0;
    if (is_number_cp(cp)) return 1;
    return 2;
}

// GPT-2 pre-tokenization. Splits `s` into byte ranges [start,end) following
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// as a hand-rolled greedy scanner over UTF-8 codepoints (the project has no regex
// engine with Unicode-property support, and std::regex cannot do \p{L}).
std::vector<std::pair<std::size_t, std::size_t>> pretokenize(const std::string& s) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        const std::size_t len = utf8_next(s, i, cp);

        // 1) contractions: ASCII apostrophe + a lowercase suffix.
        if (cp == '\'') {
            const char c1 = (i + 1 < n) ? s[i + 1] : 0;
            const char c2 = (i + 2 < n) ? s[i + 2] : 0;
            std::size_t clen = 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                clen = 2;
            else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                     (c1 == 'l' && c2 == 'l'))
                clen = 3;
            if (clen) {
                out.emplace_back(i, i + clen);
                i += clen;
                continue;
            }
        }

        // 2-4) ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+': one optional leading
        // literal space, then a maximal run of one character class.
        const bool lead_space = (cp == ' ');
        const std::size_t content = lead_space ? i + len : i;
        if (content < n) {
            uint32_t ccp = 0;
            const std::size_t clen = utf8_next(s, content, ccp);
            if (!is_space_cp(ccp)) {
                const int cls = class_of(ccp);
                std::size_t j = content + clen;
                while (j < n) {
                    uint32_t ncp = 0;
                    const std::size_t nlen = utf8_next(s, j, ncp);
                    if (is_space_cp(ncp) || class_of(ncp) != cls) break;
                    j += nlen;
                }
                out.emplace_back(i, j);
                i = j;
                continue;
            }
        }

        // 5-6) whitespace. Consume the maximal run; if it is followed by a
        // non-space, leave its LAST codepoint for the next ' ?X+' to attach to
        // (the effect of '\s+(?!\S)' sitting before ' ?\p{L}+').
        std::size_t k = i;
        std::size_t last = i;  // start of the final codepoint in the run
        while (k < n) {
            uint32_t wcp = 0;
            const std::size_t wlen = utf8_next(s, k, wcp);
            if (!is_space_cp(wcp)) break;
            last = k;
            k += wlen;
        }
        if (k < n && last > i) {
            out.emplace_back(i, last);  // all but the last whitespace codepoint
            i = last;
        } else {
            out.emplace_back(i, k);  // run reaches EOS, or is a single codepoint
            i = k;
        }
    }
    return out;
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
    // The encode side needs each byte's mapped symbol as a UTF-8 string.
    for (int b = 0; b < 256; ++b)
        byte_to_sym_[static_cast<std::size_t>(b)] =
            utf8_encode(byte_to_cp_[static_cast<std::size_t>(b)]);
}

bool Tokenizer::load(const std::string& tokenizer_bin) {
    std::ifstream in(tokenizer_bin, std::ios::binary);
    if (!in) return fail("could not open tokenizer file");

    char magic[4];
    if (!read_exact(in, magic, 4)) return fail("truncated header (magic)");
    const bool tok2 = std::memcmp(magic, "TOK2", 4) == 0;
    if (!tok2 && std::memcmp(magic, "TOK1", 4) != 0) {
        return fail("bad magic (expected \"TOK1\" or \"TOK2\")");
    }

    int32_t version = 0;
    if (!read_i32(in, version)) return fail("truncated header (fields)");
    if (version != 1) return fail("unsupported tokenizer version");

    if (tok2) {
        // SentencePiece (Llama): each id stores its precomputed decode emit bytes.
        int32_t n_vocab = 0, bos = 0, eos = 0;
        if (!read_i32(in, n_vocab) || !read_i32(in, bos) || !read_i32(in, eos)) {
            return fail("truncated header (fields)");
        }
        if (n_vocab <= 0) return fail("non-positive vocab size");
        id_to_token_.assign(static_cast<std::size_t>(n_vocab), std::string());
        for (int i = 0; i < n_vocab; ++i) {
            if (!read_string(in, id_to_token_[static_cast<std::size_t>(i)])) {
                return fail("truncated vocab strings");
            }
        }
        sp_ = true;
        bos_id_ = bos;
        eos_id_ = eos;
        return true;
    }

    // GPT-2 byte-level BPE.
    int32_t n_vocab = 0, n_merges = 0;
    if (!read_i32(in, n_vocab) || !read_i32(in, n_merges)) {
        return fail("truncated header (fields)");
    }
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

    // Encode-side lookups: vocab string -> id, and merge pair -> rank.
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size());
    for (std::size_t i = 0; i < id_to_token_.size(); ++i)
        token_to_id_.emplace(id_to_token_[i], static_cast<int>(i));
    bpe_ranks_.clear();
    for (std::size_t r = 0; r < merges_.size(); ++r)
        bpe_ranks_.emplace(merges_[r], static_cast<int>(r));

    return true;
}

void Tokenizer::bpe(std::vector<std::string>& word) const {
    while (word.size() >= 2) {
        // Find the adjacent pair with the lowest merge rank.
        int best_rank = INT_MAX;
        std::string first, second;
        bool found = false;
        for (std::size_t j = 0; j + 1 < word.size(); ++j) {
            const auto it = bpe_ranks_.find({word[j], word[j + 1]});
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                first = word[j];
                second = word[j + 1];
                found = true;
            }
        }
        if (!found) break;

        // Merge every non-overlapping occurrence of that pair, left to right.
        std::vector<std::string> merged;
        merged.reserve(word.size());
        std::size_t k = 0;
        while (k < word.size()) {
            if (k + 1 < word.size() && word[k] == first && word[k + 1] == second) {
                merged.push_back(first + second);
                k += 2;
            } else {
                merged.push_back(word[k]);
                k += 1;
            }
        }
        word.swap(merged);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (sp_) {
        std::fprintf(stderr, "[tokenizer] encode: SentencePiece (Llama) encode not implemented\n");
        return {};
    }
    if (id_to_token_.empty()) return {};

    std::vector<int> ids;
    for (const auto& chunk : pretokenize(text)) {
        // Map each byte of the chunk to its unicode-space symbol, then BPE-merge.
        std::vector<std::string> word;
        word.reserve(chunk.second - chunk.first);
        for (std::size_t b = chunk.first; b < chunk.second; ++b)
            word.push_back(byte_to_sym_[static_cast<unsigned char>(text[b])]);
        bpe(word);
        for (const std::string& sym : word) {
            const auto it = token_to_id_.find(sym);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
            else
                std::fprintf(stderr, "[tokenizer] encode: symbol not in vocab (skipped)\n");
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;

    if (sp_) {
        // SentencePiece: emit bytes are precomputed per id, so just concatenate.
        for (int id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) continue;
            out += id_to_token_[static_cast<std::size_t>(id)];
        }
        // SentencePiece prepends a space to the first piece; drop it once when the
        // sequence began at BOS (matches HF LlamaTokenizer.decode). Incremental
        // single-token decode during generation never starts with BOS, so the
        // inter-word spaces are left intact.
        if (!ids.empty() && ids.front() == bos_id_ && !out.empty() && out.front() == ' ') {
            out.erase(out.begin());
        }
        return out;
    }

    // GPT-2 byte-level BPE: reverse the byte->unicode map back to raw bytes.
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
