// tests/test_encode.cpp
//
// GPT-2 byte-level BPE *encode* tests. Two modes, mirroring test_loader:
//
//   (no args)  HERMETIC: build a tiny TOK1 tokenizer in memory (all 256 byte
//              symbols, plus two merge rules), load it, and check encode's core
//              properties without any real GPT-2 or Python:
//                * decode(encode(s)) == s round-trips for a battery of strings
//                  (ASCII, whitespace runs, contractions, UTF-8, emoji). This
//                  holds for ANY input because every byte symbol is in the vocab,
//                  so encode never drops a byte and decode reverses the map.
//                * a merge fires inside a chunk but never across a pretoken
//                  boundary (the space-splitting rule that is the fiddly part).
//              This is what CTest runs (encode_equivalence).
//
//   --tokenizer <tokenizer.bin> --fixtures <encode_fixtures.txt>
//              EQUIVALENCE: load the real GPT-2 tokenizer.bin and a golden file
//              of (string, HuggingFace ids) produced by
//              tests/make_encode_fixtures.py, and assert encode(string) equals
//              HF's ids exactly. The real HF-parity gate; run it manually (it
//              needs the exported tokenizer + a generated fixtures file), the same
//              way test_loader --fixture proves the byte layout. This is the
//              encode analogue of tests/check_argmax.py.
//
// The end-to-end correctness story stays token identity vs HuggingFace; this
// isolates the encoder.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "tokenizer.hpp"

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
        ++g_failures;
    }
}

// GPT-2 byte->unicode codepoint map, replicated so the hermetic test can write
// the vocab strings the tokenizer rebuilds the same map for on load.
uint32_t byte_to_cp(int b) {
    static const std::array<uint32_t, 256> map = [] {
        std::vector<int> bs;
        for (int x = '!'; x <= '~'; ++x) bs.push_back(x);
        for (int x = 0xA1; x <= 0xAC; ++x) bs.push_back(x);
        for (int x = 0xAE; x <= 0xFF; ++x) bs.push_back(x);
        std::vector<int> cs = bs;
        int n = 0;
        for (int x = 0; x < 256; ++x) {
            if (std::find(bs.begin(), bs.end(), x) == bs.end()) {
                bs.push_back(x);
                cs.push_back(256 + n);
                ++n;
            }
        }
        std::array<uint32_t, 256> m{};
        for (std::size_t i = 0; i < bs.size(); ++i)
            m[static_cast<std::size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
        return m;
    }();
    return map[static_cast<std::size_t>(b)];
}

// All GPT-2 mapped codepoints are < 0x800, but the 3-byte case is here for safety.
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

std::string byte_sym(int b) { return utf8_encode(byte_to_cp(b)); }

void put_i32(std::ofstream& f, std::int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}
void put_str(std::ofstream& f, const std::string& s) {
    put_i32(f, static_cast<std::int32_t>(s.size()));
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// A tiny TOK1 tokenizer: ids 0..255 are the byte symbols (id b == byte_sym(b)),
// then two whole-word vocab entries the two merge rules build -- "ab" at id 256,
// "hi" at id 257. With every byte symbol present, encode never falls off the vocab
// so decode(encode(.)) round-trips for any input.
void write_tiny_tokenizer(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("TOK1", 4);
    put_i32(f, 1);    // version
    put_i32(f, 258);  // n_vocab: 256 byte symbols + "ab" + "hi"
    put_i32(f, 2);    // n_merges
    for (int b = 0; b < 256; ++b) put_str(f, byte_sym(b));
    put_str(f, "ab");
    put_str(f, "hi");
    // merge rules, rank 0 then rank 1
    put_str(f, "a");
    put_str(f, "b");
    put_str(f, "h");
    put_str(f, "i");
}

int run_hermetic() {
    const std::string tok_path = "ie_test_encode_tokenizer.bin";
    write_tiny_tokenizer(tok_path);

    ie::Tokenizer tok;
    if (!tok.load(tok_path)) {
        std::fprintf(stderr, "  FAIL: tokenizer.load returned false\n");
        return 1;
    }
    std::printf("[test_encode] hermetic (tiny in-memory TOK1 tokenizer)\n");

    auto rt = [&](const std::string& s) { return tok.decode(tok.encode(s)) == s; };

    // decode(encode(s)) == s for everything we can throw at it (UTF-8 bytes pass
    // through the byte-level path; chunking may differ from HF for emoji, but
    // reversibility does not depend on chunking).
    const std::vector<std::string> battery = {
        "",
        "a",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and   multiple   spaces  ",
        "trailing spaces   ",
        "don't can't I'm we'll they've he'd you're it's",
        "tabs\tand\nnewlines\r\nhere",
        "MixedCase 123 456 7th",
        "punctuation: (parens), [brackets]; \"quotes\" -- dashes!",
        "ab hi abhi high",                       // exercises both merges
        "caf\xC3\xA9 na\xC3\xAFve fa\xC3\xA7""ade",  // accented Latin (UTF-8)
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",  // CJK
        "emoji \xF0\x9F\x98\x80\xF0\x9F\x8E\x89 test",  // emoji (4-byte UTF-8)
    };
    for (const std::string& s : battery) check(rt(s), "decode(encode(s)) == s");

    // Merges fire inside a single pretoken chunk.
    check(tok.encode("ab") == std::vector<int>{256}, "merge 'a'+'b' -> id 256");
    check(tok.encode("hi") == std::vector<int>{257}, "merge 'h'+'i' -> id 257");
    check(static_cast<int>(tok.encode("ab").size()) == 1, "'ab' is one token");

    // ...but never across a pretoken boundary: the space splits "a" | " b", so the
    // 'a'+'b' merge cannot reach across it. "a b" stays 'a', ' ', 'b' (3 tokens).
    check(static_cast<int>(tok.encode("a b").size()) == 3, "space blocks the 'a'+'b' merge");
    check(tok.encode("a b").size() > tok.encode("ab").size(),
          "boundary keeps the tokens separate");

    if (g_failures == 0) {
        std::printf("test_encode: OK (%zu round-trips + merge/boundary checks)\n",
                    battery.size());
        return 0;
    }
    std::fprintf(stderr, "test_encode: %d check(s) failed\n", g_failures);
    return 1;
}

// Reverse the make_encode_fixtures.py escaping: \\ \n \t \r ; other bytes literal.
std::string unescape(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            switch (c) {
                case 'n': o.push_back('\n'); break;
                case 't': o.push_back('\t'); break;
                case 'r': o.push_back('\r'); break;
                case '\\': o.push_back('\\'); break;
                default:
                    o.push_back('\\');
                    o.push_back(c);
                    break;
            }
        } else {
            o.push_back(s[i]);
        }
    }
    return o;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    const char* p = s.c_str();
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        ids.push_back(static_cast<int>(v));
        p = end;
    }
    return ids;
}

int run_fixture(const std::string& tok_path, const std::string& fixtures_path) {
    ie::Tokenizer tok;
    if (!tok.load(tok_path)) {
        std::fprintf(stderr, "  FAIL: could not load tokenizer '%s'\n", tok_path.c_str());
        return 1;
    }
    std::ifstream in(fixtures_path);
    if (!in) {
        std::fprintf(stderr, "  FAIL: could not open fixtures '%s'\n", fixtures_path.c_str());
        return 1;
    }
    std::printf("[test_encode] equivalence: '%s' vs HuggingFace ids in '%s'\n", tok_path.c_str(),
                fixtures_path.c_str());

    int n = 0, shown = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;  // skip blank / comment lines
        const std::vector<int> expected = parse_ids(line.substr(0, tab));
        const std::string text = unescape(line.substr(tab + 1));
        const std::vector<int> got = tok.encode(text);
        ++n;
        if (got != expected) {
            ++g_failures;
            if (shown < 8) {
                ++shown;
                std::fprintf(stderr, "  MISMATCH on \"%s\"\n    expected:", text.c_str());
                for (int v : expected) std::fprintf(stderr, " %d", v);
                std::fprintf(stderr, "\n    got:     ");
                for (int v : got) std::fprintf(stderr, " %d", v);
                std::fprintf(stderr, "\n");
            }
        }
    }
    if (n == 0) {
        std::fprintf(stderr, "  FAIL: no fixtures read from '%s'\n", fixtures_path.c_str());
        return 1;
    }
    if (g_failures == 0) {
        std::printf("test_encode: OK (%d strings match HuggingFace)\n", n);
        return 0;
    }
    std::fprintf(stderr, "test_encode: %d/%d strings mismatched\n", g_failures, n);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string tok_path, fixtures_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc)
            tok_path = argv[++i];
        else if (std::strcmp(argv[i], "--fixtures") == 0 && i + 1 < argc)
            fixtures_path = argv[++i];
    }
    if (tok_path.empty() && fixtures_path.empty()) return run_hermetic();
    if (tok_path.empty() || fixtures_path.empty()) {
        std::fprintf(stderr,
                     "usage: test_encode [--tokenizer <tokenizer.bin> --fixtures <file>]\n"
                     "  no args: hermetic self-test; both flags: HuggingFace-parity check\n");
        return 2;
    }
    return run_fixture(tok_path, fixtures_path);
}