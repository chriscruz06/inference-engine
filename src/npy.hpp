// src/npy.hpp
#pragma once
// Minimal NumPy .npy (v1.0) I/O for the verification harness.
//   - writer: 2-D float32 arrays  (the engine's activation dumps)
//   - reader: 1-D int32 arrays    (reference input_ids.npy during bringup)
// Scope is deliberately tiny: it only has to round-trip the shapes this project
// produces, not the whole .npy spec. Assumes a little-endian host (x86/ARM).

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npy {

static const unsigned char MAGIC[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

// Write `data` (rows*cols floats, row-major) as a 2-D float32 .npy.
// Returns false on shape mismatch or I/O error (message on stderr).
inline bool save_2d(const std::string& path, const std::vector<float>& data, int rows, int cols) {
    if (static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) != data.size()) {
        std::fprintf(stderr, "npy::save_2d: %s shape %dx%d != %zu elems\n", path.c_str(), rows,
                     cols, data.size());
        return false;
    }
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
                      ", " + std::to_string(cols) + "), }";
    // magic(6) + version(2) + hlen(2) + header must be a multiple of 64; header ends in '\n'.
    const std::size_t total = 10 + hdr.size() + 1;
    const std::size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(("npy::save_2d: " + path).c_str());
        return false;
    }

    const unsigned char version[2] = {0x01, 0x00};
    const std::uint16_t hlen = static_cast<std::uint16_t>(hdr.size());
    const unsigned char hlen_le[2] = {static_cast<unsigned char>(hlen & 0xFF),
                                      static_cast<unsigned char>((hlen >> 8) & 0xFF)};

    const bool ok = std::fwrite(MAGIC, 1, 6, f) == 6 && std::fwrite(version, 1, 2, f) == 2 &&
                    std::fwrite(hlen_le, 1, 2, f) == 2 &&
                    std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size() &&
                    std::fwrite(data.data(), sizeof(float), data.size(), f) == data.size();
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "npy::save_2d: short write to %s\n", path.c_str());
    return ok;
}

// Read a 1-D int32 .npy (e.g. reference input_ids.npy) into a vector<int>.
// Returns an empty vector on anything it doesn't recognize (message on stderr).
inline std::vector<int> load_i32_1d(const std::string& path) {
    std::vector<int> out;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror(("npy::load_i32_1d: " + path).c_str());
        return out;
    }

    unsigned char head[8];
    if (std::fread(head, 1, 8, f) != 8 || std::memcmp(head, MAGIC, 6) != 0) {
        std::fprintf(stderr, "npy::load_i32_1d: %s is not a .npy file\n", path.c_str());
        std::fclose(f);
        return out;
    }

    std::size_t hlen = 0;
    if (head[6] >= 2) {  // version 2.0+: 4-byte header length
        unsigned char b[4];
        if (std::fread(b, 1, 4, f) != 4) {
            std::fclose(f);
            return out;
        }
        hlen = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8) |
               (static_cast<std::size_t>(b[2]) << 16) | (static_cast<std::size_t>(b[3]) << 24);
    } else {  // version 1.0: 2-byte header length
        unsigned char b[2];
        if (std::fread(b, 1, 2, f) != 2) {
            std::fclose(f);
            return out;
        }
        hlen = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8);
    }

    std::string hdr(hlen, '\0');
    if (std::fread(&hdr[0], 1, hlen, f) != hlen) {
        std::fclose(f);
        return out;
    }

    if (hdr.find("i4") == std::string::npos) {
        std::fprintf(stderr, "npy::load_i32_1d: %s is not int32 (header: %s)\n", path.c_str(),
                     hdr.c_str());
        std::fclose(f);
        return out;
    }

    // Pull the element count out of the shape tuple, e.g. "(4,)" -> 4 (product of dims).
    const std::size_t sp = hdr.find("'shape':");
    const std::size_t lp = hdr.find('(', sp);
    const std::size_t rp = hdr.find(')', lp);
    std::size_t count = 1;
    bool any = false;
    for (std::size_t i = lp + 1; i < rp;) {
        if (std::isdigit(static_cast<unsigned char>(hdr[i]))) {
            std::size_t v = 0;
            while (i < rp && std::isdigit(static_cast<unsigned char>(hdr[i])))
                v = v * 10 + static_cast<std::size_t>(hdr[i++] - '0');
            count *= v;
            any = true;
        } else {
            ++i;
        }
    }
    if (!any) count = 0;

    static_assert(sizeof(int) == 4, "load_i32_1d assumes 32-bit int");
    out.resize(count);
    if (std::fread(out.data(), sizeof(int), count, f) != count) {
        std::fprintf(stderr, "npy::load_i32_1d: short read of %zu ints from %s\n", count,
                     path.c_str());
        out.clear();
    }
    std::fclose(f);
    return out;
}

}  // namespace npy