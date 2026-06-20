#include "model.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace ie {
namespace {

// The reader assumes a little-endian host (x86-64 / ARM64), matching the
// little-endian file format. The whole project targets those platforms.

bool read_exact(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

bool read_i32(std::ifstream& in, int32_t& v) {
    return read_exact(in, &v, sizeof(v));
}
bool read_f32(std::ifstream& in, float& v) {
    return read_exact(in, &v, sizeof(v));
}

bool fail(const char* msg) {
    std::fprintf(stderr, "[model] %s\n", msg);
    return false;
}

}  // namespace

bool load_model(const std::string& path, Model& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("could not open model file");

    char magic[4];
    if (!read_exact(in, magic, 4)) return fail("truncated header (magic)");
    if (std::memcmp(magic, "GPT2", 4) != 0) return fail("bad magic (expected \"GPT2\")");

    int32_t version = 0, n_layers = 0, n_heads = 0, d_model = 0, head_dim = 0;
    int32_t n_ctx = 0, vocab_size = 0, d_mlp = 0;
    float ln_eps = 0.0f;
    if (!read_i32(in, version) || !read_i32(in, n_layers) || !read_i32(in, n_heads) ||
        !read_i32(in, d_model) || !read_i32(in, head_dim) || !read_i32(in, n_ctx) ||
        !read_i32(in, vocab_size) || !read_i32(in, d_mlp) || !read_f32(in, ln_eps)) {
        return fail("truncated header (fields)");
    }
    if (version != 1) return fail("unsupported format version");

    // Validate self-consistency before trusting the sizes.
    if (n_layers <= 0 || n_heads <= 0 || d_model <= 0 || n_ctx <= 0 || vocab_size <= 0 ||
        d_mlp <= 0 || head_dim <= 0) {
        return fail("non-positive config field");
    }
    if (head_dim * n_heads != d_model) return fail("head_dim * n_heads != d_model");

    out.config.n_layers = n_layers;
    out.config.n_heads = n_heads;
    out.config.d_model = d_model;
    out.config.head_dim = head_dim;
    out.config.n_ctx = n_ctx;
    out.config.vocab_size = vocab_size;
    out.config.d_mlp = d_mlp;
    out.config.ln_eps = ln_eps;

    const std::size_t V = static_cast<std::size_t>(vocab_size);
    const std::size_t D = static_cast<std::size_t>(d_model);
    const std::size_t M = static_cast<std::size_t>(d_mlp);
    const std::size_t C = static_cast<std::size_t>(n_ctx);

    // Assign offsets in the SAME order as tools/format_spec.tensor_specs.
    std::size_t cursor = 0;
    auto take = [&cursor](std::size_t count) {
        std::size_t off = cursor;
        cursor += count;
        return off;
    };

    out.w = Weights{};
    out.w.layers.resize(static_cast<std::size_t>(n_layers));
    out.w.wte = take(V * D);
    out.w.wpe = take(C * D);
    for (std::size_t l = 0; l < static_cast<std::size_t>(n_layers); ++l) {
        LayerWeights& lw = out.w.layers[l];
        lw.ln_1_w = take(D);
        lw.ln_1_b = take(D);
        lw.c_attn_w = take(3 * D * D);
        lw.c_attn_b = take(3 * D);
        lw.c_proj_w = take(D * D);
        lw.c_proj_b = take(D);
        lw.ln_2_w = take(D);
        lw.ln_2_b = take(D);
        lw.mlp_fc_w = take(M * D);
        lw.mlp_fc_b = take(M);
        lw.mlp_proj_w = take(D * M);
        lw.mlp_proj_b = take(D);
    }
    out.w.ln_f_w = take(D);
    out.w.ln_f_b = take(D);

    const std::size_t total = cursor;
    out.weights.resize(total);
    if (!read_exact(in, out.weights.data(), total * sizeof(float))) {
        return fail("truncated tensor data (file smaller than header implies)");
    }
    // There should be nothing left but EOF.
    char extra = 0;
    in.read(&extra, 1);
    if (in.gcount() != 0) return fail("trailing bytes after tensors (size mismatch)");

    return true;
}

std::size_t quantize_model(Model& m, QuantType type, int group, std::size_t* fp32_bytes_out) {
    const ModelConfig& c = m.config;
    const int D = c.d_model, M = c.d_mlp, V = c.vocab_size;

    m.quant_type = type;
    m.qlayers.assign(static_cast<std::size_t>(c.n_layers), QuantLayerWeights{});

    std::size_t qbytes = 0, fbytes = 0;
    auto do_one = [&](std::size_t off, int out_dim, int in_dim, QuantTensor& dst) {
        quantize_tensor(m.at(off), out_dim, in_dim, type, group, dst);
        qbytes += dst.bytes();
        fbytes += static_cast<std::size_t>(out_dim) * static_cast<std::size_t>(in_dim) *
                  sizeof(float);
    };

    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = m.w.layers[static_cast<std::size_t>(l)];
        QuantLayerWeights& Q = m.qlayers[static_cast<std::size_t>(l)];
        do_one(L.c_attn_w, 3 * D, D, Q.c_attn_w);
        do_one(L.c_proj_w, D, D, Q.c_proj_w);
        do_one(L.mlp_fc_w, M, D, Q.mlp_fc_w);
        do_one(L.mlp_proj_w, D, M, Q.mlp_proj_w);
    }
    do_one(m.w.wte, V, D, m.q_head);  // tied LM head (wte reused as [vocab, d])

    m.quantized = true;
    if (fp32_bytes_out) *fp32_bytes_out = fbytes;
    return qbytes;
}

}  // namespace ie
