#include "model.hpp"

#include <cstdio>

namespace ie {

bool load_model(const std::string& path, Model& out) {
    // TODO:
    //   1. open `path`, read the header (magic + config), validate against
    //      `out.config`;
    //   2. read raw fp32 tensors in the fixed order written by
    //      tools/export_weights.py into `out.weights`;
    //   3. record per-tensor offsets for the forward pass.
    (void)path;
    (void)out;
    std::fprintf(stderr, "[model] load_model: not yet implemented\n");
    return false;
}

}  // namespace ie
