#include "forward.hpp"

namespace ie {

void layernorm(const float* x, const float* weight, const float* bias, float* out, int n,
               float eps) {
    // TODO: mean & variance over n, normalize, then out = norm * weight + bias.
    (void)x;
    (void)weight;
    (void)bias;
    (void)out;
    (void)n;
    (void)eps;
}

void gelu_tanh(float* x, int n) {
    // TODO: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).
    (void)x;
    (void)n;
}

void softmax(float* x, int n) {
    // TODO: subtract max for stability, exponentiate, divide by the sum.
    (void)x;
    (void)n;
}

std::vector<float> forward(const Model& model, const std::vector<int>& tokens) {
    // TODO: embed + position -> [LN -> attn -> LN -> mlp] * n_layers
    //       -> final LN -> (tied) LM head -> logits.
    (void)tokens;
    return std::vector<float>(static_cast<size_t>(model.config.vocab_size), 0.0f);
}

}  // namespace ie
