#include "titaninfer/ops/activations.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace titaninfer {
namespace ops {

// ========================================
// ReLU
// ========================================

void relu(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    const size_t n = input.size();
    const float* in = input.data();
    float* out = output.data();

    for (size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

void relu_inplace(Tensor& input) {
    const size_t n = input.size();
    float* data = input.data();

    for (size_t i = 0; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

// ========================================
// Sigmoid
// ========================================

void sigmoid(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    const size_t n = input.size();
    const float* in = input.data();
    float* out = output.data();

    for (size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void sigmoid_inplace(Tensor& input) {
    const size_t n = input.size();
    float* data = input.data();

    for (size_t i = 0; i < n; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// ========================================
// Tanh
// ========================================

void tanh_activation(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    const size_t n = input.size();
    const float* in = input.data();
    float* out = output.data();

    for (size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

void tanh_inplace(Tensor& input) {
    const size_t n = input.size();
    float* data = input.data();

    for (size_t i = 0; i < n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

// ========================================
// Softmax (numerically stable)
// ========================================

static void softmax_1d(const float* in, float* out, size_t n) {
    // Step 1: find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        max_val = std::max(max_val, in[i]);
    }

    // Step 2: compute exp(x_i - max) and accumulate sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }

    // Step 3: normalize
    for (size_t i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

void softmax(const Tensor& input, Tensor& output) {
    if (input.ndim() > 2) {
        throw std::invalid_argument(
            "softmax supports 1D and 2D tensors only, got " +
            std::to_string(input.ndim()) + "D"
        );
    }

    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    const float* in = input.data();
    float* out = output.data();

    if (input.ndim() == 1) {
        softmax_1d(in, out, input.size());
    } else {
        // 2D: row-wise softmax
        const size_t rows = input.shape()[0];
        const size_t cols = input.shape()[1];

        for (size_t r = 0; r < rows; ++r) {
            softmax_1d(in + r * cols, out + r * cols, cols);
        }
    }
}

} // namespace ops
} // namespace titaninfer
