#include "titaninfer/layers/fused_layers.hpp"
#include "titaninfer/ops/matrix_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace titaninfer {
namespace layers {

// ========================================
// FusedDenseReluLayer
// ========================================

FusedDenseReluLayer::FusedDenseReluLayer(const DenseLayer& dense)
    : in_features_(dense.in_features())
    , out_features_(dense.out_features())
    , use_bias_(dense.has_bias())
    , weights_(dense.weights())
    , bias_(dense.bias())
{
}

std::unique_ptr<Layer> FusedDenseReluLayer::clone() const {
    // Reconstruct from a temporary DenseLayer
    DenseLayer temp(in_features_, out_features_, use_bias_);
    temp.set_weights(weights_);
    if (use_bias_) {
        temp.set_bias(bias_);
    }
    return std::make_unique<FusedDenseReluLayer>(temp);
}

void FusedDenseReluLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 1) {
        if (input.shape()[0] != in_features_) {
            throw std::invalid_argument("FusedDenseReluLayer: input size mismatch");
        }

        ops::matvec(weights_, input, output);

        // Fused bias + ReLU in single pass
        for (size_t i = 0; i < out_features_; ++i) {
            float val = output.data()[i];
            if (use_bias_) {
                val += bias_.data()[i];
            }
            output.data()[i] = std::max(0.0f, val);
        }

    } else if (input.ndim() == 2) {
        if (input.shape()[1] != in_features_) {
            throw std::invalid_argument("FusedDenseReluLayer: input features mismatch");
        }

        Tensor wt({in_features_, out_features_});
        ops::transpose(weights_, wt);
        ops::matmul(input, wt, output);

        // Fused bias + ReLU
        const size_t batch = input.shape()[0];
        for (size_t r = 0; r < batch; ++r) {
            for (size_t c = 0; c < out_features_; ++c) {
                float val = output.data()[r * out_features_ + c];
                if (use_bias_) {
                    val += bias_.data()[c];
                }
                output.data()[r * out_features_ + c] = std::max(0.0f, val);
            }
        }

    } else {
        throw std::invalid_argument("FusedDenseReluLayer: expected 1D or 2D input");
    }
}

std::string FusedDenseReluLayer::name() const {
    return "FusedDenseReLU(" + std::to_string(in_features_) + ", " +
           std::to_string(out_features_) + ")";
}

size_t FusedDenseReluLayer::parameter_count() const {
    size_t count = out_features_ * in_features_;
    if (use_bias_) count += out_features_;
    return count;
}

std::vector<size_t> FusedDenseReluLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 1) return {out_features_};
    if (input_shape.size() == 2) return {input_shape[0], out_features_};
    throw std::invalid_argument("FusedDenseReluLayer::output_shape: expected 1D or 2D");
}

// ========================================
// FusedDenseSigmoidLayer
// ========================================

FusedDenseSigmoidLayer::FusedDenseSigmoidLayer(const DenseLayer& dense)
    : in_features_(dense.in_features())
    , out_features_(dense.out_features())
    , use_bias_(dense.has_bias())
    , weights_(dense.weights())
    , bias_(dense.bias())
{
}

std::unique_ptr<Layer> FusedDenseSigmoidLayer::clone() const {
    DenseLayer temp(in_features_, out_features_, use_bias_);
    temp.set_weights(weights_);
    if (use_bias_) {
        temp.set_bias(bias_);
    }
    return std::make_unique<FusedDenseSigmoidLayer>(temp);
}

void FusedDenseSigmoidLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 1) {
        if (input.shape()[0] != in_features_) {
            throw std::invalid_argument("FusedDenseSigmoidLayer: input size mismatch");
        }

        ops::matvec(weights_, input, output);

        for (size_t i = 0; i < out_features_; ++i) {
            float val = output.data()[i];
            if (use_bias_) {
                val += bias_.data()[i];
            }
            output.data()[i] = 1.0f / (1.0f + std::exp(-val));
        }

    } else if (input.ndim() == 2) {
        if (input.shape()[1] != in_features_) {
            throw std::invalid_argument("FusedDenseSigmoidLayer: input features mismatch");
        }

        Tensor wt({in_features_, out_features_});
        ops::transpose(weights_, wt);
        ops::matmul(input, wt, output);

        const size_t batch = input.shape()[0];
        for (size_t r = 0; r < batch; ++r) {
            for (size_t c = 0; c < out_features_; ++c) {
                float val = output.data()[r * out_features_ + c];
                if (use_bias_) {
                    val += bias_.data()[c];
                }
                output.data()[r * out_features_ + c] = 1.0f / (1.0f + std::exp(-val));
            }
        }

    } else {
        throw std::invalid_argument("FusedDenseSigmoidLayer: expected 1D or 2D input");
    }
}

std::string FusedDenseSigmoidLayer::name() const {
    return "FusedDenseSigmoid(" + std::to_string(in_features_) + ", " +
           std::to_string(out_features_) + ")";
}

size_t FusedDenseSigmoidLayer::parameter_count() const {
    size_t count = out_features_ * in_features_;
    if (use_bias_) count += out_features_;
    return count;
}

std::vector<size_t> FusedDenseSigmoidLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 1) return {out_features_};
    if (input_shape.size() == 2) return {input_shape[0], out_features_};
    throw std::invalid_argument("FusedDenseSigmoidLayer::output_shape: expected 1D or 2D");
}

} // namespace layers
} // namespace titaninfer
