#include "titaninfer/layers/dense_layer.hpp"
#include <stdexcept>

namespace titaninfer {
namespace layers {

DenseLayer::DenseLayer(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(use_bias)
    , weights_({out_features, in_features})
    , bias_({out_features})
{
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument(
            "DenseLayer: in_features and out_features must be > 0");
    }
}

void DenseLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 1) {
        // 1D: y = W @ x + b
        if (input.shape()[0] != in_features_) {
            throw std::invalid_argument(
                "DenseLayer: expected input size " +
                std::to_string(in_features_) + ", got " +
                std::to_string(input.shape()[0]));
        }

        ops::matvec(weights_, input, output);

        if (use_bias_) {
            for (size_t i = 0; i < out_features_; ++i) {
                output.data()[i] += bias_.data()[i];
            }
        }

    } else if (input.ndim() == 2) {
        // 2D batched: Y = X @ W^T + b
        if (input.shape()[1] != in_features_) {
            throw std::invalid_argument(
                "DenseLayer: expected input features " +
                std::to_string(in_features_) + ", got " +
                std::to_string(input.shape()[1]));
        }

        Tensor wt({in_features_, out_features_});
        ops::transpose(weights_, wt);
        ops::matmul(input, wt, output);

        if (use_bias_) {
            const size_t batch = input.shape()[0];
            for (size_t r = 0; r < batch; ++r) {
                for (size_t c = 0; c < out_features_; ++c) {
                    output.data()[r * out_features_ + c] += bias_.data()[c];
                }
            }
        }

    } else {
        throw std::invalid_argument(
            "DenseLayer: expected 1D or 2D input, got " +
            std::to_string(input.ndim()) + "D");
    }
}

std::string DenseLayer::name() const {
    return "Dense(" + std::to_string(in_features_) + ", " +
           std::to_string(out_features_) + ")";
}

size_t DenseLayer::parameter_count() const {
    size_t count = out_features_ * in_features_;
    if (use_bias_) {
        count += out_features_;
    }
    return count;
}

std::vector<size_t> DenseLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.empty()) {
        throw std::invalid_argument(
            "DenseLayer::output_shape: empty input shape");
    }
    if (input_shape.size() == 1) {
        return {out_features_};
    } else if (input_shape.size() == 2) {
        return {input_shape[0], out_features_};
    } else {
        throw std::invalid_argument(
            "DenseLayer::output_shape: expected 1D or 2D, got " +
            std::to_string(input_shape.size()) + "D");
    }
}

void DenseLayer::set_weights(const Tensor& weights) {
    std::vector<size_t> expected = {out_features_, in_features_};
    if (weights.shape() != expected) {
        throw std::invalid_argument(
            "DenseLayer::set_weights: expected shape (" +
            std::to_string(out_features_) + ", " +
            std::to_string(in_features_) + "), got (" +
            std::to_string(weights.shape()[0]) + ", " +
            std::to_string(weights.shape()[1]) + ")");
    }
    weights_ = weights;
}

void DenseLayer::set_bias(const Tensor& bias) {
    if (!use_bias_) {
        throw std::invalid_argument(
            "DenseLayer::set_bias: bias is disabled for this layer");
    }
    std::vector<size_t> expected = {out_features_};
    if (bias.shape() != expected) {
        throw std::invalid_argument(
            "DenseLayer::set_bias: expected shape (" +
            std::to_string(out_features_) + ",), got (" +
            std::to_string(bias.shape()[0]) + ",)");
    }
    bias_ = bias;
}

} // namespace layers
} // namespace titaninfer
