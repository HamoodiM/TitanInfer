#include "titaninfer/layers/quantized_dense_layer.hpp"
#include "titaninfer/ops/quantized_ops.hpp"
#include "titaninfer/ops/matrix_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace titaninfer {
namespace layers {

QuantizedDenseLayer::QuantizedDenseLayer(const DenseLayer& dense)
    : in_features_(dense.in_features())
    , out_features_(dense.out_features())
    , use_bias_(dense.has_bias())
    , weights_q_({1})  // placeholder, will be replaced
    , bias_(dense.bias())
{
    // DenseLayer stores weights as (out, in).
    // For 2D batched: Y = X @ W^T = X(batch, in) @ W^T(in, out) = Y(batch, out)
    // Transpose weights to (in, out) and quantize
    Tensor wt({in_features_, out_features_});
    ops::transpose(dense.weights(), wt);
    weights_q_ = QuantizedTensor::quantize(wt);
}

std::unique_ptr<Layer> QuantizedDenseLayer::clone() const {
    // Reconstruct from a temporary DenseLayer... or just copy directly
    auto copy = std::make_unique<QuantizedDenseLayer>(*this);
    return copy;
}

void QuantizedDenseLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 1) {
        // Treat as (1, in_features)
        if (input.shape()[0] != in_features_) {
            throw std::invalid_argument(
                "QuantizedDenseLayer: expected input size " +
                std::to_string(in_features_));
        }

        Tensor input_2d({1, in_features_});
        std::memcpy(input_2d.data(), input.data(), in_features_ * sizeof(float));

        QuantizedTensor input_q = QuantizedTensor::quantize(input_2d);
        Tensor result({1});
        ops::gemm_int8(input_q, weights_q_, result);

        // Reshape to 1D and add bias
        std::vector<size_t> out_shape = {out_features_};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }
        std::memcpy(output.data(), result.data(), out_features_ * sizeof(float));

        if (use_bias_) {
            for (size_t i = 0; i < out_features_; ++i) {
                output.data()[i] += bias_.data()[i];
            }
        }

    } else if (input.ndim() == 2) {
        if (input.shape()[1] != in_features_) {
            throw std::invalid_argument(
                "QuantizedDenseLayer: expected input features " +
                std::to_string(in_features_));
        }

        QuantizedTensor input_q = QuantizedTensor::quantize(input);
        // input_q (batch, in) @ weights_q_ (in, out) = result (batch, out)
        ops::gemm_int8(input_q, weights_q_, output);

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
            "QuantizedDenseLayer: expected 1D or 2D input");
    }
}

std::string QuantizedDenseLayer::name() const {
    return "QuantizedDense(" + std::to_string(in_features_) + ", " +
           std::to_string(out_features_) + ")";
}

size_t QuantizedDenseLayer::parameter_count() const {
    size_t count = out_features_ * in_features_;
    if (use_bias_) {
        count += out_features_;
    }
    return count;
}

std::vector<size_t> QuantizedDenseLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 1) {
        return {out_features_};
    } else if (input_shape.size() == 2) {
        return {input_shape[0], out_features_};
    }
    throw std::invalid_argument(
        "QuantizedDenseLayer::output_shape: expected 1D or 2D");
}

} // namespace layers
} // namespace titaninfer
