#pragma once

#include "titaninfer/layers/layer.hpp"
#include "titaninfer/ops/matrix_ops.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief Fully-connected (dense/linear) layer: y = Wx + b
 *
 * Stores weight matrix W of shape (out_features, in_features) and
 * optional bias vector b of shape (out_features).
 *
 * Supports:
 * - 1D input (in_features,) -> 1D output (out_features,) via matvec
 * - 2D input (batch, in_features) -> 2D output (batch, out_features) via matmul
 */
class DenseLayer : public Layer {
public:
    /**
     * @brief Construct a dense layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param use_bias Whether to include a bias term (default: true)
     * @throws std::invalid_argument if in_features or out_features is 0
     */
    DenseLayer(size_t in_features, size_t out_features, bool use_bias = true);

    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    size_t parameter_count() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    /**
     * @brief Set weight matrix
     * @param weights Tensor of shape (out_features, in_features)
     * @throws std::invalid_argument if shape does not match
     */
    void set_weights(const Tensor& weights);

    /**
     * @brief Set bias vector
     * @param bias Tensor of shape (out_features,)
     * @throws std::invalid_argument if shape does not match or bias is disabled
     */
    void set_bias(const Tensor& bias);

    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

    Tensor weights_;  // Shape: (out_features_, in_features_)
    Tensor bias_;     // Shape: (out_features_,)
};

} // namespace layers
} // namespace titaninfer
