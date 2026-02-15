#pragma once

#include "titaninfer/layers/layer.hpp"
#include "titaninfer/layers/dense_layer.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief Fused Dense + ReLU layer
 *
 * Computes matmul + bias + ReLU in a single pass over the output tensor,
 * eliminating the intermediate tensor read/write from separate Dense + ReLU.
 */
class FusedDenseReluLayer : public Layer {
public:
    explicit FusedDenseReluLayer(const DenseLayer& dense);

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    size_t parameter_count() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

private:
    size_t in_features_, out_features_;
    bool use_bias_;
    Tensor weights_, bias_;
};

/**
 * @brief Fused Dense + Sigmoid layer
 */
class FusedDenseSigmoidLayer : public Layer {
public:
    explicit FusedDenseSigmoidLayer(const DenseLayer& dense);

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    size_t parameter_count() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

private:
    size_t in_features_, out_features_;
    bool use_bias_;
    Tensor weights_, bias_;
};

} // namespace layers
} // namespace titaninfer
