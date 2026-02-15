#pragma once

#include "titaninfer/layers/layer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/quantized_tensor.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief INT8 quantized dense layer
 *
 * Constructed from an existing DenseLayer, quantizing weights at construction time.
 * Forward: quantize input -> INT8 GEMM -> add FP32 bias
 */
class QuantizedDenseLayer : public Layer {
public:
    /// Construct by quantizing an existing DenseLayer's weights
    explicit QuantizedDenseLayer(const DenseLayer& dense);

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
    QuantizedTensor weights_q_;   // quantized (in_features, out_features) -- transposed
    Tensor bias_;                 // FP32
};

} // namespace layers
} // namespace titaninfer
