#pragma once

#include "titaninfer/layers/layer.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief Flatten layer: reshapes multi-dimensional tensor to 1D/2D
 *
 * (C, H, W)       -> (C*H*W,)
 * (N, C, H, W)    -> (N, C*H*W)
 * Data is contiguous in row-major, so this is a memcpy + shape change.
 */
class FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;
};

} // namespace layers
} // namespace titaninfer
