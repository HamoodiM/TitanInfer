#pragma once

#include "titaninfer/layers/layer.hpp"
#include <memory>
#include <vector>
#include <string>

namespace titaninfer {
namespace layers {

/**
 * @brief Sequential model container for stacking layers
 *
 * Feeds the output of each layer as input to the next.
 * Owns all layers via unique_ptr for automatic lifecycle management.
 */
class Sequential {
public:
    Sequential() = default;

    /**
     * @brief Append a layer to the model
     * @param layer Layer to add (ownership transferred via move)
     * @throws std::invalid_argument if layer is null
     */
    void add(std::unique_ptr<Layer> layer);

    /**
     * @brief Run forward pass through all layers sequentially
     * @param input Input tensor to the first layer
     * @return Output tensor from the last layer
     * @throws std::runtime_error if no layers have been added
     */
    Tensor forward(const Tensor& input);

    /**
     * @brief Model architecture summary
     * @param input_shape Shape of the expected input tensor
     * @return Formatted string with layer names, output shapes, parameter counts
     */
    std::string summary(const std::vector<size_t>& input_shape) const;

    /** @brief Number of layers */
    size_t size() const { return layers_.size(); }

    /** @brief True if no layers added */
    bool empty() const { return layers_.empty(); }

    /** @brief Access a layer by index (non-owning reference) */
    Layer& layer(size_t index);
    const Layer& layer(size_t index) const;

    /** @brief Total parameter count across all layers */
    size_t total_parameters() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace layers
} // namespace titaninfer
