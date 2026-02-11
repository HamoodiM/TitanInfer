#pragma once

#include "titaninfer/tensor.hpp"
#include <string>
#include <vector>

namespace titaninfer {
namespace layers {

/**
 * @brief Abstract base class for inference layers
 *
 * Provides a minimal interface for composable inference pipelines.
 * No backward(), parameters(), or train/eval modes — this is an inference engine.
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Run forward pass
     *
     * @param input Input tensor (const ref)
     * @param output Output tensor (mutable ref, auto-allocated if needed)
     */
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    /**
     * @brief Human-readable layer name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Total number of learnable parameters (weights + bias)
     * @return 0 for parameterless layers (e.g. activations)
     */
    virtual size_t parameter_count() const { return 0; }

    /**
     * @brief Compute output shape given an input shape
     * @param input_shape Shape of the input tensor
     * @return Output shape (identity for shape-preserving layers)
     */
    virtual std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const {
        return input_shape;
    }
};

} // namespace layers
} // namespace titaninfer
