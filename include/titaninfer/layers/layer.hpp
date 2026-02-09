#pragma once

#include "titaninfer/tensor.hpp"
#include <string>

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
};

} // namespace layers
} // namespace titaninfer
