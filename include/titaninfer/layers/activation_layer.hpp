#pragma once

#include "titaninfer/layers/layer.hpp"
#include "titaninfer/ops/activations.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief ReLU activation layer
 */
class ReluLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
};

/**
 * @brief Sigmoid activation layer
 */
class SigmoidLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
};

/**
 * @brief Tanh activation layer
 */
class TanhLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
};

/**
 * @brief Softmax activation layer (1D or 2D only)
 */
class SoftmaxLayer : public Layer {
public:
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
};

} // namespace layers
} // namespace titaninfer
