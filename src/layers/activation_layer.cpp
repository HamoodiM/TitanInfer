#include "titaninfer/layers/activation_layer.hpp"

namespace titaninfer {
namespace layers {

// ========================================
// ReluLayer
// ========================================

void ReluLayer::forward(const Tensor& input, Tensor& output) {
    ops::relu(input, output);
}

std::string ReluLayer::name() const {
    return "ReLU";
}

// ========================================
// SigmoidLayer
// ========================================

void SigmoidLayer::forward(const Tensor& input, Tensor& output) {
    ops::sigmoid(input, output);
}

std::string SigmoidLayer::name() const {
    return "Sigmoid";
}

// ========================================
// TanhLayer
// ========================================

void TanhLayer::forward(const Tensor& input, Tensor& output) {
    ops::tanh_activation(input, output);
}

std::string TanhLayer::name() const {
    return "Tanh";
}

// ========================================
// SoftmaxLayer
// ========================================

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {
    ops::softmax(input, output);
}

std::string SoftmaxLayer::name() const {
    return "Softmax";
}

} // namespace layers
} // namespace titaninfer
