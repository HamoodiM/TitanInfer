#include "titaninfer/layers/activation_layer.hpp"

namespace titaninfer {
namespace layers {

// ========================================
// ReluLayer
// ========================================

std::unique_ptr<Layer> ReluLayer::clone() const {
    return std::make_unique<ReluLayer>();
}

void ReluLayer::forward(const Tensor& input, Tensor& output) {
    ops::relu(input, output);
}

std::string ReluLayer::name() const {
    return "ReLU";
}

// ========================================
// SigmoidLayer
// ========================================

std::unique_ptr<Layer> SigmoidLayer::clone() const {
    return std::make_unique<SigmoidLayer>();
}

void SigmoidLayer::forward(const Tensor& input, Tensor& output) {
    ops::sigmoid(input, output);
}

std::string SigmoidLayer::name() const {
    return "Sigmoid";
}

// ========================================
// TanhLayer
// ========================================

std::unique_ptr<Layer> TanhLayer::clone() const {
    return std::make_unique<TanhLayer>();
}

void TanhLayer::forward(const Tensor& input, Tensor& output) {
    ops::tanh_activation(input, output);
}

std::string TanhLayer::name() const {
    return "Tanh";
}

// ========================================
// SoftmaxLayer
// ========================================

std::unique_ptr<Layer> SoftmaxLayer::clone() const {
    return std::make_unique<SoftmaxLayer>();
}

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {
    ops::softmax(input, output);
}

std::string SoftmaxLayer::name() const {
    return "Softmax";
}

} // namespace layers
} // namespace titaninfer
