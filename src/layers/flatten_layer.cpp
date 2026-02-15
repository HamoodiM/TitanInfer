#include "titaninfer/layers/flatten_layer.hpp"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace titaninfer {
namespace layers {

std::unique_ptr<Layer> FlattenLayer::clone() const {
    return std::make_unique<FlattenLayer>();
}

void FlattenLayer::forward(const Tensor& input, Tensor& output) {
    auto out_shape = output_shape(input.shape());
    if (output.shape() != out_shape) {
        output = Tensor(out_shape);
    }
    std::memcpy(output.data(), input.data(), input.size() * sizeof(float));
}

std::string FlattenLayer::name() const {
    return "Flatten";
}

std::vector<size_t> FlattenLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() <= 1) {
        return input_shape; // Already flat
    }
    if (input_shape.size() == 2) {
        return input_shape; // (N, features) is already flat
    }
    if (input_shape.size() == 3) {
        // (C, H, W) -> (C*H*W,)
        size_t total = input_shape[0] * input_shape[1] * input_shape[2];
        return {total};
    }
    if (input_shape.size() >= 4) {
        // (N, C, H, W, ...) -> (N, C*H*W*...)
        size_t flat = 1;
        for (size_t i = 1; i < input_shape.size(); ++i) {
            flat *= input_shape[i];
        }
        return {input_shape[0], flat};
    }
    return input_shape;
}

} // namespace layers
} // namespace titaninfer
