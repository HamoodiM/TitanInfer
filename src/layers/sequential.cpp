#include "titaninfer/layers/sequential.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace titaninfer {
namespace layers {

void Sequential::add(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("Sequential::add: cannot add null layer");
    }
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential::forward: no layers added");
    }

    // Ping-pong buffers avoid repeated allocation for shape-preserving layers.
    // Each layer's forward() auto-allocates if shape mismatches.
    Tensor buffer_a({1});
    Tensor buffer_b({1});

    layers_[0]->forward(input, buffer_a);

    for (size_t i = 1; i < layers_.size(); ++i) {
        if (i % 2 == 1) {
            layers_[i]->forward(buffer_a, buffer_b);
        } else {
            layers_[i]->forward(buffer_b, buffer_a);
        }
    }

    // Result is in buffer_a for odd layer count, buffer_b for even.
    // Local variables are implicitly moved on return.
    if (layers_.size() % 2 == 1) {
        return buffer_a;
    } else {
        return buffer_b;
    }
}

std::string Sequential::summary(const std::vector<size_t>& input_shape) const {
    std::ostringstream oss;

    oss << "================================================================\n";
    oss << std::left << std::setw(25) << "Layer"
        << std::setw(25) << "Output Shape"
        << std::right << std::setw(12) << "Parameters" << "\n";
    oss << "================================================================\n";

    std::vector<size_t> current_shape = input_shape;
    size_t total_params = 0;

    for (size_t i = 0; i < layers_.size(); ++i) {
        current_shape = layers_[i]->output_shape(current_shape);
        size_t params = layers_[i]->parameter_count();
        total_params += params;

        std::string shape_str = "(";
        for (size_t d = 0; d < current_shape.size(); ++d) {
            if (d > 0) shape_str += ", ";
            shape_str += std::to_string(current_shape[d]);
        }
        shape_str += ")";

        oss << std::left << std::setw(25) << layers_[i]->name()
            << std::setw(25) << shape_str
            << std::right << std::setw(12) << params << "\n";
    }

    oss << "================================================================\n";
    oss << "Total parameters: " << total_params << "\n";
    oss << "================================================================\n";

    return oss.str();
}

Layer& Sequential::layer(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Sequential::layer: index " + std::to_string(index) +
            " out of range (size " + std::to_string(layers_.size()) + ")");
    }
    return *layers_[index];
}

const Layer& Sequential::layer(size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Sequential::layer: index " + std::to_string(index) +
            " out of range (size " + std::to_string(layers_.size()) + ")");
    }
    return *layers_[index];
}

size_t Sequential::total_parameters() const {
    size_t total = 0;
    for (const auto& l : layers_) {
        total += l->parameter_count();
    }
    return total;
}

} // namespace layers
} // namespace titaninfer
