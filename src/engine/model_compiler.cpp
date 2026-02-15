#include "titaninfer/engine/model_compiler.hpp"
#include "titaninfer/engine/fusion.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/quantized_dense_layer.hpp"

#include <cstring>
#include <stdexcept>

namespace titaninfer {
namespace engine {

Tensor CompiledModel::predict(const Tensor& input) {
    if (!model_ || model_->empty()) {
        throw std::runtime_error("CompiledModel: no model loaded");
    }

    // Validate input shape
    if (input.shape().size() != input_shape_.size()) {
        throw std::invalid_argument("CompiledModel: input dimension mismatch");
    }
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        if (input.shape()[i] != input_shape_[i]) {
            throw std::invalid_argument("CompiledModel: input shape mismatch");
        }
    }

    const size_t n_layers = model_->size();

    // Execute first layer
    model_->layer(0).forward(input, buffers_[0]);

    // Chain remaining layers
    for (size_t i = 1; i < n_layers; ++i) {
        model_->layer(i).forward(buffers_[i - 1], buffers_[i]);
    }

    // Return deep copy of output
    Tensor result(buffers_[n_layers - 1].shape());
    std::memcpy(result.data(), buffers_[n_layers - 1].data(),
                buffers_[n_layers - 1].size() * sizeof(float));
    return result;
}

std::string CompiledModel::summary() const {
    if (!model_) return "(no model)";
    return model_->summary(input_shape_);
}

size_t CompiledModel::layer_count() const {
    if (!model_) return 0;
    return model_->size();
}

CompiledModel ModelCompiler::compile(
    const layers::Sequential& model,
    const std::vector<size_t>& input_shape,
    const CompileOptions& options) {

    if (model.empty()) {
        throw std::invalid_argument("ModelCompiler: empty model");
    }

    CompiledModel compiled;
    compiled.input_shape_ = input_shape;

    // Step 1: Clone the model
    auto cloned = std::make_unique<layers::Sequential>();
    for (size_t i = 0; i < model.size(); ++i) {
        cloned->add(model.layer(i).clone());
    }

    // Step 2: Apply fusion
    if (options.enable_fusion) {
        cloned = apply_fusion(*cloned);
    }

    // Step 3: Apply quantization (replace DenseLayers with QuantizedDenseLayers)
    if (options.enable_quantization) {
        auto quantized = std::make_unique<layers::Sequential>();
        for (size_t i = 0; i < cloned->size(); ++i) {
            auto* dense = dynamic_cast<layers::DenseLayer*>(&cloned->layer(i));
            if (dense) {
                quantized->add(std::make_unique<layers::QuantizedDenseLayer>(*dense));
            } else {
                quantized->add(cloned->layer(i).clone());
            }
        }
        cloned = std::move(quantized);
    }

    compiled.model_ = std::move(cloned);

    // Step 4: Pre-allocate buffers
    const size_t n_layers = compiled.model_->size();
    compiled.buffers_.reserve(n_layers);

    std::vector<size_t> current_shape = input_shape;
    for (size_t i = 0; i < n_layers; ++i) {
        current_shape = compiled.model_->layer(i).output_shape(current_shape);
        compiled.buffers_.emplace_back(current_shape);
    }

    return compiled;
}

} // namespace engine
} // namespace titaninfer
