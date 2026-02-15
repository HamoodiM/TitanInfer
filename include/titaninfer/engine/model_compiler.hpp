#pragma once

#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace titaninfer {
namespace engine {

struct CompileOptions {
    bool enable_fusion = true;
    bool enable_quantization = false;
};

/**
 * @brief Compiled model with optimized execution plan
 *
 * Pre-allocates all intermediate buffers for zero-alloc inference.
 */
class CompiledModel {
public:
    /// Run inference with pre-allocated buffers
    Tensor predict(const Tensor& input);

    std::string summary() const;
    const std::vector<size_t>& input_shape() const { return input_shape_; }
    size_t layer_count() const;

private:
    friend class ModelCompiler;

    std::unique_ptr<layers::Sequential> model_;
    std::vector<size_t> input_shape_;
    std::vector<Tensor> buffers_;
};

/**
 * @brief Model compilation pass: analyzes and optimizes a Sequential model
 *
 * Pipeline: clone -> fusion -> quantization -> buffer pre-allocation
 */
class ModelCompiler {
public:
    static CompiledModel compile(
        const layers::Sequential& model,
        const std::vector<size_t>& input_shape,
        const CompileOptions& options = {});
};

} // namespace engine
} // namespace titaninfer
