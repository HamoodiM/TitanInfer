#pragma once

#include "titaninfer/layers/sequential.hpp"
#include <memory>

namespace titaninfer {
namespace engine {

/**
 * @brief Apply operator fusion optimizations to a Sequential model
 *
 * Detects and merges consecutive layer patterns:
 * - Dense + ReLU -> FusedDenseReluLayer
 * - Dense + Sigmoid -> FusedDenseSigmoidLayer
 *
 * @param model Source model (unmodified)
 * @return New Sequential with fused layers where applicable
 */
std::unique_ptr<layers::Sequential> apply_fusion(
    const layers::Sequential& model);

} // namespace engine
} // namespace titaninfer
