#include "titaninfer/engine/fusion.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/layers/fused_layers.hpp"

namespace titaninfer {
namespace engine {

std::unique_ptr<layers::Sequential> apply_fusion(
    const layers::Sequential& model) {

    auto result = std::make_unique<layers::Sequential>();
    const size_t n = model.size();

    size_t i = 0;
    while (i < n) {
        const auto& current = model.layer(i);

        // Try to fuse Dense + activation
        if (i + 1 < n) {
            const auto* dense = dynamic_cast<const layers::DenseLayer*>(&current);
            if (dense) {
                const auto& next = model.layer(i + 1);

                // Dense + ReLU
                if (dynamic_cast<const layers::ReluLayer*>(&next)) {
                    result->add(std::make_unique<layers::FusedDenseReluLayer>(*dense));
                    i += 2;
                    continue;
                }

                // Dense + Sigmoid
                if (dynamic_cast<const layers::SigmoidLayer*>(&next)) {
                    result->add(std::make_unique<layers::FusedDenseSigmoidLayer>(*dense));
                    i += 2;
                    continue;
                }
            }
        }

        // No fusion possible — clone the layer
        result->add(current.clone());
        ++i;
    }

    return result;
}

} // namespace engine
} // namespace titaninfer
