#pragma once

#include "titaninfer/io/format.hpp"
#include "titaninfer/layers/sequential.hpp"
#include <string>

namespace titaninfer {
namespace io {

/**
 * @brief Serializes a Sequential model to the .titan binary format
 *
 * File format: little-endian, magic "TITN", version 1.
 * Writes layer types, configurations, and contiguous weight data.
 */
class ModelSerializer {
public:
    /**
     * @brief Save a Sequential model to a .titan file
     * @param model The model to serialize
     * @param filepath Output file path (e.g., "model.titan")
     * @throws std::runtime_error if file cannot be opened for writing
     * @throws std::invalid_argument if model contains unsupported layer types
     */
    static void save(const layers::Sequential& model,
                     const std::string& filepath);

private:
    /**
     * @brief Determine the LayerType enum for a given layer
     * @param layer The layer to identify
     * @return The corresponding LayerType
     * @throws std::invalid_argument if the layer type is not recognized
     */
    static LayerType identify_layer_type(const layers::Layer& layer);
};

} // namespace io
} // namespace titaninfer
