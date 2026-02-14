#pragma once

#include "titaninfer/io/format.hpp"
#include "titaninfer/layers/sequential.hpp"
#include <memory>
#include <string>

namespace titaninfer {
namespace io {

/**
 * @brief Deserializes a Sequential model from the .titan binary format
 *
 * Validates magic number, format version, and layer structure.
 * Reconstructs the full model architecture with loaded weights.
 */
class ModelParser {
public:
    /**
     * @brief Load a Sequential model from a .titan file
     * @param filepath Path to the .titan file
     * @return A fully-constructed Sequential model with loaded weights
     * @throws std::runtime_error if file cannot be opened or is corrupted
     * @throws std::runtime_error if format version is unsupported
     * @throws std::runtime_error if layer type ID is unknown
     * @throws std::runtime_error if file ends prematurely (truncated)
     */
    static std::unique_ptr<layers::Sequential> load(
        const std::string& filepath);
};

} // namespace io
} // namespace titaninfer
