#pragma once

#include <cstdint>

namespace titaninfer {
namespace io {

/// Magic number for .titan files: "TITN"
static constexpr char TITAN_MAGIC[4] = {'T', 'I', 'T', 'N'};

/// Current format version
static constexpr uint32_t TITAN_FORMAT_VERSION = 2;

/// Layer type identifiers for the .titan binary format
enum class LayerType : uint32_t {
    DENSE     = 1,
    RELU      = 2,
    SIGMOID   = 3,
    TANH      = 4,
    SOFTMAX   = 5,
    CONV2D    = 6,
    MAXPOOL2D = 7,
    AVGPOOL2D = 8,
    FLATTEN   = 9
};

} // namespace io
} // namespace titaninfer
