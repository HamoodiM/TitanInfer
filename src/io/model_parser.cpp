#include "titaninfer/io/model_parser.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace titaninfer {
namespace io {

namespace {

template<typename T>
T read_value(std::ifstream& in) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error(
            "ModelParser: unexpected end of file");
    }
    return value;
}

void read_floats(std::ifstream& in, float* data, size_t count) {
    in.read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error(
            "ModelParser: unexpected end of file while reading weight data");
    }
}

} // anonymous namespace

std::unique_ptr<layers::Sequential> ModelParser::load(
        const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "ModelParser: cannot open file '" + filepath + "' for reading");
    }

    // Read and validate magic number
    char magic[4];
    in.read(magic, 4);
    if (!in) {
        throw std::runtime_error(
            "ModelParser: unexpected end of file while reading header");
    }
    if (std::memcmp(magic, TITAN_MAGIC, 4) != 0) {
        throw std::runtime_error(
            "ModelParser: invalid magic number -- not a .titan file");
    }

    // Read and validate version
    uint32_t version = read_value<uint32_t>(in);
    if (version > TITAN_FORMAT_VERSION) {
        throw std::runtime_error(
            "ModelParser: unsupported format version " +
            std::to_string(version) + " (max supported: " +
            std::to_string(TITAN_FORMAT_VERSION) + ")");
    }

    // Read layer count
    uint32_t layer_count = read_value<uint32_t>(in);

    auto model = std::make_unique<layers::Sequential>();

    for (uint32_t i = 0; i < layer_count; ++i) {
        uint32_t type_id = read_value<uint32_t>(in);
        auto type = static_cast<LayerType>(type_id);

        switch (type) {
            case LayerType::DENSE: {
                uint32_t in_features = read_value<uint32_t>(in);
                uint32_t out_features = read_value<uint32_t>(in);
                uint8_t has_bias = read_value<uint8_t>(in);

                auto dense = std::make_unique<layers::DenseLayer>(
                    static_cast<size_t>(in_features),
                    static_cast<size_t>(out_features),
                    has_bias != 0);

                // Read weights
                Tensor weights({static_cast<size_t>(out_features),
                                static_cast<size_t>(in_features)});
                read_floats(in, weights.data(),
                            static_cast<size_t>(out_features) *
                            static_cast<size_t>(in_features));
                dense->set_weights(weights);

                // Read bias if present
                if (has_bias != 0) {
                    Tensor bias({static_cast<size_t>(out_features)});
                    read_floats(in, bias.data(),
                                static_cast<size_t>(out_features));
                    dense->set_bias(bias);
                }

                model->add(std::move(dense));
                break;
            }
            case LayerType::RELU:
                model->add(std::make_unique<layers::ReluLayer>());
                break;
            case LayerType::SIGMOID:
                model->add(std::make_unique<layers::SigmoidLayer>());
                break;
            case LayerType::TANH:
                model->add(std::make_unique<layers::TanhLayer>());
                break;
            case LayerType::SOFTMAX:
                model->add(std::make_unique<layers::SoftmaxLayer>());
                break;
            default:
                throw std::runtime_error(
                    "ModelParser: unknown layer type ID " +
                    std::to_string(type_id) + " at layer index " +
                    std::to_string(i));
        }
    }

    return model;
}

} // namespace io
} // namespace titaninfer
