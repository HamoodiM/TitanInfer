#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <fstream>
#include <stdexcept>

namespace titaninfer {
namespace io {

namespace {

template<typename T>
void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_floats(std::ofstream& out, const float* data, size_t count) {
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(float)));
}

} // anonymous namespace

void ModelSerializer::save(const layers::Sequential& model,
                           const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error(
            "ModelSerializer: cannot open file '" + filepath + "' for writing");
    }

    // Write header
    out.write(TITAN_MAGIC, 4);
    write_value<uint32_t>(out, TITAN_FORMAT_VERSION);
    write_value<uint32_t>(out, static_cast<uint32_t>(model.size()));

    // Write each layer
    for (size_t i = 0; i < model.size(); ++i) {
        const auto& layer = model.layer(i);
        LayerType type = identify_layer_type(layer);
        write_value<uint32_t>(out, static_cast<uint32_t>(type));

        if (type == LayerType::DENSE) {
            const auto& dense =
                static_cast<const layers::DenseLayer&>(layer);

            write_value<uint32_t>(out,
                static_cast<uint32_t>(dense.in_features()));
            write_value<uint32_t>(out,
                static_cast<uint32_t>(dense.out_features()));
            write_value<uint8_t>(out,
                dense.has_bias() ? uint8_t{1} : uint8_t{0});

            // Write weight data: (out_features, in_features) contiguous floats
            write_floats(out, dense.weights().data(),
                         dense.out_features() * dense.in_features());

            // Write bias data if present
            if (dense.has_bias()) {
                write_floats(out, dense.bias().data(),
                             dense.out_features());
            }
        }
        // Activation layers have no config or weight data beyond the type tag
    }
}

LayerType ModelSerializer::identify_layer_type(const layers::Layer& layer) {
    if (dynamic_cast<const layers::DenseLayer*>(&layer))
        return LayerType::DENSE;
    if (dynamic_cast<const layers::ReluLayer*>(&layer))
        return LayerType::RELU;
    if (dynamic_cast<const layers::SigmoidLayer*>(&layer))
        return LayerType::SIGMOID;
    if (dynamic_cast<const layers::TanhLayer*>(&layer))
        return LayerType::TANH;
    if (dynamic_cast<const layers::SoftmaxLayer*>(&layer))
        return LayerType::SOFTMAX;

    throw std::invalid_argument(
        "ModelSerializer: unsupported layer type '" + layer.name() + "'");
}

} // namespace io
} // namespace titaninfer
