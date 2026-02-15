#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/layers/pooling_layers.hpp"
#include "titaninfer/layers/flatten_layer.hpp"
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

            write_floats(out, dense.weights().data(),
                         dense.out_features() * dense.in_features());

            if (dense.has_bias()) {
                write_floats(out, dense.bias().data(),
                             dense.out_features());
            }
        } else if (type == LayerType::CONV2D) {
            const auto& conv =
                static_cast<const layers::Conv2DLayer&>(layer);

            write_value<uint32_t>(out, static_cast<uint32_t>(conv.in_channels()));
            write_value<uint32_t>(out, static_cast<uint32_t>(conv.out_channels()));
            write_value<uint32_t>(out, static_cast<uint32_t>(conv.kernel_h()));
            write_value<uint32_t>(out, static_cast<uint32_t>(conv.kernel_w()));
            write_value<uint32_t>(out, static_cast<uint32_t>(conv.stride_h()));
            write_value<uint32_t>(out, static_cast<uint32_t>(conv.stride_w()));
            write_value<uint8_t>(out,
                conv.padding() == ops::PaddingMode::SAME ? uint8_t{1} : uint8_t{0});
            write_value<uint8_t>(out,
                conv.has_bias() ? uint8_t{1} : uint8_t{0});

            write_floats(out, conv.weights().data(),
                         conv.out_channels() * conv.in_channels() *
                         conv.kernel_h() * conv.kernel_w());

            if (conv.has_bias()) {
                write_floats(out, conv.bias().data(), conv.out_channels());
            }
        } else if (type == LayerType::MAXPOOL2D) {
            const auto& pool =
                static_cast<const layers::MaxPool2DLayer&>(layer);
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.kernel_size()));
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.stride()));
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.padding()));
        } else if (type == LayerType::AVGPOOL2D) {
            const auto& pool =
                static_cast<const layers::AvgPool2DLayer&>(layer);
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.kernel_size()));
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.stride()));
            write_value<uint32_t>(out, static_cast<uint32_t>(pool.padding()));
        }
        // RELU, SIGMOID, TANH, SOFTMAX, FLATTEN have no additional data
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
    if (dynamic_cast<const layers::Conv2DLayer*>(&layer))
        return LayerType::CONV2D;
    if (dynamic_cast<const layers::MaxPool2DLayer*>(&layer))
        return LayerType::MAXPOOL2D;
    if (dynamic_cast<const layers::AvgPool2DLayer*>(&layer))
        return LayerType::AVGPOOL2D;
    if (dynamic_cast<const layers::FlattenLayer*>(&layer))
        return LayerType::FLATTEN;

    throw std::invalid_argument(
        "ModelSerializer: unsupported layer type '" + layer.name() + "'");
}

} // namespace io
} // namespace titaninfer
