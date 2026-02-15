#include <gtest/gtest.h>
#include <fstream>
#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/io/model_parser.hpp"
#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/layers/pooling_layers.hpp"
#include "titaninfer/layers/flatten_layer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"

#include <cstdio>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::io;

namespace {

struct TempFile {
    std::string path;
    TempFile(const std::string& name) : path(name) {}
    ~TempFile() { std::remove(path.c_str()); }
};

} // anonymous namespace

TEST(ConvSerializationTest, RoundTripConv2D) {
    Sequential model;
    auto conv = std::make_unique<Conv2DLayer>(3, 8, 3, 2, ops::PaddingMode::SAME, true);
    Tensor w({8, 3, 3, 3});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = static_cast<float>(i) * 0.01f;
    conv->set_weights(w);
    Tensor b({8});
    for (size_t i = 0; i < 8; ++i) b.data()[i] = static_cast<float>(i) * 0.1f;
    conv->set_bias(b);
    model.add(std::move(conv));

    TempFile tmp("conv_round_trip.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    ASSERT_EQ(loaded->size(), 1u);
    auto* loaded_conv = dynamic_cast<Conv2DLayer*>(&loaded->layer(0));
    ASSERT_NE(loaded_conv, nullptr);
    EXPECT_EQ(loaded_conv->in_channels(), 3u);
    EXPECT_EQ(loaded_conv->out_channels(), 8u);
    EXPECT_EQ(loaded_conv->kernel_h(), 3u);
    EXPECT_EQ(loaded_conv->kernel_w(), 3u);
    EXPECT_EQ(loaded_conv->stride_h(), 2u);
    EXPECT_EQ(loaded_conv->stride_w(), 2u);
    EXPECT_EQ(loaded_conv->padding(), ops::PaddingMode::SAME);
    EXPECT_TRUE(loaded_conv->has_bias());
}

TEST(ConvSerializationTest, RoundTripMaxPool2D) {
    Sequential model;
    model.add(std::make_unique<MaxPool2DLayer>(3, 2, 1));

    TempFile tmp("maxpool_round_trip.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    ASSERT_EQ(loaded->size(), 1u);
    auto* pool = dynamic_cast<MaxPool2DLayer*>(&loaded->layer(0));
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->kernel_size(), 3u);
    EXPECT_EQ(pool->stride(), 2u);
    EXPECT_EQ(pool->padding(), 1u);
}

TEST(ConvSerializationTest, RoundTripAvgPool2D) {
    Sequential model;
    model.add(std::make_unique<AvgPool2DLayer>(2, 2, 0));

    TempFile tmp("avgpool_round_trip.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    ASSERT_EQ(loaded->size(), 1u);
    auto* pool = dynamic_cast<AvgPool2DLayer*>(&loaded->layer(0));
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->kernel_size(), 2u);
    EXPECT_EQ(pool->stride(), 2u);
}

TEST(ConvSerializationTest, RoundTripFlatten) {
    Sequential model;
    model.add(std::make_unique<FlattenLayer>());

    TempFile tmp("flatten_round_trip.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    ASSERT_EQ(loaded->size(), 1u);
    auto* flat = dynamic_cast<FlattenLayer*>(&loaded->layer(0));
    EXPECT_NE(flat, nullptr);
}

TEST(ConvSerializationTest, RoundTripCNNModel) {
    Sequential model;

    auto conv = std::make_unique<Conv2DLayer>(1, 4, 3, 1, ops::PaddingMode::VALID);
    Tensor cw({4, 1, 3, 3});
    for (size_t i = 0; i < cw.size(); ++i) cw.data()[i] = 0.1f;
    conv->set_weights(cw);
    model.add(std::move(conv));

    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<MaxPool2DLayer>(2, 2));
    model.add(std::make_unique<FlattenLayer>());

    // After conv on 8x8: (4,6,6), pool: (4,3,3), flatten: (36)
    auto dense = std::make_unique<DenseLayer>(36, 10, true);
    Tensor dw({10, 36});
    for (size_t i = 0; i < dw.size(); ++i) dw.data()[i] = 0.01f;
    dense->set_weights(dw);
    Tensor db({10}); db.fill(0.1f);
    dense->set_bias(db);
    model.add(std::move(dense));
    model.add(std::make_unique<SoftmaxLayer>());

    TempFile tmp("cnn_model.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 6u);

    // Verify round-trip produces same output
    Tensor input({1, 8, 8});
    input.fill(1.0f);

    Tensor original = model.forward(input);
    Tensor round_tripped = loaded->forward(input);

    ASSERT_EQ(original.shape(), round_tripped.shape());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(round_tripped.data()[i], original.data()[i], 1e-5f);
    }
}

TEST(ConvSerializationTest, WeightPreservation) {
    Sequential model;
    auto conv = std::make_unique<Conv2DLayer>(2, 3, 3, 1, ops::PaddingMode::VALID, true);
    Tensor w({3, 2, 3, 3});
    for (size_t i = 0; i < w.size(); ++i) {
        w.data()[i] = static_cast<float>(i) * 0.123f;
    }
    conv->set_weights(w);
    Tensor b({3});
    b.data()[0] = 1.1f; b.data()[1] = 2.2f; b.data()[2] = 3.3f;
    conv->set_bias(b);
    model.add(std::move(conv));

    TempFile tmp("weight_preservation.titan");
    ModelSerializer::save(model, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    auto* loaded_conv = dynamic_cast<Conv2DLayer*>(&loaded->layer(0));
    ASSERT_NE(loaded_conv, nullptr);

    for (size_t i = 0; i < w.size(); ++i) {
        EXPECT_FLOAT_EQ(loaded_conv->weights().data()[i], w.data()[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(loaded_conv->bias().data()[i], b.data()[i]);
    }
}

TEST(ConvSerializationTest, FormatVersion2) {
    Sequential model;
    model.add(std::make_unique<FlattenLayer>());

    TempFile tmp("version2.titan");
    ModelSerializer::save(model, tmp.path);

    // Read raw bytes to check version
    std::ifstream in(tmp.path, std::ios::binary);
    char magic[4];
    in.read(magic, 4);
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    EXPECT_EQ(magic[0], 'T');
    EXPECT_EQ(magic[1], 'I');
    EXPECT_EQ(magic[2], 'T');
    EXPECT_EQ(magic[3], 'N');
    EXPECT_EQ(version, 2u);
}
