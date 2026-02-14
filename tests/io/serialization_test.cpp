#include <gtest/gtest.h>
#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/io/model_parser.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <memory>
#include <fstream>
#include <cstdio>
#include <cstring>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::io;

// RAII helper for temporary test files
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& name) : path(name) {}
    ~TempFile() { std::remove(path.c_str()); }
};

// Helper: build a known 4-layer MLP with deterministic weights
static Sequential make_test_mlp() {
    Sequential model;

    auto dense1 = std::make_unique<DenseLayer>(4, 8);
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.1f * static_cast<float>(i + 1);
    }
    dense1->set_weights(w1);
    Tensor b1({8});
    for (size_t i = 0; i < b1.size(); ++i) {
        b1.data()[i] = 0.01f * static_cast<float>(i);
    }
    dense1->set_bias(b1);
    model.add(std::move(dense1));

    model.add(std::make_unique<ReluLayer>());

    auto dense2 = std::make_unique<DenseLayer>(8, 3, false); // no bias
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.05f * static_cast<float>(i + 1);
    }
    dense2->set_weights(w2);
    model.add(std::move(dense2));

    model.add(std::make_unique<SoftmaxLayer>());

    return model;
}

// ============================================================
// Round-trip tests
// ============================================================

TEST(SerializationTest, RoundTripMLPArchitecture) {
    TempFile tmp("test_rt_arch.titan");
    Sequential original = make_test_mlp();

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 4u);
    EXPECT_EQ(loaded->layer(0).name(), "Dense(4, 8)");
    EXPECT_EQ(loaded->layer(1).name(), "ReLU");
    EXPECT_EQ(loaded->layer(2).name(), "Dense(8, 3)");
    EXPECT_EQ(loaded->layer(3).name(), "Softmax");
}

TEST(SerializationTest, RoundTripDenseWeights) {
    TempFile tmp("test_rt_weights.titan");
    Sequential original = make_test_mlp();

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    const auto& orig_dense =
        static_cast<const DenseLayer&>(original.layer(0));
    const auto& load_dense =
        static_cast<const DenseLayer&>(loaded->layer(0));

    EXPECT_EQ(load_dense.in_features(), orig_dense.in_features());
    EXPECT_EQ(load_dense.out_features(), orig_dense.out_features());
    EXPECT_EQ(load_dense.weights().shape(), orig_dense.weights().shape());

    for (size_t i = 0; i < orig_dense.weights().size(); ++i) {
        EXPECT_FLOAT_EQ(load_dense.weights().data()[i],
                        orig_dense.weights().data()[i]);
    }
}

TEST(SerializationTest, RoundTripDenseBias) {
    TempFile tmp("test_rt_bias.titan");
    Sequential original = make_test_mlp();

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    const auto& orig_dense =
        static_cast<const DenseLayer&>(original.layer(0));
    const auto& load_dense =
        static_cast<const DenseLayer&>(loaded->layer(0));

    EXPECT_TRUE(load_dense.has_bias());
    EXPECT_EQ(load_dense.bias().shape(), orig_dense.bias().shape());

    for (size_t i = 0; i < orig_dense.bias().size(); ++i) {
        EXPECT_FLOAT_EQ(load_dense.bias().data()[i],
                        orig_dense.bias().data()[i]);
    }
}

TEST(SerializationTest, RoundTripNoBias) {
    TempFile tmp("test_rt_nobias.titan");
    Sequential original = make_test_mlp();

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    // Layer index 2 is Dense(8,3) with no bias
    const auto& load_dense =
        static_cast<const DenseLayer&>(loaded->layer(2));

    EXPECT_FALSE(load_dense.has_bias());
    EXPECT_EQ(load_dense.in_features(), 8u);
    EXPECT_EQ(load_dense.out_features(), 3u);

    const auto& orig_dense =
        static_cast<const DenseLayer&>(original.layer(2));
    for (size_t i = 0; i < orig_dense.weights().size(); ++i) {
        EXPECT_FLOAT_EQ(load_dense.weights().data()[i],
                        orig_dense.weights().data()[i]);
    }
}

TEST(SerializationTest, RoundTripForwardEquivalence) {
    TempFile tmp("test_rt_forward.titan");
    Sequential original = make_test_mlp();

    // Run forward on original
    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    Tensor orig_output = original.forward(input);

    // Save, load, run forward on loaded model
    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);
    Tensor load_output = loaded->forward(input);

    EXPECT_EQ(load_output.shape(), orig_output.shape());
    for (size_t i = 0; i < orig_output.size(); ++i) {
        EXPECT_FLOAT_EQ(load_output.data()[i], orig_output.data()[i]);
    }
}

TEST(SerializationTest, RoundTripSingleDense) {
    TempFile tmp("test_rt_single.titan");
    Sequential original;
    auto dense = std::make_unique<DenseLayer>(3, 2);
    Tensor w({2, 3});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f; w.data()[2] = 3.0f;
    w.data()[3] = 4.0f; w.data()[4] = 5.0f; w.data()[5] = 6.0f;
    dense->set_weights(w);
    Tensor b({2});
    b.data()[0] = 0.5f; b.data()[1] = -0.5f;
    dense->set_bias(b);
    original.add(std::move(dense));

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 1u);
    const auto& ld = static_cast<const DenseLayer&>(loaded->layer(0));
    EXPECT_EQ(ld.in_features(), 3u);
    EXPECT_EQ(ld.out_features(), 2u);
    EXPECT_FLOAT_EQ(ld.weights().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(ld.weights().data()[5], 6.0f);
    EXPECT_FLOAT_EQ(ld.bias().data()[0], 0.5f);
    EXPECT_FLOAT_EQ(ld.bias().data()[1], -0.5f);
}

TEST(SerializationTest, RoundTripActivationsOnly) {
    TempFile tmp("test_rt_actonly.titan");
    Sequential original;
    original.add(std::make_unique<ReluLayer>());
    original.add(std::make_unique<SigmoidLayer>());

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 2u);
    EXPECT_EQ(loaded->layer(0).name(), "ReLU");
    EXPECT_EQ(loaded->layer(1).name(), "Sigmoid");
}

TEST(SerializationTest, RoundTripAllActivations) {
    TempFile tmp("test_rt_allact.titan");
    Sequential original;
    auto d1 = std::make_unique<DenseLayer>(4, 4);
    original.add(std::move(d1));
    original.add(std::make_unique<ReluLayer>());
    auto d2 = std::make_unique<DenseLayer>(4, 4);
    original.add(std::move(d2));
    original.add(std::make_unique<SigmoidLayer>());
    auto d3 = std::make_unique<DenseLayer>(4, 4);
    original.add(std::move(d3));
    original.add(std::make_unique<TanhLayer>());
    auto d4 = std::make_unique<DenseLayer>(4, 4);
    original.add(std::move(d4));
    original.add(std::make_unique<SoftmaxLayer>());

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 8u);
    EXPECT_EQ(loaded->layer(1).name(), "ReLU");
    EXPECT_EQ(loaded->layer(3).name(), "Sigmoid");
    EXPECT_EQ(loaded->layer(5).name(), "Tanh");
    EXPECT_EQ(loaded->layer(7).name(), "Softmax");
}

TEST(SerializationTest, RoundTripEmptyModel) {
    TempFile tmp("test_rt_empty.titan");
    Sequential original;

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 0u);
    EXPECT_TRUE(loaded->empty());
}

TEST(SerializationTest, RoundTripLargeModel) {
    TempFile tmp("test_rt_large.titan");
    Sequential original;

    auto d1 = std::make_unique<DenseLayer>(256, 512);
    // Set distinct weight values
    for (size_t i = 0; i < d1->weights().size(); ++i) {
        // Use a pattern that won't overflow float
        const_cast<float*>(d1->weights().data())[i] =
            0.001f * static_cast<float>(i % 1000);
    }
    original.add(std::move(d1));
    original.add(std::make_unique<ReluLayer>());

    auto d2 = std::make_unique<DenseLayer>(512, 10);
    original.add(std::move(d2));

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->size(), 3u);

    const auto& orig_d1 =
        static_cast<const DenseLayer&>(original.layer(0));
    const auto& load_d1 =
        static_cast<const DenseLayer&>(loaded->layer(0));
    EXPECT_EQ(load_d1.in_features(), 256u);
    EXPECT_EQ(load_d1.out_features(), 512u);

    // Spot-check some weight values
    EXPECT_FLOAT_EQ(load_d1.weights().data()[0],
                    orig_d1.weights().data()[0]);
    EXPECT_FLOAT_EQ(load_d1.weights().data()[999],
                    orig_d1.weights().data()[999]);
    EXPECT_FLOAT_EQ(load_d1.weights().data()[131071],
                    orig_d1.weights().data()[131071]);
}

TEST(SerializationTest, RoundTripParameterCount) {
    TempFile tmp("test_rt_params.titan");
    Sequential original = make_test_mlp();

    ModelSerializer::save(original, tmp.path);
    auto loaded = ModelParser::load(tmp.path);

    EXPECT_EQ(loaded->total_parameters(), original.total_parameters());
}

// ============================================================
// Error condition tests
// ============================================================

TEST(SerializationTest, LoadNonexistentFile) {
    EXPECT_THROW(ModelParser::load("nonexistent_file.titan"),
                 std::runtime_error);
}

TEST(SerializationTest, LoadBadMagic) {
    TempFile tmp("test_err_magic.titan");
    {
        std::ofstream out(tmp.path, std::ios::binary);
        out.write("XXXX", 4);                   // wrong magic
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), 4);
        uint32_t layers = 0;
        out.write(reinterpret_cast<const char*>(&layers), 4);
    }
    EXPECT_THROW(ModelParser::load(tmp.path), std::runtime_error);
}

TEST(SerializationTest, LoadUnsupportedVersion) {
    TempFile tmp("test_err_version.titan");
    {
        std::ofstream out(tmp.path, std::ios::binary);
        out.write("TITN", 4);
        uint32_t version = 99;
        out.write(reinterpret_cast<const char*>(&version), 4);
        uint32_t layers = 0;
        out.write(reinterpret_cast<const char*>(&layers), 4);
    }
    EXPECT_THROW(ModelParser::load(tmp.path), std::runtime_error);
}

TEST(SerializationTest, LoadUnknownLayerType) {
    TempFile tmp("test_err_layertype.titan");
    {
        std::ofstream out(tmp.path, std::ios::binary);
        out.write("TITN", 4);
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), 4);
        uint32_t layers = 1;
        out.write(reinterpret_cast<const char*>(&layers), 4);
        uint32_t bad_type = 99;
        out.write(reinterpret_cast<const char*>(&bad_type), 4);
    }
    EXPECT_THROW(ModelParser::load(tmp.path), std::runtime_error);
}

TEST(SerializationTest, LoadTruncatedHeader) {
    TempFile tmp("test_err_truncheader.titan");
    {
        std::ofstream out(tmp.path, std::ios::binary);
        out.write("TITN", 4);
        // Missing version and layer_count
    }
    EXPECT_THROW(ModelParser::load(tmp.path), std::runtime_error);
}

TEST(SerializationTest, LoadTruncatedWeights) {
    TempFile tmp("test_err_truncweights.titan");
    {
        std::ofstream out(tmp.path, std::ios::binary);
        out.write("TITN", 4);
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), 4);
        uint32_t layers = 1;
        out.write(reinterpret_cast<const char*>(&layers), 4);
        // Dense layer header
        uint32_t type = 1;
        out.write(reinterpret_cast<const char*>(&type), 4);
        uint32_t in_f = 4;
        out.write(reinterpret_cast<const char*>(&in_f), 4);
        uint32_t out_f = 3;
        out.write(reinterpret_cast<const char*>(&out_f), 4);
        uint8_t has_bias = 1;
        out.write(reinterpret_cast<const char*>(&has_bias), 1);
        // Write only 2 floats instead of 4*3=12
        float partial[2] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(partial), sizeof(partial));
    }
    EXPECT_THROW(ModelParser::load(tmp.path), std::runtime_error);
}

// ============================================================
// Binary format validation tests
// ============================================================

TEST(SerializationTest, BinaryFormatMagic) {
    TempFile tmp("test_fmt_magic.titan");
    Sequential model = make_test_mlp();
    ModelSerializer::save(model, tmp.path);

    std::ifstream in(tmp.path, std::ios::binary);
    char magic[4];
    in.read(magic, 4);
    EXPECT_EQ(std::memcmp(magic, "TITN", 4), 0);
}

TEST(SerializationTest, BinaryFormatVersion) {
    TempFile tmp("test_fmt_version.titan");
    Sequential model = make_test_mlp();
    ModelSerializer::save(model, tmp.path);

    std::ifstream in(tmp.path, std::ios::binary);
    in.seekg(4); // skip magic
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), 4);
    EXPECT_EQ(version, 1u);
}

TEST(SerializationTest, BinaryFormatLayerCount) {
    TempFile tmp("test_fmt_count.titan");
    Sequential model = make_test_mlp();
    ModelSerializer::save(model, tmp.path);

    std::ifstream in(tmp.path, std::ios::binary);
    in.seekg(8); // skip magic + version
    uint32_t count;
    in.read(reinterpret_cast<char*>(&count), 4);
    EXPECT_EQ(count, 4u);
}
