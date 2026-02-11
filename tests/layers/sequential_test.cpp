#include <gtest/gtest.h>
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <memory>
#include <cmath>

using namespace titaninfer;
using namespace titaninfer::layers;

// ========================================
// DenseLayer Tests
// ========================================

TEST(DenseLayerTest, Construction) {
    DenseLayer layer(4, 3);
    EXPECT_EQ(layer.in_features(), 4u);
    EXPECT_EQ(layer.out_features(), 3u);
    EXPECT_TRUE(layer.has_bias());
    EXPECT_EQ(layer.weights().shape(), (std::vector<size_t>{3, 4}));
    EXPECT_EQ(layer.bias().shape(), (std::vector<size_t>{3}));
    EXPECT_EQ(layer.parameter_count(), 3u * 4u + 3u);
}

TEST(DenseLayerTest, ConstructionNoBias) {
    DenseLayer layer(4, 3, false);
    EXPECT_FALSE(layer.has_bias());
    EXPECT_EQ(layer.parameter_count(), 12u);
}

TEST(DenseLayerTest, SetWeightsAndBias) {
    DenseLayer layer(2, 3);

    Tensor w({3, 2});
    w.data()[0] = 1.0f; w.data()[1] = 0.0f;
    w.data()[2] = 0.0f; w.data()[3] = 1.0f;
    w.data()[4] = 1.0f; w.data()[5] = 1.0f;
    layer.set_weights(w);

    Tensor b({3});
    b.data()[0] = 0.1f; b.data()[1] = 0.2f; b.data()[2] = 0.3f;
    layer.set_bias(b);

    EXPECT_FLOAT_EQ(layer.weights().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(layer.bias().data()[2], 0.3f);
}

TEST(DenseLayerTest, SetWeightsShapeMismatch) {
    DenseLayer layer(4, 3);
    Tensor bad({4, 3});
    EXPECT_THROW(layer.set_weights(bad), std::invalid_argument);
}

TEST(DenseLayerTest, SetBiasDisabled) {
    DenseLayer layer(4, 3, false);
    Tensor b({3});
    EXPECT_THROW(layer.set_bias(b), std::invalid_argument);
}

TEST(DenseLayerTest, Forward1D) {
    // W = [[1, 0], [0, 1], [1, 1]], b = [0.1, 0.2, 0.3]
    // x = [2.0, 3.0]
    // y = W @ x + b = [2+0+0.1, 0+3+0.2, 2+3+0.3] = [2.1, 3.2, 5.3]
    DenseLayer layer(2, 3);

    Tensor w({3, 2});
    w.data()[0] = 1.0f; w.data()[1] = 0.0f;
    w.data()[2] = 0.0f; w.data()[3] = 1.0f;
    w.data()[4] = 1.0f; w.data()[5] = 1.0f;
    layer.set_weights(w);

    Tensor b({3});
    b.data()[0] = 0.1f; b.data()[1] = 0.2f; b.data()[2] = 0.3f;
    layer.set_bias(b);

    Tensor input({2});
    input.data()[0] = 2.0f;
    input.data()[1] = 3.0f;

    Tensor output({3});
    layer.forward(input, output);

    EXPECT_EQ(output.shape(), (std::vector<size_t>{3}));
    EXPECT_NEAR(output.data()[0], 2.1f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 3.2f, 1e-6f);
    EXPECT_NEAR(output.data()[2], 5.3f, 1e-6f);
}

TEST(DenseLayerTest, Forward2D) {
    // W = [[1, 0], [0, 1]], b = [0.5, -0.5]
    // X = [[1, 2], [3, 4]]
    // Y = X @ W^T + b = [[1+0.5, 2-0.5], [3+0.5, 4-0.5]] = [[1.5, 1.5], [3.5, 3.5]]
    DenseLayer layer(2, 2);

    Tensor w({2, 2});
    w.data()[0] = 1.0f; w.data()[1] = 0.0f;
    w.data()[2] = 0.0f; w.data()[3] = 1.0f;
    layer.set_weights(w);

    Tensor b({2});
    b.data()[0] = 0.5f; b.data()[1] = -0.5f;
    layer.set_bias(b);

    Tensor input({2, 2});
    input.data()[0] = 1.0f; input.data()[1] = 2.0f;
    input.data()[2] = 3.0f; input.data()[3] = 4.0f;

    Tensor output({2, 2});
    layer.forward(input, output);

    EXPECT_EQ(output.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_NEAR(output.data()[0], 1.5f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 1.5f, 1e-6f);
    EXPECT_NEAR(output.data()[2], 3.5f, 1e-6f);
    EXPECT_NEAR(output.data()[3], 3.5f, 1e-6f);
}

TEST(DenseLayerTest, Forward1DNoBias) {
    DenseLayer layer(2, 2, false);

    Tensor w({2, 2});
    w.data()[0] = 1.0f; w.data()[1] = 2.0f;
    w.data()[2] = 3.0f; w.data()[3] = 4.0f;
    layer.set_weights(w);

    Tensor input({2});
    input.data()[0] = 1.0f;
    input.data()[1] = 1.0f;

    Tensor output({2});
    layer.forward(input, output);

    // y = W @ x = [1+2, 3+4] = [3, 7]
    EXPECT_NEAR(output.data()[0], 3.0f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 7.0f, 1e-6f);
}

TEST(DenseLayerTest, ForwardDimensionMismatch) {
    DenseLayer layer(4, 3);
    Tensor input({5});  // expects 4
    Tensor output({3});
    EXPECT_THROW(layer.forward(input, output), std::invalid_argument);
}

TEST(DenseLayerTest, Forward3DThrows) {
    DenseLayer layer(4, 3);
    Tensor input({2, 3, 4});
    Tensor output({1});
    EXPECT_THROW(layer.forward(input, output), std::invalid_argument);
}

TEST(DenseLayerTest, Name) {
    DenseLayer layer(4, 8);
    EXPECT_EQ(layer.name(), "Dense(4, 8)");
}

TEST(DenseLayerTest, OutputShape) {
    DenseLayer layer(4, 8);
    EXPECT_EQ(layer.output_shape({4}), (std::vector<size_t>{8}));
    EXPECT_EQ(layer.output_shape({2, 4}), (std::vector<size_t>{2, 8}));
    EXPECT_THROW(layer.output_shape({2, 3, 4}), std::invalid_argument);
}

TEST(DenseLayerTest, ZeroWeightsGiveBiasOnly) {
    DenseLayer layer(3, 2);

    Tensor b({2});
    b.data()[0] = 1.0f; b.data()[1] = 2.0f;
    layer.set_bias(b);

    Tensor input({3});
    input.data()[0] = 99.0f; input.data()[1] = 99.0f; input.data()[2] = 99.0f;

    Tensor output({2});
    layer.forward(input, output);

    // Zero weights, so output = 0 + bias
    EXPECT_NEAR(output.data()[0], 1.0f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 2.0f, 1e-6f);
}

// ========================================
// Sequential Tests
// ========================================

TEST(SequentialTest, EmptyModelThrows) {
    Sequential model;
    Tensor input({4});
    EXPECT_THROW(model.forward(input), std::runtime_error);
}

TEST(SequentialTest, SizeAndEmpty) {
    Sequential model;
    EXPECT_TRUE(model.empty());
    EXPECT_EQ(model.size(), 0u);

    model.add(std::make_unique<ReluLayer>());
    EXPECT_FALSE(model.empty());
    EXPECT_EQ(model.size(), 1u);
}

TEST(SequentialTest, NullLayerThrows) {
    Sequential model;
    EXPECT_THROW(model.add(nullptr), std::invalid_argument);
}

TEST(SequentialTest, LayerAccess) {
    Sequential model;
    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<SigmoidLayer>());

    EXPECT_EQ(model.layer(0).name(), "ReLU");
    EXPECT_EQ(model.layer(1).name(), "Sigmoid");
    EXPECT_THROW(model.layer(2), std::out_of_range);
}

TEST(SequentialTest, SingleLayer) {
    Sequential model;
    model.add(std::make_unique<ReluLayer>());

    Tensor input({3});
    input.data()[0] = -1.0f;
    input.data()[1] = 0.0f;
    input.data()[2] = 2.0f;

    Tensor output = model.forward(input);

    EXPECT_EQ(output.shape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 2.0f);
}

TEST(SequentialTest, ActivationOnlyPipeline) {
    // ReLU -> Sigmoid (2 layers, tests even-count ping-pong)
    Sequential model;
    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<SigmoidLayer>());

    Tensor input({3});
    input.data()[0] = -1.0f;
    input.data()[1] = 0.0f;
    input.data()[2] = 2.0f;

    Tensor output = model.forward(input);

    // After ReLU: [0, 0, 2]
    // After Sigmoid: [0.5, 0.5, sigmoid(2)]
    EXPECT_NEAR(output.data()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[2], 1.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
}

TEST(SequentialTest, MLPForward) {
    // Dense(4,8) -> ReLU -> Dense(8,3) -> Softmax
    Sequential model;
    model.add(std::make_unique<DenseLayer>(4, 8));
    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<DenseLayer>(8, 3));
    model.add(std::make_unique<SoftmaxLayer>());

    // Set weights on dense layers for reproducibility
    auto& dense1 = dynamic_cast<DenseLayer&>(model.layer(0));
    auto& dense2 = dynamic_cast<DenseLayer&>(model.layer(2));

    // Fill dense1 weights: small positive values
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.1f * static_cast<float>((i % 5) + 1);
    }
    dense1.set_weights(w1);

    Tensor b1({8});
    b1.fill(0.01f);
    dense1.set_bias(b1);

    // Fill dense2 weights
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.05f * static_cast<float>((i % 4) + 1);
    }
    dense2.set_weights(w2);

    Tensor b2({3});
    b2.fill(0.0f);
    dense2.set_bias(b2);

    // Input
    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;

    Tensor output = model.forward(input);

    // Verify output shape
    EXPECT_EQ(output.shape(), (std::vector<size_t>{3}));

    // Verify softmax properties: all in [0,1], sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_GE(output.data()[i], 0.0f);
        EXPECT_LE(output.data()[i], 1.0f);
        sum += output.data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(SequentialTest, MLPBatched) {
    // Same MLP with 2D input (batch=2, features=4)
    Sequential model;
    model.add(std::make_unique<DenseLayer>(4, 8));
    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<DenseLayer>(8, 3));
    model.add(std::make_unique<SoftmaxLayer>());

    // Set simple weights
    auto& dense1 = dynamic_cast<DenseLayer&>(model.layer(0));
    auto& dense2 = dynamic_cast<DenseLayer&>(model.layer(2));

    Tensor w1({8, 4});
    w1.fill(0.1f);
    dense1.set_weights(w1);

    Tensor w2({3, 8});
    w2.fill(0.05f);
    dense2.set_weights(w2);

    Tensor input({2, 4});
    input.data()[0] = 1.0f; input.data()[1] = 2.0f;
    input.data()[2] = 3.0f; input.data()[3] = 4.0f;
    input.data()[4] = 0.5f; input.data()[5] = 1.5f;
    input.data()[6] = 2.5f; input.data()[7] = 3.5f;

    Tensor output = model.forward(input);

    // Verify output shape (batch=2, classes=3)
    EXPECT_EQ(output.shape(), (std::vector<size_t>{2, 3}));

    // Verify softmax properties per row
    for (size_t r = 0; r < 2; ++r) {
        float row_sum = 0.0f;
        for (size_t c = 0; c < 3; ++c) {
            float val = output.data()[r * 3 + c];
            EXPECT_GE(val, 0.0f);
            EXPECT_LE(val, 1.0f);
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-6f);
    }
}

TEST(SequentialTest, Summary) {
    Sequential model;
    model.add(std::make_unique<DenseLayer>(4, 8));
    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<DenseLayer>(8, 3));
    model.add(std::make_unique<SoftmaxLayer>());

    std::string s = model.summary({4});

    EXPECT_NE(s.find("Dense(4, 8)"), std::string::npos);
    EXPECT_NE(s.find("ReLU"), std::string::npos);
    EXPECT_NE(s.find("Dense(8, 3)"), std::string::npos);
    EXPECT_NE(s.find("Softmax"), std::string::npos);

    // Total params: (4*8+8) + 0 + (8*3+3) + 0 = 40 + 27 = 67
    EXPECT_EQ(model.total_parameters(), 67u);
}

TEST(SequentialTest, TotalParametersEmpty) {
    Sequential model;
    EXPECT_EQ(model.total_parameters(), 0u);
}
