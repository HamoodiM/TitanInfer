#include <gtest/gtest.h>
#include "titaninfer/layers/activation_layer.hpp"
#include <memory>
#include <vector>
#include <cmath>

using namespace titaninfer;
using namespace titaninfer::layers;

// ========================================
// Polymorphic Layer Tests
// ========================================

TEST(LayerTest, PolymorphicForward) {
    std::unique_ptr<Layer> layer = std::make_unique<ReluLayer>();

    Tensor input({4});
    input.data()[0] = -2.0f;
    input.data()[1] = -1.0f;
    input.data()[2] =  1.0f;
    input.data()[3] =  2.0f;

    Tensor output({4});
    layer->forward(input, output);

    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 2.0f);
}

TEST(LayerTest, LayerNames) {
    ReluLayer relu;
    SigmoidLayer sigmoid;
    TanhLayer tanh_layer;
    SoftmaxLayer softmax;

    EXPECT_EQ(relu.name(), "ReLU");
    EXPECT_EQ(sigmoid.name(), "Sigmoid");
    EXPECT_EQ(tanh_layer.name(), "Tanh");
    EXPECT_EQ(softmax.name(), "Softmax");
}

TEST(LayerTest, Pipeline) {
    // Compose ReLU -> Sigmoid
    std::vector<std::unique_ptr<Layer>> pipeline;
    pipeline.push_back(std::make_unique<ReluLayer>());
    pipeline.push_back(std::make_unique<SigmoidLayer>());

    Tensor input({3});
    input.data()[0] = -1.0f;
    input.data()[1] =  0.0f;
    input.data()[2] =  2.0f;

    Tensor temp({3});
    Tensor output({3});

    pipeline[0]->forward(input, temp);
    pipeline[1]->forward(temp, output);

    // After ReLU: [0, 0, 2]
    // After Sigmoid: [0.5, 0.5, sigmoid(2)]
    EXPECT_NEAR(output.data()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[2], 1.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
}

TEST(LayerTest, SigmoidLayerForward) {
    SigmoidLayer layer;

    Tensor input({3});
    input.data()[0] = 0.0f;
    input.data()[1] = 1.0f;
    input.data()[2] = -1.0f;

    Tensor output({3});
    layer.forward(input, output);

    EXPECT_NEAR(output.data()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
    EXPECT_NEAR(output.data()[2], 1.0f / (1.0f + std::exp(1.0f)), 1e-6f);
}

TEST(LayerTest, TanhLayerForward) {
    TanhLayer layer;

    Tensor input({3});
    input.data()[0] = 0.0f;
    input.data()[1] = 1.0f;
    input.data()[2] = -1.0f;

    Tensor output({3});
    layer.forward(input, output);

    EXPECT_NEAR(output.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(output.data()[1], std::tanh(1.0f), 1e-6f);
    EXPECT_NEAR(output.data()[2], std::tanh(-1.0f), 1e-6f);
}

TEST(LayerTest, SoftmaxLayerForward) {
    SoftmaxLayer layer;

    Tensor input({3});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;

    Tensor output({3});
    layer.forward(input, output);

    float sum = output.data()[0] + output.data()[1] + output.data()[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    EXPECT_LT(output.data()[0], output.data()[1]);
    EXPECT_LT(output.data()[1], output.data()[2]);
}
