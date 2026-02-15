#include <gtest/gtest.h>
#include "titaninfer/engine/fusion.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/layers/fused_layers.hpp"

#include <cmath>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::engine;

namespace {

std::unique_ptr<DenseLayer> make_dense(size_t in, size_t out) {
    auto dense = std::make_unique<DenseLayer>(in, out, true);
    Tensor w({out, in});
    for (size_t i = 0; i < w.size(); ++i) {
        w.data()[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
    }
    dense->set_weights(w);
    Tensor b({out});
    for (size_t i = 0; i < out; ++i) b.data()[i] = 0.01f * static_cast<float>(i);
    dense->set_bias(b);
    return dense;
}

} // anonymous namespace

TEST(FusedDenseReluTest, ForwardMatchesUnfused1D) {
    auto dense = make_dense(4, 3);

    // Unfused path
    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i) - 1.0f;

    Tensor dense_out({1}), relu_out({1});
    dense->forward(input, dense_out);
    for (size_t i = 0; i < dense_out.size(); ++i) {
        relu_out = Tensor(dense_out.shape());
        relu_out.data()[i] = std::max(0.0f, dense_out.data()[i]);
    }
    // Actually do it properly
    ops::relu(dense_out, relu_out);

    // Fused path
    FusedDenseReluLayer fused(*dense);
    Tensor fused_out({1});
    fused.forward(input, fused_out);

    ASSERT_EQ(fused_out.shape(), relu_out.shape());
    for (size_t i = 0; i < relu_out.size(); ++i) {
        EXPECT_FLOAT_EQ(fused_out.data()[i], relu_out.data()[i]);
    }
}

TEST(FusedDenseReluTest, ForwardMatchesUnfused2D) {
    auto dense = make_dense(4, 3);

    Tensor input({2, 4});
    for (size_t i = 0; i < 8; ++i) input.data()[i] = static_cast<float>(i) * 0.5f - 2.0f;

    Tensor dense_out({1}), relu_out({1});
    dense->forward(input, dense_out);
    ops::relu(dense_out, relu_out);

    FusedDenseReluLayer fused(*dense);
    Tensor fused_out({1});
    fused.forward(input, fused_out);

    ASSERT_EQ(fused_out.shape(), relu_out.shape());
    for (size_t i = 0; i < relu_out.size(); ++i) {
        EXPECT_FLOAT_EQ(fused_out.data()[i], relu_out.data()[i]);
    }
}

TEST(FusedDenseSigmoidTest, ForwardMatchesUnfused1D) {
    auto dense = make_dense(4, 3);

    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i);

    Tensor dense_out({1}), sig_out({1});
    dense->forward(input, dense_out);
    ops::sigmoid(dense_out, sig_out);

    FusedDenseSigmoidLayer fused(*dense);
    Tensor fused_out({1});
    fused.forward(input, fused_out);

    ASSERT_EQ(fused_out.shape(), sig_out.shape());
    for (size_t i = 0; i < sig_out.size(); ++i) {
        EXPECT_NEAR(fused_out.data()[i], sig_out.data()[i], 1e-6f);
    }
}

TEST(FusionPassTest, DenseReluFusion) {
    Sequential model;
    model.add(make_dense(4, 3));
    model.add(std::make_unique<ReluLayer>());

    auto fused = apply_fusion(model);
    EXPECT_EQ(fused->size(), 1u);
    EXPECT_NE(dynamic_cast<FusedDenseReluLayer*>(&fused->layer(0)), nullptr);
}

TEST(FusionPassTest, DenseSigmoidFusion) {
    Sequential model;
    model.add(make_dense(4, 3));
    model.add(std::make_unique<SigmoidLayer>());

    auto fused = apply_fusion(model);
    EXPECT_EQ(fused->size(), 1u);
    EXPECT_NE(dynamic_cast<FusedDenseSigmoidLayer*>(&fused->layer(0)), nullptr);
}

TEST(FusionPassTest, NoFusionPossible) {
    Sequential model;
    model.add(make_dense(4, 3));
    model.add(std::make_unique<SoftmaxLayer>());

    auto fused = apply_fusion(model);
    EXPECT_EQ(fused->size(), 2u);
}

TEST(FusionPassTest, MultipleFusions) {
    Sequential model;
    model.add(make_dense(4, 3));
    model.add(std::make_unique<ReluLayer>());
    model.add(make_dense(3, 2));
    model.add(std::make_unique<SigmoidLayer>());

    auto fused = apply_fusion(model);
    EXPECT_EQ(fused->size(), 2u);
    EXPECT_NE(dynamic_cast<FusedDenseReluLayer*>(&fused->layer(0)), nullptr);
    EXPECT_NE(dynamic_cast<FusedDenseSigmoidLayer*>(&fused->layer(1)), nullptr);
}

TEST(FusionPassTest, NonConsecutivePatternsPreserved) {
    Sequential model;
    model.add(make_dense(4, 3));
    model.add(std::make_unique<TanhLayer>());  // Not fusable
    model.add(make_dense(3, 2));
    model.add(std::make_unique<ReluLayer>());

    auto fused = apply_fusion(model);
    EXPECT_EQ(fused->size(), 3u);
    EXPECT_NE(dynamic_cast<DenseLayer*>(&fused->layer(0)), nullptr);
    EXPECT_NE(dynamic_cast<TanhLayer*>(&fused->layer(1)), nullptr);
    EXPECT_NE(dynamic_cast<FusedDenseReluLayer*>(&fused->layer(2)), nullptr);
}

TEST(FusionPassTest, EndToEndEquivalence) {
    Sequential model;
    model.add(make_dense(4, 8));
    model.add(std::make_unique<ReluLayer>());
    model.add(make_dense(8, 3));
    model.add(std::make_unique<SoftmaxLayer>());

    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i) * 0.5f;

    Tensor original = model.forward(input);

    auto fused_model = apply_fusion(model);
    Tensor fused_output = fused_model->forward(input);

    ASSERT_EQ(original.shape(), fused_output.shape());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(fused_output.data()[i], original.data()[i], 1e-5f);
    }
}
