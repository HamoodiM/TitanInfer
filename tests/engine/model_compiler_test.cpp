#include <gtest/gtest.h>
#include "titaninfer/engine/model_compiler.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/layers/fused_layers.hpp"
#include "titaninfer/layers/quantized_dense_layer.hpp"
#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/layers/pooling_layers.hpp"
#include "titaninfer/layers/flatten_layer.hpp"

#include <cmath>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::engine;

namespace {

std::unique_ptr<Sequential> make_mlp() {
    auto model = std::make_unique<Sequential>();

    auto d1 = std::make_unique<DenseLayer>(4, 8, true);
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) w1.data()[i] = 0.1f * static_cast<float>(i % 5);
    d1->set_weights(w1);
    Tensor b1({8});
    b1.fill(0.01f);
    d1->set_bias(b1);
    model->add(std::move(d1));
    model->add(std::make_unique<ReluLayer>());

    auto d2 = std::make_unique<DenseLayer>(8, 3, true);
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) w2.data()[i] = 0.05f * static_cast<float>(i % 7);
    d2->set_weights(w2);
    Tensor b2({3});
    b2.fill(0.02f);
    d2->set_bias(b2);
    model->add(std::move(d2));
    model->add(std::make_unique<SoftmaxLayer>());

    return model;
}

} // anonymous namespace

TEST(ModelCompilerTest, CompileBasicMLP) {
    auto model = make_mlp();
    auto compiled = ModelCompiler::compile(*model, {4});

    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i);

    Tensor result = compiled.predict(input);
    EXPECT_EQ(result.shape().size(), 1u);
    EXPECT_EQ(result.shape()[0], 3u);

    // Softmax output should sum to ~1
    float sum = 0;
    for (size_t i = 0; i < 3; ++i) sum += result.data()[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(ModelCompilerTest, FusionEnabled) {
    auto model = make_mlp();
    CompileOptions opts;
    opts.enable_fusion = true;
    opts.enable_quantization = false;

    auto compiled = ModelCompiler::compile(*model, {4}, opts);
    // Dense+ReLU should be fused, Dense+Softmax should not
    // Result: FusedDenseRelu, Dense, Softmax = 3 layers
    EXPECT_EQ(compiled.layer_count(), 3u);
}

TEST(ModelCompilerTest, FusionDisabled) {
    auto model = make_mlp();
    CompileOptions opts;
    opts.enable_fusion = false;
    opts.enable_quantization = false;

    auto compiled = ModelCompiler::compile(*model, {4}, opts);
    EXPECT_EQ(compiled.layer_count(), 4u);  // Dense, ReLU, Dense, Softmax
}

TEST(ModelCompilerTest, QuantizationEnabled) {
    Sequential model;
    auto dense = std::make_unique<DenseLayer>(4, 3, true);
    Tensor w({3, 4});
    for (size_t i = 0; i < 12; ++i) w.data()[i] = static_cast<float>(i) * 0.1f;
    dense->set_weights(w);
    Tensor b({3}); b.fill(0.01f);
    dense->set_bias(b);
    model.add(std::move(dense));

    CompileOptions opts;
    opts.enable_fusion = false;
    opts.enable_quantization = true;

    auto compiled = ModelCompiler::compile(model, {4}, opts);
    EXPECT_EQ(compiled.layer_count(), 1u);

    Tensor input({4});
    input.fill(1.0f);
    Tensor result = compiled.predict(input);
    EXPECT_EQ(result.shape()[0], 3u);
}

TEST(ModelCompilerTest, PredictMatchesOriginal) {
    auto model = make_mlp();

    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i) * 0.5f;

    Tensor original = model->forward(input);

    CompileOptions opts;
    opts.enable_fusion = true;
    opts.enable_quantization = false;
    auto compiled = ModelCompiler::compile(*model, {4}, opts);
    Tensor compiled_out = compiled.predict(input);

    ASSERT_EQ(original.shape(), compiled_out.shape());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(compiled_out.data()[i], original.data()[i], 1e-5f);
    }
}

TEST(ModelCompilerTest, EmptyModelThrows) {
    Sequential empty;
    EXPECT_THROW(ModelCompiler::compile(empty, {4}), std::invalid_argument);
}

TEST(ModelCompilerTest, InputShapeMismatch) {
    auto model = make_mlp();
    auto compiled = ModelCompiler::compile(*model, {4});

    Tensor bad_input({8});  // wrong shape
    EXPECT_THROW(compiled.predict(bad_input), std::invalid_argument);
}

TEST(ModelCompilerTest, CNNModel) {
    Sequential model;

    auto conv = std::make_unique<Conv2DLayer>(1, 4, 3, 1, ops::PaddingMode::VALID, true);
    Tensor cw({4, 1, 3, 3});
    for (size_t i = 0; i < cw.size(); ++i) cw.data()[i] = 0.01f * static_cast<float>(i % 9);
    conv->set_weights(cw);
    Tensor cb({4}); cb.fill(0.01f);
    conv->set_bias(cb);
    model.add(std::move(conv));

    model.add(std::make_unique<ReluLayer>());
    model.add(std::make_unique<MaxPool2DLayer>(2, 2));
    model.add(std::make_unique<FlattenLayer>());

    // After conv: (4, 6, 6), after pool: (4, 3, 3), after flatten: (36)
    auto dense = std::make_unique<DenseLayer>(36, 2, true);
    Tensor dw({2, 36});
    for (size_t i = 0; i < dw.size(); ++i) dw.data()[i] = 0.01f;
    dense->set_weights(dw);
    Tensor db({2}); db.fill(0.01f);
    dense->set_bias(db);
    model.add(std::move(dense));

    auto compiled = ModelCompiler::compile(model, {1, 8, 8});

    Tensor input({1, 8, 8});
    input.fill(1.0f);

    Tensor result = compiled.predict(input);
    EXPECT_EQ(result.shape().size(), 1u);
    EXPECT_EQ(result.shape()[0], 2u);
}

TEST(ModelCompilerTest, Summary) {
    auto model = make_mlp();
    auto compiled = ModelCompiler::compile(*model, {4});
    std::string summary = compiled.summary();
    EXPECT_FALSE(summary.empty());
}
