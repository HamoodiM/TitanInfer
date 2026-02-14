#include <gtest/gtest.h>
#include "titaninfer/engine/inference_engine.hpp"
#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <memory>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::io;
using namespace titaninfer::engine;

// RAII helper for temporary test files
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& name) : path(name) {}
    ~TempFile() { std::remove(path.c_str()); }
};

// Helper: build and save a known MLP: Dense(4,8) -> ReLU -> Dense(8,3) -> Softmax
static void save_test_mlp(const std::string& filename) {
    Sequential model;

    auto dense1 = std::make_unique<DenseLayer>(4, 8);
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.1f * static_cast<float>((i % 5) + 1);
    }
    dense1->set_weights(w1);
    Tensor b1({8});
    for (size_t i = 0; i < b1.size(); ++i) {
        b1.data()[i] = 0.01f * static_cast<float>(i);
    }
    dense1->set_bias(b1);
    model.add(std::move(dense1));

    model.add(std::make_unique<ReluLayer>());

    auto dense2 = std::make_unique<DenseLayer>(8, 3);
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.05f * static_cast<float>((i % 4) + 1);
    }
    dense2->set_weights(w2);
    Tensor b2({3});
    b2.zero();
    dense2->set_bias(b2);
    model.add(std::move(dense2));

    model.add(std::make_unique<SoftmaxLayer>());

    ModelSerializer::save(model, filename);
}

// Helper: build the same MLP in memory (for reference comparison)
static Sequential make_reference_mlp() {
    Sequential model;

    auto dense1 = std::make_unique<DenseLayer>(4, 8);
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.1f * static_cast<float>((i % 5) + 1);
    }
    dense1->set_weights(w1);
    Tensor b1({8});
    for (size_t i = 0; i < b1.size(); ++i) {
        b1.data()[i] = 0.01f * static_cast<float>(i);
    }
    dense1->set_bias(b1);
    model.add(std::move(dense1));

    model.add(std::make_unique<ReluLayer>());

    auto dense2 = std::make_unique<DenseLayer>(8, 3);
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.05f * static_cast<float>((i % 4) + 1);
    }
    dense2->set_weights(w2);
    Tensor b2({3});
    b2.zero();
    dense2->set_bias(b2);
    model.add(std::move(dense2));

    model.add(std::make_unique<SoftmaxLayer>());

    return model;
}

// Helper: create a test input tensor
static Tensor make_test_input() {
    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    return input;
}

// ============================================================
// Basic functionality tests
// ============================================================

TEST(InferenceEngineTest, LoadAndPredict) {
    TempFile tmp("test_ie_load.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Tensor input = make_test_input();
    Tensor output = engine.predict(input);

    EXPECT_EQ(output.shape(), (std::vector<size_t>{3}));
}

TEST(InferenceEngineTest, PredictBatch) {
    TempFile tmp("test_ie_batch.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    std::vector<Tensor> inputs;
    for (int i = 0; i < 3; ++i) {
        Tensor t({4});
        for (size_t j = 0; j < 4; ++j) {
            t.data()[j] = static_cast<float>(i + j);
        }
        inputs.push_back(t);
    }

    auto outputs = engine.predict_batch(inputs);

    EXPECT_EQ(outputs.size(), 3u);
    for (const auto& out : outputs) {
        EXPECT_EQ(out.shape(), (std::vector<size_t>{3}));
    }
}

TEST(InferenceEngineTest, PredictOutputCorrectness) {
    TempFile tmp("test_ie_correct.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Sequential ref = make_reference_mlp();
    Tensor input = make_test_input();

    Tensor engine_output = engine.predict(input);
    Tensor ref_output = ref.forward(input);

    ASSERT_EQ(engine_output.shape(), ref_output.shape());
    for (size_t i = 0; i < ref_output.size(); ++i) {
        EXPECT_FLOAT_EQ(engine_output.data()[i], ref_output.data()[i]);
    }
}

TEST(InferenceEngineTest, EmptyBatch) {
    TempFile tmp("test_ie_empty.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    std::vector<Tensor> empty_inputs;
    auto outputs = engine.predict_batch(empty_inputs);

    EXPECT_TRUE(outputs.empty());
}

TEST(InferenceEngineTest, MultiplePredictConsistent) {
    TempFile tmp("test_ie_consistent.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Tensor input = make_test_input();
    Tensor first_output = engine.predict(input);

    for (int run = 0; run < 5; ++run) {
        Tensor output = engine.predict(input);
        ASSERT_EQ(output.shape(), first_output.shape());
        for (size_t i = 0; i < first_output.size(); ++i) {
            EXPECT_FLOAT_EQ(output.data()[i], first_output.data()[i])
                << "Mismatch on run " << run << " at index " << i;
        }
    }
}

// ============================================================
// Memory pool tests
// ============================================================

TEST(InferenceEngineTest, BufferReuseCorrectness) {
    TempFile tmp("test_ie_reuse.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    for (int run = 0; run < 100; ++run) {
        Tensor input({4});
        for (size_t j = 0; j < 4; ++j) {
            input.data()[j] = 0.5f * static_cast<float>(run + j + 1);
        }

        Tensor output = engine.predict(input);
        ASSERT_EQ(output.shape(), (std::vector<size_t>{3}));

        // Softmax output must sum to ~1.0
        float sum = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_GE(output.data()[i], 0.0f);
            EXPECT_LE(output.data()[i], 1.0f);
            sum += output.data()[i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(InferenceEngineTest, BufferPreallocation) {
    TempFile tmp("test_ie_prealloc.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    EXPECT_EQ(engine.layer_count(), 4u);
    EXPECT_TRUE(engine.is_loaded());
    EXPECT_EQ(engine.expected_input_shape(),
              (std::vector<size_t>{4}));
}

// ============================================================
// Input validation tests
// ============================================================

TEST(InferenceEngineTest, MismatchedInputShape) {
    TempFile tmp("test_ie_badshape.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Tensor bad_input({5}); // model expects {4}
    EXPECT_THROW(engine.predict(bad_input), std::invalid_argument);
}

TEST(InferenceEngineTest, NaNInputDetection) {
    TempFile tmp("test_ie_nan.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = std::numeric_limits<float>::quiet_NaN();
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;

    EXPECT_THROW(engine.predict(input), std::invalid_argument);
}

TEST(InferenceEngineTest, WrongDimensionality) {
    TempFile tmp("test_ie_wrongdim.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    Tensor input_2d({2, 4}); // engine expects 1D {4}
    EXPECT_THROW(engine.predict(input_2d), std::invalid_argument);
}

// ============================================================
// Profiling tests
// ============================================================

TEST(InferenceEngineTest, ProfilingDisabledByDefault) {
    TempFile tmp("test_ie_noprof.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    auto s = engine.stats();
    EXPECT_EQ(s.inference_count, 0u);
    EXPECT_DOUBLE_EQ(s.total_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.min_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.max_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.mean_latency_ms, 0.0);

    // Run predict — stats should remain zero (profiling off)
    Tensor input = make_test_input();
    engine.predict(input);

    s = engine.stats();
    EXPECT_EQ(s.inference_count, 0u);
}

TEST(InferenceEngineTest, ProfilingTracksLatency) {
    TempFile tmp("test_ie_prof.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .build();

    Tensor input = make_test_input();
    for (int i = 0; i < 10; ++i) {
        engine.predict(input);
    }

    auto s = engine.stats();
    EXPECT_EQ(s.inference_count, 10u);
    EXPECT_GT(s.total_time_ms, 0.0);
    EXPECT_GT(s.min_latency_ms, 0.0);
    EXPECT_LE(s.min_latency_ms, s.mean_latency_ms);
    EXPECT_LE(s.mean_latency_ms, s.max_latency_ms);
}

TEST(InferenceEngineTest, ProfilingPerLayerTiming) {
    TempFile tmp("test_ie_perlayer.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .build();

    Tensor input = make_test_input();
    engine.predict(input);

    auto s = engine.stats();
    ASSERT_EQ(s.layer_times_ms.size(), 4u);
    for (size_t i = 0; i < s.layer_times_ms.size(); ++i) {
        EXPECT_GE(s.layer_times_ms[i], 0.0)
            << "Layer " << i << " has negative time";
    }
}

// ============================================================
// Builder pattern tests
// ============================================================

TEST(InferenceEngineTest, BuilderBasic) {
    TempFile tmp("test_ie_builder.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    EXPECT_TRUE(engine.is_loaded());
    EXPECT_EQ(engine.layer_count(), 4u);
}

TEST(InferenceEngineTest, BuilderWithProfiling) {
    TempFile tmp("test_ie_bprof.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .build();

    Tensor input = make_test_input();
    engine.predict(input);

    auto s = engine.stats();
    EXPECT_EQ(s.inference_count, 1u);
    EXPECT_GT(s.total_time_ms, 0.0);
}

TEST(InferenceEngineTest, BuilderWithWarmup) {
    TempFile tmp("test_ie_warmup.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .setWarmupRuns(3)
        .build();

    // Warmup should reset stats
    auto s = engine.stats();
    EXPECT_EQ(s.inference_count, 0u);
    EXPECT_DOUBLE_EQ(s.total_time_ms, 0.0);

    // Predictions after warmup should still work correctly
    Tensor input = make_test_input();
    Tensor output = engine.predict(input);
    EXPECT_EQ(output.shape(), (std::vector<size_t>{3}));

    s = engine.stats();
    EXPECT_EQ(s.inference_count, 1u);
}

TEST(InferenceEngineTest, BuilderNoModelPath) {
    EXPECT_THROW(
        InferenceEngine::Builder().build(),
        std::invalid_argument);
}

TEST(InferenceEngineTest, BuilderInvalidPath) {
    EXPECT_THROW(
        InferenceEngine::Builder()
            .setModelPath("nonexistent_model.titan")
            .build(),
        std::runtime_error);
}

// ============================================================
// Edge case tests
// ============================================================

TEST(InferenceEngineTest, ResetStats) {
    TempFile tmp("test_ie_reset.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .build();

    Tensor input = make_test_input();
    engine.predict(input);

    auto s = engine.stats();
    EXPECT_EQ(s.inference_count, 1u);

    engine.reset_stats();

    s = engine.stats();
    EXPECT_EQ(s.inference_count, 0u);
    EXPECT_DOUBLE_EQ(s.total_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.min_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.max_latency_ms, 0.0);
    EXPECT_DOUBLE_EQ(s.mean_latency_ms, 0.0);
    for (double t : s.layer_times_ms) {
        EXPECT_DOUBLE_EQ(t, 0.0);
    }
}

TEST(InferenceEngineTest, SummaryDelegation) {
    TempFile tmp("test_ie_summary.titan");
    save_test_mlp(tmp.path);

    auto engine = InferenceEngine::Builder()
        .setModelPath(tmp.path)
        .build();

    std::string s = engine.summary();
    EXPECT_NE(s.find("Dense(4, 8)"), std::string::npos);
    EXPECT_NE(s.find("ReLU"), std::string::npos);
    EXPECT_NE(s.find("Dense(8, 3)"), std::string::npos);
    EXPECT_NE(s.find("Softmax"), std::string::npos);
}
