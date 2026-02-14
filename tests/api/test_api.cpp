#include <gtest/gtest.h>
#include "titaninfer/TitanInfer.hpp"
#include "titaninfer/titaninfer_c.h"
#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include <memory>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <thread>
#include <atomic>
#include <vector>
#include <limits>
#include <cstring>
#include <fstream>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::io;

// ============================================================
// RAII helper for temporary test files
// ============================================================

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

static Tensor make_test_input() {
    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    return input;
}

// ============================================================
// Exception Tests
// ============================================================

TEST(ExceptionTest, BaseExceptionIsRuntimeError) {
    TitanInferException e("test error", ErrorCode::UNKNOWN);
    const std::runtime_error& re = e;  // must compile
    EXPECT_STREQ(re.what(), "test error");
}

TEST(ExceptionTest, ModelLoadExceptionCode) {
    ModelLoadException e("file not found", ErrorCode::FILE_NOT_FOUND);
    EXPECT_EQ(e.error_code(), ErrorCode::FILE_NOT_FOUND);
    EXPECT_STREQ(e.what(), "file not found");
}

TEST(ExceptionTest, InferenceExceptionInheritance) {
    InferenceException e("no model", ErrorCode::NO_MODEL_LOADED);
    const TitanInferException& base = e;  // must compile
    EXPECT_EQ(base.error_code(), ErrorCode::NO_MODEL_LOADED);
}

TEST(ExceptionTest, ValidationExceptionIsDistinct) {
    ValidationException ve("shape mismatch", ErrorCode::SHAPE_MISMATCH);
    bool caught_validation = false;
    try {
        throw ve;
    } catch (const ValidationException&) {
        caught_validation = true;
    } catch (const ModelLoadException&) {
        FAIL() << "Should not catch as ModelLoadException";
    }
    EXPECT_TRUE(caught_validation);
}

TEST(ExceptionTest, WhatMessagePreserved) {
    TitanInferException e("detailed error message 123", ErrorCode::INTERNAL_ERROR);
    EXPECT_STREQ(e.what(), "detailed error message 123");
    EXPECT_EQ(e.error_code(), ErrorCode::INTERNAL_ERROR);
}

// ============================================================
// Logger Tests
// ============================================================

class LoggerTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().set_level(LogLevel::INFO);
        Logger::instance().set_stream(std::cerr);
    }
    void TearDown() override {
        Logger::instance().set_level(LogLevel::INFO);
        Logger::instance().set_stream(std::cerr);
    }
};

TEST_F(LoggerTestFixture, SingletonIdentity) {
    Logger& a = Logger::instance();
    Logger& b = Logger::instance();
    EXPECT_EQ(&a, &b);
}

TEST_F(LoggerTestFixture, DefaultLevelIsInfo) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().debug("should not appear");
    EXPECT_TRUE(oss.str().empty());

    Logger::instance().info("should appear");
    EXPECT_FALSE(oss.str().empty());
    EXPECT_NE(oss.str().find("should appear"), std::string::npos);
}

TEST_F(LoggerTestFixture, SetLevelFiltersMessages) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().set_level(LogLevel::WARNING);

    Logger::instance().debug("d");
    Logger::instance().info("i");
    EXPECT_TRUE(oss.str().empty());

    Logger::instance().warning("w");
    EXPECT_NE(oss.str().find("[WARNING]"), std::string::npos);
}

TEST_F(LoggerTestFixture, StreamRedirect) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().info("redirected message");
    EXPECT_NE(oss.str().find("redirected message"), std::string::npos);
    EXPECT_NE(oss.str().find("[INFO]"), std::string::npos);
}

TEST_F(LoggerTestFixture, ThreadSafeMultipleWriters) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().set_level(LogLevel::DEBUG);

    const int num_threads = 10;
    const int msgs_per_thread = 20;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int m = 0; m < msgs_per_thread; ++m) {
                Logger::instance().info("thread" + std::to_string(t) +
                                        "_msg" + std::to_string(m));
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    // Count lines — each log call produces exactly one line
    std::string output = oss.str();
    size_t line_count = 0;
    for (char c : output) {
        if (c == '\n') ++line_count;
    }
    EXPECT_EQ(line_count, static_cast<size_t>(num_threads * msgs_per_thread));
}

TEST_F(LoggerTestFixture, SilentSuppressesAll) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().set_level(LogLevel::SILENT);

    Logger::instance().debug("d");
    Logger::instance().info("i");
    Logger::instance().warning("w");
    Logger::instance().error("e");
    EXPECT_TRUE(oss.str().empty());
}

TEST_F(LoggerTestFixture, MacroSkipsConstructionWhenSilent) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().set_level(LogLevel::SILENT);

    std::atomic<int> counter{0};
    auto make_msg = [&counter]() -> std::string {
        counter++;
        return "side-effect";
    };

    TITANINFER_LOG_INFO(make_msg());
    EXPECT_EQ(counter.load(), 0);
}

TEST_F(LoggerTestFixture, RestoreStreamAfterTest) {
    // This test simply verifies the fixture teardown works
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().info("test");
    // TearDown will restore std::cerr
}

// ============================================================
// ModelHandle Tests
// ============================================================

TEST(ModelHandleTest, LoadValidModel) {
    TempFile tmp("api_test_load.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    EXPECT_TRUE(handle.is_loaded());
    EXPECT_EQ(handle.layer_count(), 4u);
}

TEST(ModelHandleTest, PredictProducesOutput) {
    TempFile tmp("api_test_predict.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor input = make_test_input();
    Tensor output = handle.predict(input);

    EXPECT_EQ(output.ndim(), 1u);
    EXPECT_EQ(output.shape()[0], 3u);

    // Softmax output: all values in [0,1], sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_GE(output.data()[i], 0.0f);
        EXPECT_LE(output.data()[i], 1.0f);
        sum += output.data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(ModelHandleTest, PredictBatch) {
    TempFile tmp("api_test_batch.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    std::vector<Tensor> inputs;
    for (int i = 0; i < 5; ++i) {
        inputs.push_back(make_test_input());
    }

    auto outputs = handle.predict_batch(inputs);
    EXPECT_EQ(outputs.size(), 5u);

    for (const auto& out : outputs) {
        EXPECT_EQ(out.shape()[0], 3u);
    }
}

TEST(ModelHandleTest, InvalidPathThrowsModelLoadException) {
    EXPECT_THROW({
        ModelHandle::Builder()
            .setModelPath("nonexistent_file_12345.titan")
            .setLogLevel(LogLevel::SILENT)
            .build();
    }, ModelLoadException);
}

TEST(ModelHandleTest, ShapeMismatchThrowsValidationException) {
    TempFile tmp("api_test_shape.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor wrong_input({7});  // expected {4}
    EXPECT_THROW(handle.predict(wrong_input), ValidationException);
}

TEST(ModelHandleTest, NaNInputThrowsValidationException) {
    TempFile tmp("api_test_nan.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = std::numeric_limits<float>::quiet_NaN();
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;

    try {
        handle.predict(input);
        FAIL() << "Expected ValidationException";
    } catch (const ValidationException& e) {
        EXPECT_EQ(e.error_code(), ErrorCode::NAN_INPUT);
    }
}

TEST(ModelHandleTest, StatsAndReset) {
    TempFile tmp("api_test_stats.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor input = make_test_input();
    handle.predict(input);
    handle.predict(input);

    auto s = handle.stats();
    EXPECT_EQ(s.inference_count, 2u);
    EXPECT_GT(s.mean_latency_ms, 0.0);

    handle.reset_stats();
    auto s2 = handle.stats();
    EXPECT_EQ(s2.inference_count, 0u);
}

TEST(ModelHandleTest, MoveSemantics) {
    TempFile tmp("api_test_move.titan");
    save_test_mlp(tmp.path);

    auto handle1 = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    ModelHandle handle2(std::move(handle1));
    EXPECT_TRUE(handle2.is_loaded());

    // Moved-from handle should not be loaded
    EXPECT_FALSE(handle1.is_loaded());  // NOLINT(bugprone-use-after-move)
}

TEST(ModelHandleTest, SummaryDelegation) {
    TempFile tmp("api_test_summary.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    std::string s = handle.summary();
    EXPECT_NE(s.find("Dense"), std::string::npos);
    EXPECT_NE(s.find("ReLU"), std::string::npos);
}

TEST(ModelHandleTest, NoModelPathThrowsModelLoadException) {
    EXPECT_THROW({
        ModelHandle::Builder()
            .setLogLevel(LogLevel::SILENT)
            .build();
    }, ModelLoadException);
}

// ============================================================
// Thread Safety Tests
// ============================================================

TEST(ThreadSafetyTest, ConcurrentPredict) {
    TempFile tmp("api_test_concurrent.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    // Get reference result
    Tensor input = make_test_input();
    Tensor reference = handle.predict(input);

    const int num_threads = 8;
    const int iters = 20;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < iters; ++i) {
                Tensor result = handle.predict(input);
                for (size_t j = 0; j < result.size(); ++j) {
                    if (std::abs(result.data()[j] - reference.data()[j]) > 1e-5f) {
                        failures++;
                        return;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

TEST(ThreadSafetyTest, ConcurrentPredictBatch) {
    TempFile tmp("api_test_concurrent_batch.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();

    std::vector<Tensor> batch;
    for (int i = 0; i < 3; ++i) {
        batch.push_back(make_test_input());
    }

    const int num_threads = 4;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            auto results = handle.predict_batch(batch);
            if (results.size() != batch.size()) {
                failures++;
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

TEST(ThreadSafetyTest, ConcurrentStatsRead) {
    TempFile tmp("api_test_concurrent_stats.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor input = make_test_input();
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};

    // Predictor threads
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            while (!stop.load()) {
                handle.predict(input);
            }
        });
    }

    // Stats reader threads
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                auto s = handle.stats();
                // inference_count should never be negative (unsigned, so just check it doesn't wrap)
                if (s.inference_count > 100000) {
                    failures++;
                }
            }
        });
    }

    // Let threads run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

TEST(ThreadSafetyTest, ConcurrentResetAndPredict) {
    TempFile tmp("api_test_concurrent_reset.titan");
    save_test_mlp(tmp.path);

    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .enableProfiling()
        .setLogLevel(LogLevel::SILENT)
        .build();

    Tensor input = make_test_input();
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};

    std::vector<std::thread> threads;

    // Predict threads
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            while (!stop.load()) {
                try {
                    handle.predict(input);
                } catch (...) {
                    failures++;
                }
            }
        });
    }

    // Reset thread
    threads.emplace_back([&]() {
        while (!stop.load()) {
            handle.reset_stats();
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

TEST(ThreadSafetyTest, LoggerConcurrentWrite) {
    std::ostringstream oss;
    Logger::instance().set_stream(oss);
    Logger::instance().set_level(LogLevel::DEBUG);

    const int num_threads = 20;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            Logger::instance().info("thread_" + std::to_string(t));
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    std::string output = oss.str();
    size_t line_count = 0;
    for (char c : output) {
        if (c == '\n') ++line_count;
    }
    EXPECT_EQ(line_count, static_cast<size_t>(num_threads));

    // Restore
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().set_stream(std::cerr);
}

// ============================================================
// C API Tests
// ============================================================

TEST(CApiTest, LoadAndPredict) {
    TempFile tmp("api_test_c_load.titan");
    save_test_mlp(tmp.path);

    size_t shape[] = {4};
    TitanInferModelHandle h = titaninfer_load(tmp.path.c_str(), shape, 1);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(titaninfer_is_loaded(h), 1);

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[3] = {};
    size_t actual_len = 0;

    int status = titaninfer_predict(h, input, 4, output, 3, &actual_len);
    EXPECT_EQ(status, TITANINFER_OK);
    EXPECT_EQ(actual_len, 3u);

    // Softmax output sums to 1
    float sum = output[0] + output[1] + output[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    titaninfer_free(h);
}

TEST(CApiTest, OutputMatchesCppApi) {
    TempFile tmp("api_test_c_match.titan");
    save_test_mlp(tmp.path);

    // C++ reference
    auto handle = ModelHandle::Builder()
        .setModelPath(tmp.path)
        .setLogLevel(LogLevel::SILENT)
        .build();
    Tensor cpp_result = handle.predict(make_test_input());

    // C API
    size_t shape[] = {4};
    TitanInferModelHandle h = titaninfer_load(tmp.path.c_str(), shape, 1);
    ASSERT_NE(h, nullptr);

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[3] = {};
    size_t actual_len = 0;

    titaninfer_predict(h, input, 4, output, 3, &actual_len);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(output[i], cpp_result.data()[i]);
    }

    titaninfer_free(h);
}

TEST(CApiTest, InvalidPathReturnsNull) {
    TitanInferModelHandle h = titaninfer_load("nonexistent_99999.titan", nullptr, 0);
    EXPECT_EQ(h, nullptr);
}

TEST(CApiTest, ShapeMismatchReturnsValidationError) {
    TempFile tmp("api_test_c_shape.titan");
    save_test_mlp(tmp.path);

    size_t shape[] = {4};
    TitanInferModelHandle h = titaninfer_load(tmp.path.c_str(), shape, 1);
    ASSERT_NE(h, nullptr);

    float wrong_input[] = {1.0f, 2.0f, 3.0f};  // wrong length
    float output[3] = {};
    size_t actual_len = 0;

    int status = titaninfer_predict(h, wrong_input, 3, output, 3, &actual_len);
    EXPECT_EQ(status, TITANINFER_ERR_VALIDATION);
    EXPECT_NE(titaninfer_last_error(h), nullptr);

    titaninfer_free(h);
}

TEST(CApiTest, LastErrorMessage) {
    TempFile tmp("api_test_c_err.titan");
    save_test_mlp(tmp.path);

    size_t shape[] = {4};
    TitanInferModelHandle h = titaninfer_load(tmp.path.c_str(), shape, 1);
    ASSERT_NE(h, nullptr);

    // Before any error, last_error should be null
    EXPECT_EQ(titaninfer_last_error(h), nullptr);

    // Cause an error
    float wrong_input[] = {1.0f};
    float output[3] = {};
    size_t actual_len = 0;
    titaninfer_predict(h, wrong_input, 1, output, 3, &actual_len);

    const char* err = titaninfer_last_error(h);
    ASSERT_NE(err, nullptr);
    EXPECT_GT(std::strlen(err), 0u);

    titaninfer_free(h);
}

TEST(CApiTest, FreeNullHandleIsSafe) {
    titaninfer_free(nullptr);  // should not crash
}

// ============================================================
// Resource Cleanup Tests
// ============================================================

TEST(ResourceTest, DestructorReleasesResources) {
    TempFile tmp("api_test_resource.titan");
    save_test_mlp(tmp.path);

    // Create and destroy handles in a loop — should not leak
    for (int i = 0; i < 100; ++i) {
        auto handle = ModelHandle::Builder()
            .setModelPath(tmp.path)
            .setLogLevel(LogLevel::SILENT)
            .build();
        Tensor input = make_test_input();
        handle.predict(input);
    }
    // If we get here without crashing or running out of memory, pass
    SUCCEED();
}

TEST(ResourceTest, TempFileCleanup) {
    const std::string path = "api_test_tempfile_cleanup.titan";
    {
        TempFile tmp(path);
        save_test_mlp(tmp.path);
        // File should exist
        std::ifstream f(path);
        EXPECT_TRUE(f.good());
    }
    // File should be removed by TempFile destructor
    std::ifstream f(path);
    EXPECT_FALSE(f.good());
}
