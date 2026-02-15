#include <gtest/gtest.h>
#include "titaninfer/engine/dynamic_batcher.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::engine;

namespace {

std::unique_ptr<Sequential> make_simple_model() {
    auto model = std::make_unique<Sequential>();
    auto dense = std::make_unique<DenseLayer>(4, 2, true);

    Tensor w({2, 4});
    for (size_t i = 0; i < 8; ++i) w.data()[i] = 0.1f * static_cast<float>(i);
    dense->set_weights(w);

    Tensor b({2});
    b.data()[0] = 0.1f; b.data()[1] = 0.2f;
    dense->set_bias(b);

    model->add(std::move(dense));
    model->add(std::make_unique<ReluLayer>());
    return model;
}

} // anonymous namespace

TEST(DynamicBatcherTest, SingleRequest) {
    auto model = make_simple_model();
    DynamicBatcher batcher(*model, {4}, {32, 100});

    Tensor input({4});
    input.fill(1.0f);

    auto future = batcher.submit(std::move(input));
    Tensor result = future.get();

    // Just verify we got an output of the right shape
    EXPECT_EQ(result.shape().size(), 1u);
    EXPECT_EQ(result.shape()[0], 2u);
}

TEST(DynamicBatcherTest, MultipleRequests) {
    auto model = make_simple_model();
    DynamicBatcher batcher(*model, {4}, {32, 100});

    std::vector<std::future<Tensor>> futures;
    for (int i = 0; i < 5; ++i) {
        Tensor input({4});
        input.fill(static_cast<float>(i));
        futures.push_back(batcher.submit(std::move(input)));
    }

    for (auto& f : futures) {
        Tensor result = f.get();
        EXPECT_EQ(result.shape()[0], 2u);
    }
}

TEST(DynamicBatcherTest, ConcurrentSubmission) {
    auto model = make_simple_model();
    DynamicBatcher batcher(*model, {4}, {16, 50});

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&batcher, &success_count]() {
            for (int i = 0; i < 10; ++i) {
                Tensor input({4});
                input.fill(1.0f);
                auto future = batcher.submit(std::move(input));
                Tensor result = future.get();
                if (result.shape()[0] == 2) {
                    success_count.fetch_add(1);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), 40);
}

TEST(DynamicBatcherTest, TimeoutFlush) {
    auto model = make_simple_model();
    // Low max_wait_ms so single requests get flushed quickly
    DynamicBatcher batcher(*model, {4}, {1000, 5});

    Tensor input({4});
    input.fill(1.0f);

    auto start = std::chrono::steady_clock::now();
    auto future = batcher.submit(std::move(input));
    Tensor result = future.get();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    EXPECT_EQ(result.shape()[0], 2u);
    // Should complete within reasonable time (timeout + processing)
    EXPECT_LT(elapsed, 1000);
}

TEST(DynamicBatcherTest, DestructorDrains) {
    auto model = make_simple_model();
    std::vector<std::future<Tensor>> futures;

    {
        DynamicBatcher batcher(*model, {4}, {32, 100});

        for (int i = 0; i < 10; ++i) {
            Tensor input({4});
            input.fill(1.0f);
            futures.push_back(batcher.submit(std::move(input)));
        }
    } // Destructor should process remaining

    // All futures should be ready
    for (auto& f : futures) {
        // Should not hang — batcher processed before destruction
        auto status = f.wait_for(std::chrono::seconds(5));
        EXPECT_EQ(status, std::future_status::ready);
    }
}

TEST(DynamicBatcherTest, OutputMatchesDirect) {
    auto model = make_simple_model();

    Tensor input({4});
    for (size_t i = 0; i < 4; ++i) input.data()[i] = static_cast<float>(i);

    // Direct forward
    Tensor direct = model->forward(input);

    // Via batcher
    DynamicBatcher batcher(*model, {4}, {1, 5});
    Tensor input_copy({4});
    for (size_t i = 0; i < 4; ++i) input_copy.data()[i] = static_cast<float>(i);

    auto future = batcher.submit(std::move(input_copy));
    Tensor batched = future.get();

    ASSERT_EQ(direct.shape(), batched.shape());
    for (size_t i = 0; i < direct.size(); ++i) {
        EXPECT_NEAR(batched.data()[i], direct.data()[i], 1e-5f);
    }
}
