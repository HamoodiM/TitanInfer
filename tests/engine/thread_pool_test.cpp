#include <gtest/gtest.h>
#include "titaninfer/engine/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

using namespace titaninfer::engine;

TEST(ThreadPoolTest, ConstructionDefault) {
    ThreadPool pool;
    EXPECT_GE(pool.thread_count(), 1u);
}

TEST(ThreadPoolTest, ConstructionExplicit) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.thread_count(), 4u);
}

TEST(ThreadPoolTest, SingleTask) {
    ThreadPool pool(2);
    auto future = pool.submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPoolTest, MultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST(ThreadPoolTest, ConcurrentAccess) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < 1000; ++i) {
        futures.push_back(pool.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter.load(), 1000);
}

TEST(ThreadPoolTest, VoidReturn) {
    ThreadPool pool(2);
    std::atomic<bool> executed{false};

    auto future = pool.submit([&executed]() {
        executed.store(true);
    });
    future.get();
    EXPECT_TRUE(executed.load());
}

TEST(ThreadPoolTest, ExceptionPropagation) {
    ThreadPool pool(2);
    auto future = pool.submit([]() -> int {
        throw std::runtime_error("test error");
    });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPoolTest, ZeroThreadsUsesHardwareConcurrency) {
    ThreadPool pool(0);
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    EXPECT_EQ(pool.thread_count(), static_cast<size_t>(hw));
}

TEST(ThreadPoolTest, TasksCompleteBeforeDestruction) {
    std::atomic<int> completed{0};

    {
        ThreadPool pool(2);
        for (int i = 0; i < 50; ++i) {
            pool.submit([&completed]() {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                completed.fetch_add(1);
            });
        }
    } // Destructor waits for all tasks

    EXPECT_EQ(completed.load(), 50);
}

TEST(ThreadPoolTest, StressTest) {
    ThreadPool pool(8);
    std::atomic<int64_t> sum{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < 10000; ++i) {
        futures.push_back(pool.submit([&sum, i]() {
            sum.fetch_add(i, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    int64_t expected = static_cast<int64_t>(9999) * 10000 / 2;
    EXPECT_EQ(sum.load(), expected);
}

TEST(ThreadPoolTest, SubmitFromMultipleThreads) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    auto submitter = [&pool, &counter]() {
        std::vector<std::future<void>> futures;
        for (int i = 0; i < 100; ++i) {
            futures.push_back(pool.submit([&counter]() {
                counter.fetch_add(1);
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back(submitter);
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), 400);
}
