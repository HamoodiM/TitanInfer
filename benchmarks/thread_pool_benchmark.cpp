#include <benchmark/benchmark.h>
#include "titaninfer/engine/thread_pool.hpp"

#include <atomic>
#include <vector>

using namespace titaninfer::engine;

static void BM_ThreadPool_Submit(benchmark::State& state) {
    const size_t num_threads = static_cast<size_t>(state.range(0));
    ThreadPool pool(num_threads);
    std::atomic<int> counter{0};

    for (auto _ : state) {
        auto future = pool.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
        future.get();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ThreadPool_Submit)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

static void BM_ThreadPool_Throughput(benchmark::State& state) {
    const size_t num_tasks = static_cast<size_t>(state.range(0));
    ThreadPool pool(4);

    for (auto _ : state) {
        std::atomic<int> counter{0};
        std::vector<std::future<void>> futures;
        futures.reserve(num_tasks);

        for (size_t i = 0; i < num_tasks; ++i) {
            futures.push_back(pool.submit([&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
            }));
        }

        for (auto& f : futures) {
            f.get();
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_tasks));
}
BENCHMARK(BM_ThreadPool_Throughput)->Arg(1000)->Arg(10000);

static void BM_ThreadPool_WorkloadParallel(benchmark::State& state) {
    const size_t num_threads = static_cast<size_t>(state.range(0));
    ThreadPool pool(num_threads);

    // Each task does some actual work (matrix-like compute)
    auto work = []() {
        volatile float sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum += static_cast<float>(i) * 0.001f;
        }
    };

    for (auto _ : state) {
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < 100; ++i) {
            futures.push_back(pool.submit(work));
        }
        for (auto& f : futures) {
            f.get();
        }
    }
    state.SetItemsProcessed(state.iterations() * 100);
}
BENCHMARK(BM_ThreadPool_WorkloadParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

BENCHMARK_MAIN();
