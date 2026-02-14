/**
 * @file batch_processing.cpp
 * @brief Multi-threaded concurrent inference demonstration
 *
 * Loads a model once and runs inference from multiple threads simultaneously.
 * ModelHandle is thread-safe — no external synchronization needed.
 *
 * Usage:
 *   ./batch_processing model.titan
 */

#include "titaninfer/TitanInfer.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.titan>\n";
        return 1;
    }

    try {
        titaninfer::Logger::instance().set_level(titaninfer::LogLevel::WARNING);

        auto model = titaninfer::ModelHandle::Builder()
            .setModelPath(argv[1])
            .enableProfiling()
            .build();

        std::cout << "Model loaded: " << model.layer_count() << " layers\n";

        // Prepare input batch
        auto shape = model.expected_input_shape();
        const size_t total_samples = 100;
        const size_t num_threads = 4;
        const size_t samples_per_thread = total_samples / num_threads;

        std::vector<titaninfer::Tensor> inputs;
        inputs.reserve(total_samples);
        for (size_t i = 0; i < total_samples; ++i) {
            titaninfer::Tensor t(shape);
            t.fill(static_cast<float>(i) * 0.01f);
            inputs.push_back(std::move(t));
        }

        // Run concurrent inference
        auto start = std::chrono::steady_clock::now();

        std::atomic<size_t> completed{0};
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t begin = t * samples_per_thread;
            size_t end = (t + 1) * samples_per_thread;

            threads.emplace_back([&, begin, end]() {
                for (size_t i = begin; i < end; ++i) {
                    model.predict(inputs[i]);
                    completed++;
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();

        std::cout << "Processed " << completed.load() << " samples in "
                  << elapsed_ms << " ms\n";
        std::cout << "Throughput: "
                  << (static_cast<double>(completed.load()) / elapsed_ms * 1000.0)
                  << " samples/sec\n";

        auto stats = model.stats();
        std::cout << "Mean latency: " << stats.mean_latency_ms << " ms\n";
        std::cout << "Min latency:  " << stats.min_latency_ms << " ms\n";
        std::cout << "Max latency:  " << stats.max_latency_ms << " ms\n";

    } catch (const titaninfer::TitanInferException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
