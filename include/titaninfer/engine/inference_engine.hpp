#pragma once

#include "titaninfer/tensor.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/io/model_parser.hpp"
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <cstddef>

namespace titaninfer {
namespace engine {

/**
 * @brief Profiling statistics for inference calls
 */
struct InferenceStats {
    size_t inference_count = 0;
    double total_time_ms = 0.0;
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double mean_latency_ms = 0.0;
    std::vector<double> layer_times_ms; // per-layer cumulative time (ms)
};

/**
 * @brief High-level inference engine with pre-allocated buffers and profiling
 *
 * Wraps Sequential model loaded from .titan files. Pre-allocates all
 * intermediate buffers at load time to eliminate per-inference heap
 * allocations. Provides input validation, warm-up, and optional profiling.
 *
 * Not thread-safe — internal buffers are shared mutable state.
 *
 * Usage:
 *   auto engine = InferenceEngine::Builder()
 *       .setModelPath("model.titan")
 *       .enableProfiling()
 *       .setWarmupRuns(3)
 *       .build();
 *   Tensor result = engine.predict(input);
 */
class InferenceEngine {
public:
    class Builder;

    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * @brief Run inference on a single input
     * @param input Input tensor matching expected_input_shape
     * @return Copy of the output tensor
     * @throws std::runtime_error if no model is loaded
     * @throws std::invalid_argument if input shape mismatches or contains NaN
     */
    Tensor predict(const Tensor& input);

    /**
     * @brief Run inference on a batch of individual inputs
     * @param inputs Vector of input tensors
     * @return Vector of output tensors (one per input)
     * @throws std::invalid_argument if any input is invalid
     */
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);

    /**
     * @brief Get profiling statistics (zeroed if profiling disabled)
     */
    InferenceStats stats() const;

    /**
     * @brief Reset profiling counters to zero
     */
    void reset_stats();

    /**
     * @brief Check if a model is loaded
     */
    bool is_loaded() const noexcept;

    /**
     * @brief Get the expected input shape
     * @throws std::runtime_error if no model is loaded
     */
    const std::vector<size_t>& expected_input_shape() const;

    /**
     * @brief Get the model summary string
     * @throws std::runtime_error if no model is loaded
     */
    std::string summary() const;

    /**
     * @brief Number of layers in the loaded model
     */
    size_t layer_count() const;

private:
    InferenceEngine();

    void load_model(const std::string& filepath,
                    const std::vector<size_t>& input_shape);
    void allocate_buffers();
    void warmup(size_t num_runs);
    void validate_input(const Tensor& input) const;

    std::unique_ptr<layers::Sequential> model_;
    std::vector<size_t> input_shape_;
    std::vector<Tensor> buffers_;
    bool profiling_enabled_;
    InferenceStats stats_;
};

/**
 * @brief Builder for InferenceEngine configuration
 */
class InferenceEngine::Builder {
public:
    Builder();

    /** @brief Set path to .titan model file (required) */
    Builder& setModelPath(const std::string& path);

    /** @brief Enable profiling (latency + per-layer timing) */
    Builder& enableProfiling(bool enable = true);

    /** @brief Set number of warm-up inference runs (default: 0) */
    Builder& setWarmupRuns(size_t count);

    /** @brief Override expected input shape (inferred from first DenseLayer if not set) */
    Builder& setInputShape(const std::vector<size_t>& shape);

    /**
     * @brief Construct the InferenceEngine
     * @return Configured InferenceEngine
     * @throws std::invalid_argument if model_path is not set
     * @throws std::runtime_error if model file cannot be loaded
     */
    InferenceEngine build();

private:
    std::string model_path_;
    bool profiling_enabled_;
    size_t warmup_runs_;
    std::vector<size_t> input_shape_;
};

} // namespace engine
} // namespace titaninfer
