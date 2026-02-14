#pragma once

#include "titaninfer/tensor.hpp"
#include "titaninfer/exceptions.hpp"
#include "titaninfer/logger.hpp"
#include "titaninfer/engine/inference_engine.hpp"
#include <mutex>
#include <string>
#include <vector>
#include <cstddef>

namespace titaninfer {

/// Re-export InferenceStats for public API consumers
using InferenceStats = engine::InferenceStats;

/**
 * @brief Thread-safe RAII wrapper around InferenceEngine
 *
 * This is the primary public API class. It owns an InferenceEngine,
 * protects all operations with a mutex for concurrent use, and translates
 * raw exceptions into the structured TitanInfer exception hierarchy.
 *
 * Non-copyable, movable. Thread-safe for concurrent predict() calls.
 *
 * Usage:
 * @code
 *   auto model = ModelHandle::Builder()
 *       .setModelPath("model.titan")
 *       .enableProfiling()
 *       .setWarmupRuns(3)
 *       .build();
 *   Tensor result = model.predict(input);
 * @endcode
 */
class ModelHandle {
public:
    class Builder;

    ModelHandle(ModelHandle&& other) noexcept;
    ModelHandle& operator=(ModelHandle&& other) noexcept;
    ~ModelHandle();

    ModelHandle(const ModelHandle&) = delete;
    ModelHandle& operator=(const ModelHandle&) = delete;

    /**
     * @brief Thread-safe single-input inference
     * @param input Tensor matching expected_input_shape()
     * @return Deep copy of the output tensor
     * @throws ValidationException on shape mismatch or NaN input
     * @throws InferenceException on internal engine error
     */
    Tensor predict(const Tensor& input);

    /**
     * @brief Thread-safe batch inference
     * @param inputs Vector of input tensors
     * @return Vector of output tensors (one per input)
     * @throws ValidationException if any input is invalid
     */
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);

    /**
     * @brief Get a snapshot of profiling statistics
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
     * @brief Number of layers in the loaded model
     */
    size_t layer_count() const noexcept;

    /**
     * @brief Get a formatted model summary
     * @throws InferenceException if no model is loaded
     */
    std::string summary() const;

    /**
     * @brief Get the expected input shape
     * @throws InferenceException if no model is loaded
     */
    const std::vector<size_t>& expected_input_shape() const;

private:
    explicit ModelHandle(engine::InferenceEngine engine);

    mutable std::mutex      mutex_;
    engine::InferenceEngine engine_;
};

/**
 * @brief Fluent builder for ModelHandle configuration
 */
class ModelHandle::Builder {
public:
    Builder();

    /** @brief Set path to .titan model file (required) */
    Builder& setModelPath(const std::string& path);

    /** @brief Enable profiling (latency + per-layer timing) */
    Builder& enableProfiling(bool enable = true);

    /** @brief Set number of warm-up inference runs (default: 0) */
    Builder& setWarmupRuns(size_t count);

    /** @brief Override expected input shape (inferred from first DenseLayer if omitted) */
    Builder& setInputShape(const std::vector<size_t>& shape);

    /** @brief Set the log level before loading */
    Builder& setLogLevel(LogLevel level);

    /**
     * @brief Build the ModelHandle
     * @return Configured ModelHandle with model loaded
     * @throws ModelLoadException if model path is empty or file cannot be loaded
     */
    ModelHandle build();

private:
    std::string         model_path_;
    bool                profiling_enabled_;
    size_t              warmup_runs_;
    std::vector<size_t> input_shape_;
    LogLevel            log_level_;
};

} // namespace titaninfer
