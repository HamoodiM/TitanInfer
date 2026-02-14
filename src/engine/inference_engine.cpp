#include "titaninfer/engine/inference_engine.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace titaninfer {
namespace engine {

// ============================================================
// Builder
// ============================================================

InferenceEngine::Builder::Builder()
    : profiling_enabled_(false)
    , warmup_runs_(0)
{}

InferenceEngine::Builder&
InferenceEngine::Builder::setModelPath(const std::string& path) {
    model_path_ = path;
    return *this;
}

InferenceEngine::Builder&
InferenceEngine::Builder::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
    return *this;
}

InferenceEngine::Builder&
InferenceEngine::Builder::setWarmupRuns(size_t count) {
    warmup_runs_ = count;
    return *this;
}

InferenceEngine::Builder&
InferenceEngine::Builder::setInputShape(const std::vector<size_t>& shape) {
    input_shape_ = shape;
    return *this;
}

InferenceEngine InferenceEngine::Builder::build() {
    if (model_path_.empty()) {
        throw std::invalid_argument(
            "InferenceEngine::Builder::build: model path not set");
    }

    InferenceEngine engine;
    engine.profiling_enabled_ = profiling_enabled_;
    engine.load_model(model_path_, input_shape_);

    if (warmup_runs_ > 0) {
        engine.warmup(warmup_runs_);
    }

    return engine;
}

// ============================================================
// InferenceEngine — Construction / Move / Destruction
// ============================================================

InferenceEngine::InferenceEngine()
    : profiling_enabled_(false)
{}

InferenceEngine::InferenceEngine(InferenceEngine&& other) noexcept
    : model_(std::move(other.model_))
    , input_shape_(std::move(other.input_shape_))
    , buffers_(std::move(other.buffers_))
    , profiling_enabled_(other.profiling_enabled_)
    , stats_(other.stats_)
{}

InferenceEngine&
InferenceEngine::operator=(InferenceEngine&& other) noexcept {
    if (this != &other) {
        model_ = std::move(other.model_);
        input_shape_ = std::move(other.input_shape_);
        buffers_ = std::move(other.buffers_);
        profiling_enabled_ = other.profiling_enabled_;
        stats_ = other.stats_;
    }
    return *this;
}

InferenceEngine::~InferenceEngine() = default;

// ============================================================
// Model Loading & Buffer Pre-allocation
// ============================================================

void InferenceEngine::load_model(
        const std::string& filepath,
        const std::vector<size_t>& input_shape) {

    model_ = io::ModelParser::load(filepath);

    if (model_->empty()) {
        throw std::runtime_error(
            "InferenceEngine: loaded model has no layers");
    }

    // Determine input shape
    if (!input_shape.empty()) {
        input_shape_ = input_shape;
    } else {
        // Infer from first DenseLayer's in_features
        bool found = false;
        for (size_t i = 0; i < model_->size(); ++i) {
            const auto* dense = dynamic_cast<const layers::DenseLayer*>(
                &model_->layer(i));
            if (dense) {
                input_shape_ = {dense->in_features()};
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "InferenceEngine: cannot infer input shape -- "
                "no DenseLayer found and no input shape provided");
        }
    }

    allocate_buffers();
}

void InferenceEngine::allocate_buffers() {
    buffers_.clear();
    buffers_.reserve(model_->size());

    std::vector<size_t> current_shape = input_shape_;

    for (size_t i = 0; i < model_->size(); ++i) {
        current_shape = model_->layer(i).output_shape(current_shape);
        buffers_.emplace_back(current_shape);
    }

    stats_.layer_times_ms.resize(model_->size(), 0.0);
}

// ============================================================
// Warm-up
// ============================================================

void InferenceEngine::warmup(size_t num_runs) {
    Tensor dummy(input_shape_);

    for (size_t r = 0; r < num_runs; ++r) {
        model_->layer(0).forward(dummy, buffers_[0]);
        for (size_t i = 1; i < model_->size(); ++i) {
            model_->layer(i).forward(buffers_[i - 1], buffers_[i]);
        }
    }

    reset_stats();
}

// ============================================================
// Input Validation
// ============================================================

void InferenceEngine::validate_input(const Tensor& input) const {
    // Check dimensionality
    if (input.ndim() != input_shape_.size()) {
        throw std::invalid_argument(
            "InferenceEngine: expected " +
            std::to_string(input_shape_.size()) + "D input, got " +
            std::to_string(input.ndim()) + "D");
    }

    // Check shape
    if (input.shape() != input_shape_) {
        std::string expected = "(";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (i > 0) expected += ", ";
            expected += std::to_string(input_shape_[i]);
        }
        expected += ")";

        std::string actual = "(";
        for (size_t i = 0; i < input.shape().size(); ++i) {
            if (i > 0) actual += ", ";
            actual += std::to_string(input.shape()[i]);
        }
        actual += ")";

        throw std::invalid_argument(
            "InferenceEngine: expected input shape " + expected +
            ", got " + actual);
    }

    // Check for NaN values
    const float* data = input.data();
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::isnan(data[i])) {
            throw std::invalid_argument(
                "InferenceEngine: input contains NaN at index " +
                std::to_string(i));
        }
    }
}

// ============================================================
// Predict
// ============================================================

Tensor InferenceEngine::predict(const Tensor& input) {
    if (!model_) {
        throw std::runtime_error(
            "InferenceEngine::predict: no model loaded");
    }

    validate_input(input);

    using clock = std::chrono::steady_clock;
    clock::time_point total_start;

    if (profiling_enabled_) {
        total_start = clock::now();
    }

    // Layer 0: input -> buffers_[0]
    if (profiling_enabled_) {
        auto start = clock::now();
        model_->layer(0).forward(input, buffers_[0]);
        auto end = clock::now();
        stats_.layer_times_ms[0] +=
            std::chrono::duration<double, std::milli>(end - start).count();
    } else {
        model_->layer(0).forward(input, buffers_[0]);
    }

    // Layers 1..N-1: buffers_[i-1] -> buffers_[i]
    for (size_t i = 1; i < model_->size(); ++i) {
        if (profiling_enabled_) {
            auto start = clock::now();
            model_->layer(i).forward(buffers_[i - 1], buffers_[i]);
            auto end = clock::now();
            stats_.layer_times_ms[i] +=
                std::chrono::duration<double, std::milli>(
                    end - start).count();
        } else {
            model_->layer(i).forward(buffers_[i - 1], buffers_[i]);
        }
    }

    if (profiling_enabled_) {
        auto total_end = clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                total_end - total_start).count();

        stats_.inference_count++;
        stats_.total_time_ms += elapsed_ms;
        stats_.mean_latency_ms =
            stats_.total_time_ms /
            static_cast<double>(stats_.inference_count);

        if (stats_.inference_count == 1) {
            stats_.min_latency_ms = elapsed_ms;
            stats_.max_latency_ms = elapsed_ms;
        } else {
            stats_.min_latency_ms =
                std::min(stats_.min_latency_ms, elapsed_ms);
            stats_.max_latency_ms =
                std::max(stats_.max_latency_ms, elapsed_ms);
        }
    }

    // Return a deep copy — internal buffer is reused across calls
    return Tensor(buffers_.back());
}

std::vector<Tensor> InferenceEngine::predict_batch(
        const std::vector<Tensor>& inputs) {
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        outputs.push_back(predict(input));
    }

    return outputs;
}

// ============================================================
// Utility Methods
// ============================================================

InferenceStats InferenceEngine::stats() const {
    return stats_;
}

void InferenceEngine::reset_stats() {
    stats_.inference_count = 0;
    stats_.total_time_ms = 0.0;
    stats_.min_latency_ms = 0.0;
    stats_.max_latency_ms = 0.0;
    stats_.mean_latency_ms = 0.0;
    std::fill(stats_.layer_times_ms.begin(),
              stats_.layer_times_ms.end(), 0.0);
}

bool InferenceEngine::is_loaded() const noexcept {
    return model_ != nullptr;
}

const std::vector<size_t>&
InferenceEngine::expected_input_shape() const {
    if (!model_) {
        throw std::runtime_error(
            "InferenceEngine::expected_input_shape: no model loaded");
    }
    return input_shape_;
}

std::string InferenceEngine::summary() const {
    if (!model_) {
        throw std::runtime_error(
            "InferenceEngine::summary: no model loaded");
    }
    return model_->summary(input_shape_);
}

size_t InferenceEngine::layer_count() const {
    return model_ ? model_->size() : 0;
}

} // namespace engine
} // namespace titaninfer
