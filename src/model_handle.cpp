#include "titaninfer/model_handle.hpp"
#include <algorithm>

namespace titaninfer {

// ============================================================
// Builder
// ============================================================

ModelHandle::Builder::Builder()
    : profiling_enabled_(false)
    , warmup_runs_(0)
    , log_level_(LogLevel::INFO)
{}

ModelHandle::Builder&
ModelHandle::Builder::setModelPath(const std::string& path) {
    model_path_ = path;
    return *this;
}

ModelHandle::Builder&
ModelHandle::Builder::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
    return *this;
}

ModelHandle::Builder&
ModelHandle::Builder::setWarmupRuns(size_t count) {
    warmup_runs_ = count;
    return *this;
}

ModelHandle::Builder&
ModelHandle::Builder::setInputShape(const std::vector<size_t>& shape) {
    input_shape_ = shape;
    return *this;
}

ModelHandle::Builder&
ModelHandle::Builder::setLogLevel(LogLevel level) {
    log_level_ = level;
    return *this;
}

ModelHandle ModelHandle::Builder::build() {
    Logger::instance().set_level(log_level_);

    if (model_path_.empty()) {
        throw ModelLoadException(
            "ModelHandle::Builder::build: model path not set",
            ErrorCode::FILE_NOT_FOUND);
    }

    TITANINFER_LOG_INFO("Loading model from: " + model_path_);

    try {
        auto inner_builder = engine::InferenceEngine::Builder()
            .setModelPath(model_path_)
            .enableProfiling(profiling_enabled_)
            .setWarmupRuns(warmup_runs_);

        if (!input_shape_.empty()) {
            inner_builder.setInputShape(input_shape_);
        }

        ModelHandle handle(inner_builder.build());
        TITANINFER_LOG_INFO("Model loaded successfully (" +
                            std::to_string(handle.layer_count()) + " layers)");
        return handle;
    } catch (const std::invalid_argument& e) {
        throw ModelLoadException(e.what(), ErrorCode::INVALID_FORMAT);
    } catch (const std::runtime_error& e) {
        throw ModelLoadException(e.what(), ErrorCode::FILE_NOT_FOUND);
    }
}

// ============================================================
// ModelHandle — Construction / Move / Destruction
// ============================================================

ModelHandle::ModelHandle(engine::InferenceEngine engine)
    : engine_(std::move(engine))
{}

ModelHandle::ModelHandle(ModelHandle&& other) noexcept
    : engine_(std::move(other.engine_))
{}

ModelHandle& ModelHandle::operator=(ModelHandle&& other) noexcept {
    if (this != &other) {
        std::lock(mutex_, other.mutex_);
        std::lock_guard<std::mutex> lk1(mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> lk2(other.mutex_, std::adopt_lock);
        engine_ = std::move(other.engine_);
    }
    return *this;
}

ModelHandle::~ModelHandle() = default;

// ============================================================
// Inference
// ============================================================

Tensor ModelHandle::predict(const Tensor& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        return engine_.predict(input);
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        ErrorCode code = ErrorCode::SHAPE_MISMATCH;
        if (msg.find("NaN") != std::string::npos) {
            code = ErrorCode::NAN_INPUT;
        }
        throw ValidationException(msg, code);
    } catch (const std::runtime_error& e) {
        throw InferenceException(e.what(), ErrorCode::NO_MODEL_LOADED);
    }
}

std::vector<Tensor> ModelHandle::predict_batch(
        const std::vector<Tensor>& inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    try {
        for (const auto& input : inputs) {
            outputs.push_back(engine_.predict(input));
        }
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        ErrorCode code = ErrorCode::SHAPE_MISMATCH;
        if (msg.find("NaN") != std::string::npos) {
            code = ErrorCode::NAN_INPUT;
        }
        throw ValidationException(msg, code);
    } catch (const std::runtime_error& e) {
        throw InferenceException(e.what(), ErrorCode::NO_MODEL_LOADED);
    }
    return outputs;
}

// ============================================================
// Utility Methods
// ============================================================

InferenceStats ModelHandle::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.stats();
}

void ModelHandle::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.reset_stats();
}

bool ModelHandle::is_loaded() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.is_loaded();
}

size_t ModelHandle::layer_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.layer_count();
}

std::string ModelHandle::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        return engine_.summary();
    } catch (const std::runtime_error& e) {
        throw InferenceException(e.what(), ErrorCode::NO_MODEL_LOADED);
    }
}

const std::vector<size_t>& ModelHandle::expected_input_shape() const {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        return engine_.expected_input_shape();
    } catch (const std::runtime_error& e) {
        throw InferenceException(e.what(), ErrorCode::NO_MODEL_LOADED);
    }
}

} // namespace titaninfer
