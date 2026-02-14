#pragma once

#include <stdexcept>
#include <string>

namespace titaninfer {

/**
 * @brief Error codes for structured exception handling
 */
enum class ErrorCode : int {
    // General
    UNKNOWN           = 0,

    // Model loading errors (100-199)
    FILE_NOT_FOUND    = 100,
    INVALID_FORMAT    = 101,
    EMPTY_MODEL       = 102,

    // Inference errors (200-299)
    NO_MODEL_LOADED   = 200,
    SHAPE_MISMATCH    = 201,
    NAN_INPUT         = 202,

    // Internal errors (300-399)
    INTERNAL_ERROR    = 300,
};

/**
 * @brief Base exception for all TitanInfer errors
 *
 * Inherits from std::runtime_error so existing catch(const std::exception&)
 * blocks continue to work. Adds a machine-readable error_code().
 */
class TitanInferException : public std::runtime_error {
public:
    explicit TitanInferException(const std::string& msg,
                                 ErrorCode code = ErrorCode::UNKNOWN)
        : std::runtime_error(msg), code_(code) {}

    ErrorCode error_code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

/**
 * @brief Thrown when a model file cannot be loaded
 */
class ModelLoadException : public TitanInferException {
public:
    explicit ModelLoadException(const std::string& msg,
                                ErrorCode code = ErrorCode::FILE_NOT_FOUND)
        : TitanInferException(msg, code) {}
};

/**
 * @brief Thrown when inference fails at runtime
 */
class InferenceException : public TitanInferException {
public:
    explicit InferenceException(const std::string& msg,
                                ErrorCode code = ErrorCode::NO_MODEL_LOADED)
        : TitanInferException(msg, code) {}
};

/**
 * @brief Thrown when input validation fails (shape mismatch, NaN, etc.)
 */
class ValidationException : public TitanInferException {
public:
    explicit ValidationException(const std::string& msg,
                                 ErrorCode code = ErrorCode::SHAPE_MISMATCH)
        : TitanInferException(msg, code) {}
};

} // namespace titaninfer
