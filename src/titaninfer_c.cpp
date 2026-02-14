#include "titaninfer/titaninfer_c.h"
#include "titaninfer/TitanInfer.hpp"
#include <cstring>
#include <new>

struct TitanInferModel_ {
    titaninfer::ModelHandle handle;
    std::string last_error;

    explicit TitanInferModel_(titaninfer::ModelHandle h)
        : handle(std::move(h)) {}
};

extern "C" {

TitanInferModelHandle titaninfer_load(const char* model_path,
                                      const size_t* input_shape,
                                      size_t shape_len) {
    if (!model_path) {
        return nullptr;
    }

    try {
        auto builder = titaninfer::ModelHandle::Builder()
            .setModelPath(model_path)
            .setLogLevel(titaninfer::LogLevel::SILENT);

        if (input_shape && shape_len > 0) {
            std::vector<size_t> shape(input_shape, input_shape + shape_len);
            builder.setInputShape(shape);
        }

        auto handle = builder.build();
        return new TitanInferModel_(std::move(handle));
    } catch (...) {
        return nullptr;
    }
}

void titaninfer_free(TitanInferModelHandle handle) {
    delete handle;
}

int titaninfer_predict(TitanInferModelHandle handle,
                       const float* input_data,
                       size_t input_len,
                       float* output_data,
                       size_t output_len,
                       size_t* actual_output_len) {
    if (!handle || !input_data || !output_data || !actual_output_len) {
        if (handle) {
            handle->last_error = "null pointer argument";
        }
        return TITANINFER_ERR_INVALID_ARG;
    }

    try {
        const auto& shape = handle->handle.expected_input_shape();
        titaninfer::Tensor input(shape);

        // Verify input length matches expected shape
        if (input_len != input.size()) {
            handle->last_error = "input length " +
                std::to_string(input_len) + " does not match expected " +
                std::to_string(input.size());
            return TITANINFER_ERR_VALIDATION;
        }

        std::memcpy(input.data(), input_data, input_len * sizeof(float));

        titaninfer::Tensor result = handle->handle.predict(input);

        *actual_output_len = result.size();
        if (output_len < result.size()) {
            handle->last_error = "output buffer too small: need " +
                std::to_string(result.size()) + ", got " +
                std::to_string(output_len);
            return TITANINFER_ERR_INVALID_ARG;
        }

        std::memcpy(output_data, result.data(), result.size() * sizeof(float));
        handle->last_error.clear();
        return TITANINFER_OK;
    } catch (const titaninfer::ValidationException& e) {
        handle->last_error = e.what();
        return TITANINFER_ERR_VALIDATION;
    } catch (const titaninfer::TitanInferException& e) {
        handle->last_error = e.what();
        return TITANINFER_ERR_INFERENCE;
    } catch (const std::exception& e) {
        handle->last_error = e.what();
        return TITANINFER_ERR_INFERENCE;
    }
}

const char* titaninfer_last_error(TitanInferModelHandle handle) {
    if (!handle) {
        return nullptr;
    }
    if (handle->last_error.empty()) {
        return nullptr;
    }
    return handle->last_error.c_str();
}

size_t titaninfer_layer_count(TitanInferModelHandle handle) {
    if (!handle) {
        return 0;
    }
    return handle->handle.layer_count();
}

int titaninfer_is_loaded(TitanInferModelHandle handle) {
    if (!handle) {
        return 0;
    }
    return handle->handle.is_loaded() ? 1 : 0;
}

int titaninfer_inference_count(TitanInferModelHandle handle) {
    if (!handle) {
        return 0;
    }
    return static_cast<int>(handle->handle.stats().inference_count);
}

double titaninfer_mean_latency_ms(TitanInferModelHandle handle) {
    if (!handle) {
        return 0.0;
    }
    return handle->handle.stats().mean_latency_ms;
}

} // extern "C"
