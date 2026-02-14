#ifndef TITANINFER_C_H
#define TITANINFER_C_H

/**
 * @file titaninfer_c.h
 * @brief C-compatible API for TitanInfer
 *
 * Opaque handle pattern for use with C, Python ctypes, Rust bindgen,
 * and other FFI systems. All C++ exceptions are caught and translated
 * to integer status codes. Error messages are stored per-handle.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes */
#define TITANINFER_OK              0
#define TITANINFER_ERR_LOAD        1
#define TITANINFER_ERR_INFERENCE   2
#define TITANINFER_ERR_VALIDATION  3
#define TITANINFER_ERR_INVALID_ARG 4

/* Opaque model handle */
typedef struct TitanInferModel_* TitanInferModelHandle;

/**
 * @brief Load a .titan model file
 * @param model_path   Path to the .titan model file
 * @param input_shape  Array of input dimensions (can be NULL to auto-infer)
 * @param shape_len    Length of input_shape array (0 if input_shape is NULL)
 * @return Handle to the loaded model, or NULL on failure
 */
TitanInferModelHandle titaninfer_load(const char* model_path,
                                      const size_t* input_shape,
                                      size_t shape_len);

/**
 * @brief Free a model handle
 * @param handle  Handle to free (NULL-safe)
 */
void titaninfer_free(TitanInferModelHandle handle);

/**
 * @brief Run inference on input data
 * @param handle            Model handle
 * @param input_data        Input float array
 * @param input_len         Number of floats in input_data
 * @param output_data       Pre-allocated output buffer
 * @param output_len        Size of output buffer (in floats)
 * @param actual_output_len Filled with actual output size
 * @return TITANINFER_OK on success, error code on failure
 */
int titaninfer_predict(TitanInferModelHandle handle,
                       const float* input_data,
                       size_t input_len,
                       float* output_data,
                       size_t output_len,
                       size_t* actual_output_len);

/**
 * @brief Get the last error message for a handle
 * @param handle  Model handle
 * @return Error string (valid until next call on the same handle), or NULL
 */
const char* titaninfer_last_error(TitanInferModelHandle handle);

/**
 * @brief Get the number of layers in the model
 */
size_t titaninfer_layer_count(TitanInferModelHandle handle);

/**
 * @brief Check if a model is loaded
 * @return 1 if loaded, 0 if not
 */
int titaninfer_is_loaded(TitanInferModelHandle handle);

/**
 * @brief Get the number of inference calls made
 */
int titaninfer_inference_count(TitanInferModelHandle handle);

/**
 * @brief Get the mean inference latency in milliseconds
 */
double titaninfer_mean_latency_ms(TitanInferModelHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* TITANINFER_C_H */
