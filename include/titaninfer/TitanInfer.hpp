#pragma once

/**
 * @file TitanInfer.hpp
 * @brief Single-include public API header for TitanInfer
 *
 * Include this header to access the complete public API:
 *   - titaninfer::ModelHandle  (thread-safe inference)
 *   - titaninfer::Tensor       (N-dimensional tensor)
 *   - titaninfer::Logger       (configurable logging)
 *   - titaninfer exception hierarchy
 *
 * @code
 *   #include "titaninfer/TitanInfer.hpp"
 *
 *   auto model = titaninfer::ModelHandle::Builder()
 *       .setModelPath("model.titan")
 *       .build();
 *   titaninfer::Tensor result = model.predict(input);
 * @endcode
 */

#include "titaninfer/tensor.hpp"
#include "titaninfer/exceptions.hpp"
#include "titaninfer/logger.hpp"
#include "titaninfer/model_handle.hpp"
