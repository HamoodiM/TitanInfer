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
#include "titaninfer/quantized_tensor.hpp"
#include "titaninfer/exceptions.hpp"
#include "titaninfer/logger.hpp"
#include "titaninfer/model_handle.hpp"
#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/layers/pooling_layers.hpp"
#include "titaninfer/layers/flatten_layer.hpp"
#include "titaninfer/layers/fused_layers.hpp"
#include "titaninfer/layers/quantized_dense_layer.hpp"
#include "titaninfer/engine/thread_pool.hpp"
#include "titaninfer/engine/fusion.hpp"
#include "titaninfer/engine/dynamic_batcher.hpp"
#include "titaninfer/engine/model_compiler.hpp"
