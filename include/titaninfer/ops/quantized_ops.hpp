#pragma once

#include "titaninfer/quantized_tensor.hpp"
#include "titaninfer/tensor.hpp"

namespace titaninfer {
namespace ops {

/**
 * @brief INT8 GEMM: C (float32) = dequant(A_int8 @ B_int8)
 *
 * A: (M, K), B: (K, N), C: (M, N)
 * Output is in float32 with scale correction applied.
 * Uses AVX2 acceleration when available.
 */
void gemm_int8(const QuantizedTensor& A, const QuantizedTensor& B,
               Tensor& C);

} // namespace ops
} // namespace titaninfer
