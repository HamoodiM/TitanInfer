#pragma once

#include "titaninfer/tensor.hpp"
#include <immintrin.h> // AVX2 intrinsics
#include <stdexcept>

namespace titaninfer {
namespace ops {
namespace simd {

/**
 * @brief AVX2-optimized matrix multiplication: C = A @ B
 * 
 * Implements blocked matrix multiplication with AVX2 vectorization:
 * - 8-wide float operations using 256-bit registers
 * - 4x4 register blocking for L1 cache efficiency
 * - FMA (fused multiply-add) when available
 * 
 * @param A Left matrix (M, K)
 * @param B Right matrix (K, N)
 * @param C Output matrix (M, N)
 * @throws std::invalid_argument if shapes incompatible
 * 
 * @performance Expected 3-5x speedup over naive implementation
 */
void matmul_avx2(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Check if CPU supports AVX2 and FMA
 * @return true if AVX2 + FMA available
 */
bool cpu_supports_avx2_fma();

/**
 * @brief Get CPU feature string for diagnostics
 */
std::string get_cpu_features();

} // namespace simd
} // namespace ops
} // namespace titaninfer
