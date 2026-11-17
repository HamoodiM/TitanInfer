#pragma once

#include "titaninfer/tensor.hpp"
#if defined(TITANINFER_ENABLE_SIMD)
#include <immintrin.h> // AVX2 intrinsics
#endif
#include <stdexcept>

namespace titaninfer {
namespace ops {
namespace simd {

/**
 * @brief AVX2-optimized matrix multiplication: C = A @ B
 *
 * Implements cache-blocked matrix multiplication with AVX2 vectorization:
 * - 8-wide float operations using 256-bit registers along K
 * - Loop tiling over M/N/K for cache reuse (no register blocking yet;
 *   each output element is accumulated in its own vector)
 * - FMA (fused multiply-add) when __FMA__ is defined (GCC/Clang with -mfma).
 *   MSVC never defines __FMA__, so MSVC builds use the mul+add fallback.
 *
 * @param A Left matrix (M, K)
 * @param B Right matrix (K, N)
 * @param C Output matrix (M, N)
 * @throws std::invalid_argument if shapes incompatible
 *
 * @performance Measured 1.01-1.23x over the naive loop (see docs/PERFORMANCE.md)
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
