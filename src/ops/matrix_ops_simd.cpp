#include "titaninfer/ops/matrix_ops_simd.hpp"
#include "titaninfer/ops/matrix_ops.hpp"
#include <cstring>
#include <sstream>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace titaninfer {
namespace ops {
namespace simd {

// ========================================
// CPU Feature Detection
// ========================================

bool cpu_supports_avx2_fma() {
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    bool avx = (cpu_info[2] & (1 << 28)) != 0;
    bool fma = (cpu_info[2] & (1 << 12)) != 0;
    
    __cpuid(cpu_info, 7);
    bool avx2 = (cpu_info[1] & (1 << 5)) != 0;
    
    return avx && avx2 && fma;
#else
    unsigned int eax, ebx, ecx, edx;
    
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    bool avx = (ecx & (1 << 28)) != 0;
    bool fma = (ecx & (1 << 12)) != 0;
    
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    bool avx2 = (ebx & (1 << 5)) != 0;
    
    return avx && avx2 && fma;
#endif
}

std::string get_cpu_features() {
    std::ostringstream oss;
    oss << "AVX2+FMA: " << (cpu_supports_avx2_fma() ? "YES" : "NO");
    return oss.str();
}

// ========================================
// AVX2 Helper: Horizontal Sum
// ========================================

/**
 * @brief Compute horizontal sum of 8 floats in AVX2 register
 * 
 * Input:  [a0, a1, a2, a3, a4, a5, a6, a7]
 * Output: a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
 */
inline float horizontal_sum_avx2(__m256 v) {
    // v = [a0, a1, a2, a3, a4, a5, a6, a7]
    
    // Step 1: Add upper and lower 128-bit halves
    __m128 hi = _mm256_extractf128_ps(v, 1);  // [a4, a5, a6, a7]
    __m128 lo = _mm256_castps256_ps128(v);    // [a0, a1, a2, a3]
    __m128 sum128 = _mm_add_ps(hi, lo);       // [a0+a4, a1+a5, a2+a6, a3+a7]
    
    // Step 2: Horizontal add within 128-bit register
    sum128 = _mm_hadd_ps(sum128, sum128);     // [a0+a1+a4+a5, a2+a3+a6+a7, ...]
    sum128 = _mm_hadd_ps(sum128, sum128);     // [sum_all, sum_all, ...]
    
    return _mm_cvtss_f32(sum128);
}

// ========================================
// AVX2 Matrix Multiplication
// ========================================

void matmul_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
    validate_matmul_shapes(A.shape(), B.shape());
    
    const size_t M = A.shape()[0];
    const size_t K = A.shape()[1];
    const size_t N = B.shape()[1];
    
    if (C.shape() != std::vector<size_t>{M, N}) {
        C = Tensor({M, N});
    }
    
    C.zero();
    
    // Block sizes tuned for L1 cache
    constexpr size_t MC = 64;  // Reduced for better cache locality
    constexpr size_t NC = 64;
    constexpr size_t KC = 256;
    
    const float* A_data = A.data();
    const float* B_data = B.data();
    float* C_data = C.data();
    
    // Blocked matrix multiplication: C += A @ B
    for (size_t i = 0; i < M; i += MC) {
        size_t ib = std::min(MC, M - i);
        
        for (size_t k = 0; k < K; k += KC) {
            size_t kb = std::min(KC, K - k);
            
            for (size_t j = 0; j < N; j += NC) {
                size_t jb = std::min(NC, N - j);
                
                // Micro-kernel: process ib x jb block
                for (size_t ii = i; ii < i + ib; ++ii) {
                    for (size_t jj = j; jj < j + jb; ++jj) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        
                        // Vectorized inner loop (process 8 elements at a time)
                        size_t kk = k;
                        for (; kk + 8 <= k + kb; kk += 8) {
                            // Load 8 elements from A[ii, kk:kk+8]
                            __m256 a_vec = _mm256_loadu_ps(A_data + ii * K + kk);
                            
                            // Load 8 elements from B[kk:kk+8, jj]
                            // Note: B is row-major, so we need to gather elements
                            __m256 b_vec = _mm256_set_ps(
                                B_data[(kk + 7) * N + jj],
                                B_data[(kk + 6) * N + jj],
                                B_data[(kk + 5) * N + jj],
                                B_data[(kk + 4) * N + jj],
                                B_data[(kk + 3) * N + jj],
                                B_data[(kk + 2) * N + jj],
                                B_data[(kk + 1) * N + jj],
                                B_data[(kk + 0) * N + jj]
                            );
                            
#ifdef __FMA__
                            // Use fused multiply-add
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
#else
                            // Manual multiply-add
                            __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                            sum_vec = _mm256_add_ps(sum_vec, prod);
#endif
                        }
                        
                        // Horizontal sum of vector
                        float sum = horizontal_sum_avx2(sum_vec);
                        
                        // Scalar remainder loop
                        for (; kk < k + kb; ++kk) {
                            sum += A_data[ii * K + kk] * B_data[kk * N + jj];
                        }
                        
                        C_data[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

} // namespace simd
} // namespace ops
} // namespace titaninfer
