#include "titaninfer/ops/quantized_ops.hpp"

#include <stdexcept>

#ifdef TITANINFER_ENABLE_SIMD
#include <immintrin.h>
#endif

namespace titaninfer {
namespace ops {

namespace {

void gemm_int8_impl(const int8_t* A, const int8_t* B,
                    float* C,
                    size_t M, size_t K, size_t N,
                    float scale_a, int8_t zp_a,
                    float scale_b, int8_t zp_b) {
    const float output_scale = scale_a * scale_b;
    const int32_t zp_a_32 = static_cast<int32_t>(zp_a);
    const int32_t zp_b_32 = static_cast<int32_t>(zp_b);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            size_t k = 0;

#ifdef TITANINFER_ENABLE_SIMD
            // Process 8 elements at a time using int16 widening
            // Avoids _mm_maddubs_epi16 saturation (uint8*int8 pairs can overflow int16)
            __m128i zpa16 = _mm_set1_epi16(static_cast<int16_t>(zp_a));
            __m128i zpb16 = _mm_set1_epi16(static_cast<int16_t>(zp_b));

            for (; k + 8 <= K; k += 8) {
                // Load 8 int8 values from A (contiguous)
                __m128i a_raw = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(&A[i * K + k]));

                // Load 8 int8 values from B (scattered column access)
                int8_t b_vals[8];
                for (size_t kk = 0; kk < 8; ++kk) {
                    b_vals[kk] = B[(k + kk) * N + j];
                }
                __m128i b_raw = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(b_vals));

                // Sign-extend int8 -> int16
                __m128i a16 = _mm_cvtepi8_epi16(a_raw);
                __m128i b16 = _mm_cvtepi8_epi16(b_raw);

                // Subtract zero points in int16 (no overflow: int8 - int8 fits int16)
                a16 = _mm_sub_epi16(a16, zpa16);
                b16 = _mm_sub_epi16(b16, zpb16);

                // Multiply pairs and horizontally add to int32 (no saturation)
                __m128i prod32 = _mm_madd_epi16(a16, b16);

                // Horizontal sum of 4 int32
                int32_t temp[4];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(temp), prod32);
                acc += temp[0] + temp[1] + temp[2] + temp[3];
            }
#endif
            // Scalar path (or scalar tail for SIMD)
            for (; k < K; ++k) {
                int32_t a_val = static_cast<int32_t>(A[i * K + k]) - zp_a_32;
                int32_t b_val = static_cast<int32_t>(B[k * N + j]) - zp_b_32;
                acc += a_val * b_val;
            }

            C[i * N + j] = static_cast<float>(acc) * output_scale;
        }
    }
}

} // anonymous namespace

void gemm_int8(const QuantizedTensor& A, const QuantizedTensor& B,
               Tensor& C) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("gemm_int8: A and B must be 2D");
    }
    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument(
            "gemm_int8: inner dimensions must match");
    }

    const size_t M = A.shape()[0];
    const size_t K = A.shape()[1];
    const size_t N = B.shape()[1];

    std::vector<size_t> expected = {M, N};
    if (C.shape() != expected) {
        C = Tensor(expected);
    }

    gemm_int8_impl(A.data(), B.data(), C.data(), M, K, N,
                   A.scale(), A.zero_point(), B.scale(), B.zero_point());
}

} // namespace ops
} // namespace titaninfer
