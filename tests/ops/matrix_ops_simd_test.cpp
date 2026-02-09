#include <gtest/gtest.h>
#include "titaninfer/ops/matrix_ops_simd.hpp"
#include "titaninfer/ops/matrix_ops.hpp"
#include <cmath>
#include <iostream>

using namespace titaninfer;
using namespace titaninfer::ops;

// ========================================
// CPU Feature Detection Tests
// ========================================

TEST(SIMDTest, CPUFeatureDetection) {
    std::cout << "\n=== CPU Features ===" << std::endl;
    std::cout << simd::get_cpu_features() << std::endl;
    
    bool has_avx2 = simd::cpu_supports_avx2_fma();
    std::cout << "AVX2+FMA: " << (has_avx2 ? "Supported" : "Not Supported") << std::endl;
    
    // This test always passes, but logs CPU capabilities
    EXPECT_TRUE(true);
}

// ========================================
// Correctness Tests (vs Naive)
// ========================================

TEST(SIMDTest, MatMulCorrectnessSmall) {
    // A = [[1, 2],     B = [[5, 6],
    //      [3, 4]]          [7, 8]]
    // Expected C = [[19, 22], [43, 50]]
    
    Tensor A({2, 2});
    A.data()[0] = 1.0f; A.data()[1] = 2.0f;
    A.data()[2] = 3.0f; A.data()[3] = 4.0f;
    
    Tensor B({2, 2});
    B.data()[0] = 5.0f; B.data()[1] = 6.0f;
    B.data()[2] = 7.0f; B.data()[3] = 8.0f;
    
    Tensor C_naive({2, 2});
    Tensor C_simd({2, 2});
    
    matmul(A, B, C_naive);              // Naive
    simd::matmul_avx2(A, B, C_simd);    // AVX2
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(C_simd.data()[i], C_naive.data()[i]) 
            << "Mismatch at index " << i;
    }
}

TEST(SIMDTest, MatMulCorrectnessMedium) {
    const size_t M = 64, K = 64, N = 64;
    
    Tensor A({M, K});
    Tensor B({K, N});
    
    // Fill with known pattern
    for (size_t i = 0; i < M * K; ++i) {
        A.data()[i] = static_cast<float>(i % 17) / 10.0f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        B.data()[i] = static_cast<float>(i % 13) / 10.0f;
    }
    
    Tensor C_naive({M, N});
    Tensor C_simd({M, N});
    
    matmul(A, B, C_naive);
    simd::matmul_avx2(A, B, C_simd);
    
    // Check relative error
    for (size_t i = 0; i < M * N; ++i) {
        float expected = C_naive.data()[i];
        float actual = C_simd.data()[i];
        float rel_error = std::abs(expected - actual) / (std::abs(expected) + 1e-6f);
        
        EXPECT_LT(rel_error, 1e-4f) 
            << "Large relative error at index " << i 
            << ": expected=" << expected << ", actual=" << actual;
    }
}

TEST(SIMDTest, MatMulCorrectnessRectangular) {
    const size_t M = 128, K = 256, N = 64;
    
    Tensor A({M, K});
    Tensor B({K, N});
    A.fill(0.5f);
    B.fill(2.0f);
    
    Tensor C_naive({M, N});
    Tensor C_simd({M, N});
    
    matmul(A, B, C_naive);
    simd::matmul_avx2(A, B, C_simd);
    
    // Expected: each element = K * 0.5 * 2.0 = K
    float expected = static_cast<float>(K);
    
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_simd.data()[i], expected, 1e-3f);
        EXPECT_NEAR(C_simd.data()[i], C_naive.data()[i], 1e-3f);
    }
}

// ========================================
// Numerical Stability Tests
// ========================================

TEST(SIMDTest, NumericalStability) {
    const size_t N = 100;
    Tensor A({N, N});
    Tensor B({N, N});
    
    // Fill with values that could cause precision issues
    for (size_t i = 0; i < N * N; ++i) {
        A.data()[i] = 1.0f / (i + 1.0f);
        B.data()[i] = static_cast<float>(i + 1);
    }
    
    Tensor C_naive({N, N});
    Tensor C_simd({N, N});
    
    matmul(A, B, C_naive);
    simd::matmul_avx2(A, B, C_simd);
    
    double max_error = 0.0;
    for (size_t i = 0; i < N * N; ++i) {
        double error = std::abs(C_simd.data()[i] - C_naive.data()[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max absolute error: " << max_error << std::endl;
    
    // Relaxed tolerance: SIMD can have different rounding due to operation reordering
    EXPECT_LT(max_error, 0.01) << "Numerical stability issue detected";
    
    // Verify relative error is still small
    double max_rel_error = 0.0;
    for (size_t i = 0; i < N * N; ++i) {
        if (std::abs(C_naive.data()[i]) > 1e-6f) {
            double rel_error = std::abs(C_simd.data()[i] - C_naive.data()[i]) / std::abs(C_naive.data()[i]);
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }
    
    std::cout << "Max relative error: " << max_rel_error << std::endl;
    EXPECT_LT(max_rel_error, 0.01) << "Relative error too large";
}
