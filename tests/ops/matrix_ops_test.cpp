#include <gtest/gtest.h>
#include "titaninfer/ops/matrix_ops.hpp"
#include <cmath>

using namespace titaninfer;
using namespace titaninfer::ops;

// ========================================
// Matrix Multiplication Tests
// ========================================

TEST(MatrixOpsTest, MatMulBasic) {
    // A = [[1, 2],     B = [[5, 6],
    //      [3, 4]]          [7, 8]]
    // 
    // Expected C = [[19, 22],
    //               [43, 50]]
    
    Tensor A({2, 2});
    A.data()[0] = 1.0f; A.data()[1] = 2.0f;
    A.data()[2] = 3.0f; A.data()[3] = 4.0f;
    
    Tensor B({2, 2});
    B.data()[0] = 5.0f; B.data()[1] = 6.0f;
    B.data()[2] = 7.0f; B.data()[3] = 8.0f;
    
    Tensor C({2, 2});
    matmul(A, B, C);
    
    EXPECT_FLOAT_EQ(C.data()[0], 19.0f);
    EXPECT_FLOAT_EQ(C.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(C.data()[2], 43.0f);
    EXPECT_FLOAT_EQ(C.data()[3], 50.0f);
}

TEST(MatrixOpsTest, MatMulRectangular) {
    // A(3x2) @ B(2x4) = C(3x4)
    Tensor A({3, 2});
    for (size_t i = 0; i < 6; ++i) {
        A.data()[i] = static_cast<float>(i + 1);
    }
    
    Tensor B({2, 4});
    for (size_t i = 0; i < 8; ++i) {
        B.data()[i] = static_cast<float>(i + 1);
    }
    
    Tensor C({1, 1}); // Wrong shape initially
    matmul(A, B, C);
    
    EXPECT_EQ(C.shape()[0], 3);
    EXPECT_EQ(C.shape()[1], 4);
    
    // Verify first element: A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 2*5 = 11
    EXPECT_FLOAT_EQ(C.data()[0], 11.0f);
}

TEST(MatrixOpsTest, MatMulIdentity) {
    Tensor I({3, 3});
    I.zero();
    I.data()[0] = 1.0f; // [0,0]
    I.data()[4] = 1.0f; // [1,1]
    I.data()[8] = 1.0f; // [2,2]
    
    Tensor A({3, 3});
    for (size_t i = 0; i < 9; ++i) {
        A.data()[i] = static_cast<float>(i);
    }
    
    Tensor C({3, 3});
    matmul(I, A, C);
    
    // I @ A should equal A
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(C.data()[i], A.data()[i]);
    }
}

TEST(MatrixOpsTest, MatMulInvalidShapes) {
    Tensor A({2, 3});
    Tensor B({2, 4}); // Incompatible: A cols != B rows
    Tensor C({2, 4});
    
    EXPECT_THROW(matmul(A, B, C), std::invalid_argument);
}

// ========================================
// Matrix-Vector Multiplication Tests
// ========================================

TEST(MatrixOpsTest, MatVecBasic) {
    // A = [[1, 2, 3],    x = [1]
    //      [4, 5, 6]]        [2]
    //                        [3]
    // y = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
    
    Tensor A({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        A.data()[i] = static_cast<float>(i + 1);
    }
    
    Tensor x({3});
    x.data()[0] = 1.0f;
    x.data()[1] = 2.0f;
    x.data()[2] = 3.0f;
    
    Tensor y({2});
    matvec(A, x, y);
    
    EXPECT_FLOAT_EQ(y.data()[0], 14.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 32.0f);
}

// ========================================
// Transpose Tests
// ========================================

TEST(MatrixOpsTest, TransposeSquare) {
    Tensor A({3, 3});
    for (size_t i = 0; i < 9; ++i) {
        A.data()[i] = static_cast<float>(i);
    }
    
    Tensor B({3, 3});
    transpose(A, B);
    
    // Verify B[j,i] = A[i,j]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(B.data()[j * 3 + i], A.data()[i * 3 + j]);
        }
    }
}

TEST(MatrixOpsTest, TransposeRectangular) {
    Tensor A({2, 3});
    A.data()[0] = 1.0f; A.data()[1] = 2.0f; A.data()[2] = 3.0f;
    A.data()[3] = 4.0f; A.data()[4] = 5.0f; A.data()[5] = 6.0f;
    
    Tensor B({3, 2});
    transpose(A, B);
    
    // Expected: [[1, 4],
    //            [2, 5],
    //            [3, 6]]
    EXPECT_FLOAT_EQ(B.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(B.data()[1], 4.0f);
    EXPECT_FLOAT_EQ(B.data()[2], 2.0f);
    EXPECT_FLOAT_EQ(B.data()[3], 5.0f);
    EXPECT_FLOAT_EQ(B.data()[4], 3.0f);
    EXPECT_FLOAT_EQ(B.data()[5], 6.0f);
}

// ========================================
// Element-wise Operation Tests
// ========================================

TEST(MatrixOpsTest, AddTensors) {
    Tensor A({3});
    A.data()[0] = 1.0f; A.data()[1] = 2.0f; A.data()[2] = 3.0f;
    
    Tensor B({3});
    B.data()[0] = 4.0f; B.data()[1] = 5.0f; B.data()[2] = 6.0f;
    
    Tensor C({3});
    add(A, B, C);
    
    EXPECT_FLOAT_EQ(C.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(C.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(C.data()[2], 9.0f);
}

TEST(MatrixOpsTest, AddScalar) {
    Tensor A({4});
    A.fill(10.0f);
    
    Tensor C({4});
    add_scalar(A, 5.0f, C);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(C.data()[i], 15.0f);
    }
}

TEST(MatrixOpsTest, MultiplyTensors) {
    Tensor A({3});
    A.data()[0] = 2.0f; A.data()[1] = 3.0f; A.data()[2] = 4.0f;
    
    Tensor B({3});
    B.data()[0] = 5.0f; B.data()[1] = 6.0f; B.data()[2] = 7.0f;
    
    Tensor C({3});
    multiply(A, B, C);
    
    EXPECT_FLOAT_EQ(C.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(C.data()[1], 18.0f);
    EXPECT_FLOAT_EQ(C.data()[2], 28.0f);
}

TEST(MatrixOpsTest, MultiplyScalar) {
    Tensor A({5});
    for (size_t i = 0; i < 5; ++i) {
        A.data()[i] = static_cast<float>(i + 1);
    }
    
    Tensor C({5});
    multiply_scalar(A, 2.5f, C);
    
    EXPECT_FLOAT_EQ(C.data()[0], 2.5f);
    EXPECT_FLOAT_EQ(C.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(C.data()[2], 7.5f);
    EXPECT_FLOAT_EQ(C.data()[3], 10.0f);
    EXPECT_FLOAT_EQ(C.data()[4], 12.5f);
}

TEST(MatrixOpsTest, AddShapeMismatch) {
    Tensor A({2, 3});
    Tensor B({3, 2});
    Tensor C({2, 3});
    
    EXPECT_THROW(add(A, B, C), std::invalid_argument);
}

// ========================================
// Performance Baseline Test
// ========================================

TEST(MatrixOpsTest, MatMulPerformanceBaseline) {
    // Small benchmark for Phase 3 comparison
    const size_t N = 128;
    Tensor A({N, N});
    Tensor B({N, N});
    Tensor C({N, N});
    
    A.fill(1.0f);
    B.fill(2.0f);
    
    // This will be slow (naive implementation)
    // We'll optimize in Phase 3 with AVX2
    matmul(A, B, C);
    
    // Expected: each element should be N * 1.0 * 2.0 = 2*N
    EXPECT_FLOAT_EQ(C.data()[0], static_cast<float>(2 * N));
}
