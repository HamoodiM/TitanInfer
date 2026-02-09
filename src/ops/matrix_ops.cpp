#include "titaninfer/ops/matrix_ops.hpp"
#include <algorithm>
#include <sstream>

namespace titaninfer {
namespace ops {

// ========================================
// Shape Validation Utilities
// ========================================

void validate_matmul_shapes(const std::vector<size_t>& A_shape,
                            const std::vector<size_t>& B_shape) {
    if (A_shape.size() != 2 || B_shape.size() != 2) {
        throw std::invalid_argument(
            "matmul requires 2D matrices, got shapes with dimensions " +
            std::to_string(A_shape.size()) + " and " + std::to_string(B_shape.size())
        );
    }
    
    if (A_shape[1] != B_shape[0]) {
        std::ostringstream oss;
        oss << "matmul shape mismatch: A(" << A_shape[0] << ", " << A_shape[1]
            << ") @ B(" << B_shape[0] << ", " << B_shape[1] << ") - "
            << "inner dimensions must match";
        throw std::invalid_argument(oss.str());
    }
}

// ========================================
// Matrix Multiplication
// ========================================

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    validate_matmul_shapes(A.shape(), B.shape());
    
    const size_t M = A.shape()[0];  // Rows of A
    const size_t K = A.shape()[1];  // Cols of A / Rows of B
    const size_t N = B.shape()[1];  // Cols of B
    
    // Allocate output if needed
    if (C.shape() != std::vector<size_t>{M, N}) {
        C = Tensor({M, N});
    }
    
    C.zero();
    
    // Naive O(MNK) algorithm: C[i,j] = sum_k A[i,k] * B[k,j]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A.data()[i * K + k] * B.data()[k * N + j];
            }
            C.data()[i * N + j] = sum;
        }
    }
}

// ========================================
// Matrix-Vector Multiplication
// ========================================

void matvec(const Tensor& A, const Tensor& x, Tensor& y) {
    if (A.ndim() != 2) {
        throw std::invalid_argument("matvec requires 2D matrix");
    }
    if (x.ndim() != 1) {
        throw std::invalid_argument("matvec requires 1D vector");
    }
    if (A.shape()[1] != x.shape()[0]) {
        throw std::invalid_argument(
            "matvec shape mismatch: A cols must match x size"
        );
    }
    
    const size_t M = A.shape()[0];
    const size_t N = A.shape()[1];
    
    if (y.shape() != std::vector<size_t>{M}) {
        y = Tensor({M});
    }
    
    y.zero();
    
    // y[i] = sum_j A[i,j] * x[j]
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum += A.data()[i * N + j] * x.data()[j];
        }
        y.data()[i] = sum;
    }
}

// ========================================
// Transpose
// ========================================

void transpose(const Tensor& A, Tensor& B) {
    if (A.ndim() != 2) {
        throw std::invalid_argument("transpose requires 2D matrix");
    }
    
    const size_t M = A.shape()[0];
    const size_t N = A.shape()[1];
    
    if (B.shape() != std::vector<size_t>{N, M}) {
        B = Tensor({N, M});
    }
    
    // B[j,i] = A[i,j]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B.data()[j * M + i] = A.data()[i * N + j];
        }
    }
}

// ========================================
// Element-wise Operations
// ========================================

void add(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("add requires tensors with matching shapes");
    }
    
    if (C.shape() != A.shape()) {
        C = Tensor(A.shape());
    }
    
    const size_t total = A.size();
    for (size_t i = 0; i < total; ++i) {
        C.data()[i] = A.data()[i] + B.data()[i];
    }
}

void add_scalar(const Tensor& A, float b, Tensor& C) {
    if (C.shape() != A.shape()) {
        C = Tensor(A.shape());
    }
    
    const size_t total = A.size();
    for (size_t i = 0; i < total; ++i) {
        C.data()[i] = A.data()[i] + b;
    }
}

void multiply(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("multiply requires tensors with matching shapes");
    }
    
    if (C.shape() != A.shape()) {
        C = Tensor(A.shape());
    }
    
    const size_t total = A.size();
    for (size_t i = 0; i < total; ++i) {
        C.data()[i] = A.data()[i] * B.data()[i];
    }
}

void multiply_scalar(const Tensor& A, float b, Tensor& C) {
    if (C.shape() != A.shape()) {
        C = Tensor(A.shape());
    }
    
    const size_t total = A.size();
    for (size_t i = 0; i < total; ++i) {
        C.data()[i] = A.data()[i] * b;
    }
}

} // namespace ops
} // namespace titaninfer
