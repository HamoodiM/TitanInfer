#pragma once

#include "titaninfer/tensor.hpp"
#include <stdexcept>

namespace titaninfer {
namespace ops {

/**
 * @brief Matrix multiplication: C = A @ B (naive implementation)
 * 
 * Computes matrix product using standard 3-loop algorithm.
 * 
 * @param A Left matrix with shape (M, K)
 * @param B Right matrix with shape (K, N)
 * @param C Output matrix with shape (M, N) (will be allocated/resized)
 * @throws std::invalid_argument if shapes are incompatible
 * 
 * @note This is the baseline implementation. AVX2 optimization in Phase 3.
 */
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Matrix-vector multiplication: y = A @ x
 * 
 * @param A Matrix with shape (M, N)
 * @param x Vector with shape (N,)
 * @param y Output vector with shape (M,)
 * @throws std::invalid_argument if shapes are incompatible
 */
void matvec(const Tensor& A, const Tensor& x, Tensor& y);

/**
 * @brief Transpose matrix: B = A^T
 * 
 * @param A Input matrix with shape (M, N)
 * @param B Output matrix with shape (N, M)
 */
void transpose(const Tensor& A, Tensor& B);

/**
 * @brief Element-wise addition: C = A + B
 * 
 * @param A First tensor
 * @param B Second tensor (must match A's shape)
 * @param C Output tensor
 * @throws std::invalid_argument if shapes don't match
 */
void add(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Element-wise addition with broadcasting: C = A + b (scalar)
 * 
 * @param A Tensor
 * @param b Scalar value to add
 * @param C Output tensor (same shape as A)
 */
void add_scalar(const Tensor& A, float b, Tensor& C);

/**
 * @brief Element-wise multiplication: C = A * B (Hadamard product)
 * 
 * @param A First tensor
 * @param B Second tensor (must match A's shape)
 * @param C Output tensor
 * @throws std::invalid_argument if shapes don't match
 */
void multiply(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Element-wise multiplication by scalar: C = A * b
 * 
 * @param A Tensor
 * @param b Scalar multiplier
 * @param C Output tensor
 */
void multiply_scalar(const Tensor& A, float b, Tensor& C);

/**
 * @brief Validate matrix multiplication shape compatibility
 * 
 * @param A_shape Shape of left matrix
 * @param B_shape Shape of right matrix
 * @throws std::invalid_argument if incompatible
 */
void validate_matmul_shapes(const std::vector<size_t>& A_shape,
                            const std::vector<size_t>& B_shape);

} // namespace ops
} // namespace titaninfer
