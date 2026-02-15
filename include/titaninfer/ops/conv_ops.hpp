#pragma once

#include "titaninfer/tensor.hpp"
#include <cstddef>

namespace titaninfer {
namespace ops {

enum class PaddingMode { VALID, SAME };

/**
 * @brief Compute output spatial dimension for convolution/pooling
 */
size_t conv_output_size(size_t input_size, size_t kernel_size,
                        size_t stride, size_t padding);

/**
 * @brief Compute padding needed for SAME mode (total padding, split evenly)
 */
size_t compute_same_padding(size_t input_size, size_t kernel_size,
                            size_t stride);

/**
 * @brief im2col: rearrange input patches into column matrix for GEMM-based convolution
 *
 * @param input 3D tensor (C_in, H, W)
 * @param col   Output 2D tensor (C_in*kH*kW, out_H*out_W), auto-allocated
 * @param kH, kW Kernel dimensions
 * @param stride_h, stride_w Stride
 * @param pad_h, pad_w Padding (applied symmetrically)
 */
void im2col(const Tensor& input, Tensor& col,
            size_t kH, size_t kW,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w);

/**
 * @brief col2im: inverse of im2col (accumulates into output)
 *
 * @param col    2D tensor (C_in*kH*kW, out_H*out_W)
 * @param output 3D tensor (C_in, H, W), auto-allocated
 */
void col2im(const Tensor& col, Tensor& output,
            size_t C_in, size_t H, size_t W,
            size_t kH, size_t kW,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w);

} // namespace ops
} // namespace titaninfer
