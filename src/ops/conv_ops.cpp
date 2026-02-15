#include "titaninfer/ops/conv_ops.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace titaninfer {
namespace ops {

size_t conv_output_size(size_t input_size, size_t kernel_size,
                        size_t stride, size_t padding) {
    if (stride == 0) {
        throw std::invalid_argument("conv_output_size: stride must be > 0");
    }
    if (input_size + 2 * padding < kernel_size) {
        throw std::invalid_argument(
            "conv_output_size: input + padding too small for kernel");
    }
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

size_t compute_same_padding(size_t input_size, size_t kernel_size,
                            size_t stride) {
    size_t out_size = (input_size + stride - 1) / stride;
    size_t needed = (out_size - 1) * stride + kernel_size;
    if (needed > input_size) {
        return (needed - input_size) / 2;
    }
    return 0;
}

void im2col(const Tensor& input, Tensor& col,
            size_t kH, size_t kW,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("im2col: input must be 3D (C, H, W)");
    }

    const size_t C = input.shape()[0];
    const size_t H = input.shape()[1];
    const size_t W = input.shape()[2];

    const size_t out_h = conv_output_size(H, kH, stride_h, pad_h);
    const size_t out_w = conv_output_size(W, kW, stride_w, pad_w);
    const size_t col_rows = C * kH * kW;
    const size_t col_cols = out_h * out_w;

    // Auto-allocate output
    std::vector<size_t> expected_shape = {col_rows, col_cols};
    if (col.shape() != expected_shape) {
        col = Tensor(expected_shape);
    }

    const float* in_data = input.data();
    float* col_data = col.data();

    for (size_t c = 0; c < C; ++c) {
        for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
                size_t row = c * kH * kW + kh * kW + kw;
                for (size_t oh = 0; oh < out_h; ++oh) {
                    size_t ih = oh * stride_h + kh;
                    // Account for padding offset
                    bool h_in_range = (ih >= pad_h) && (ih - pad_h < H);
                    size_t real_ih = ih - pad_h;

                    for (size_t ow = 0; ow < out_w; ++ow) {
                        size_t iw = ow * stride_w + kw;
                        bool w_in_range = (iw >= pad_w) && (iw - pad_w < W);

                        if (h_in_range && w_in_range) {
                            size_t real_iw = iw - pad_w;
                            col_data[row * col_cols + oh * out_w + ow] =
                                in_data[c * H * W + real_ih * W + real_iw];
                        } else {
                            col_data[row * col_cols + oh * out_w + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void col2im(const Tensor& col, Tensor& output,
            size_t C_in, size_t H, size_t W,
            size_t kH, size_t kW,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w) {
    const size_t out_h = conv_output_size(H, kH, stride_h, pad_h);
    const size_t out_w = conv_output_size(W, kW, stride_w, pad_w);
    const size_t col_cols = out_h * out_w;

    std::vector<size_t> expected_shape = {C_in, H, W};
    if (output.shape() != expected_shape) {
        output = Tensor(expected_shape);
    }
    output.zero();

    const float* col_data = col.data();
    float* out_data = output.data();

    for (size_t c = 0; c < C_in; ++c) {
        for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
                size_t row = c * kH * kW + kh * kW + kw;
                for (size_t oh = 0; oh < out_h; ++oh) {
                    size_t ih = oh * stride_h + kh;
                    bool h_in_range = (ih >= pad_h) && (ih - pad_h < H);
                    size_t real_ih = ih - pad_h;

                    for (size_t ow = 0; ow < out_w; ++ow) {
                        size_t iw = ow * stride_w + kw;
                        bool w_in_range = (iw >= pad_w) && (iw - pad_w < W);

                        if (h_in_range && w_in_range) {
                            size_t real_iw = iw - pad_w;
                            out_data[c * H * W + real_ih * W + real_iw] +=
                                col_data[row * col_cols + oh * out_w + ow];
                        }
                    }
                }
            }
        }
    }
}

} // namespace ops
} // namespace titaninfer
