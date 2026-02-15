#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/ops/matrix_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace titaninfer {
namespace layers {

Conv2DLayer::Conv2DLayer(size_t in_channels, size_t out_channels,
                         size_t kernel_h, size_t kernel_w,
                         size_t stride_h, size_t stride_w,
                         ops::PaddingMode padding, bool use_bias)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_h_(kernel_h)
    , kernel_w_(kernel_w)
    , stride_h_(stride_h)
    , stride_w_(stride_w)
    , padding_(padding)
    , use_bias_(use_bias)
    , weights_({out_channels, in_channels, kernel_h, kernel_w})
    , bias_({out_channels})
    , col_buf_({1})
    , weights_2d_({1})
    , gemm_buf_({1})
{
    if (in_channels == 0 || out_channels == 0) {
        throw std::invalid_argument(
            "Conv2DLayer: channels must be > 0");
    }
    if (kernel_h == 0 || kernel_w == 0) {
        throw std::invalid_argument(
            "Conv2DLayer: kernel size must be > 0");
    }
    if (stride_h == 0 || stride_w == 0) {
        throw std::invalid_argument(
            "Conv2DLayer: stride must be > 0");
    }

    // Cache 2D weight view: (C_out, C_in*kH*kW)
    // The data layout is identical in row-major
    size_t fan_in = in_channels * kernel_h * kernel_w;
    weights_2d_ = Tensor({out_channels, fan_in});
    std::memcpy(weights_2d_.data(), weights_.data(),
                out_channels * fan_in * sizeof(float));
}

Conv2DLayer::Conv2DLayer(size_t in_channels, size_t out_channels,
                         size_t kernel_size, size_t stride,
                         ops::PaddingMode padding, bool use_bias)
    : Conv2DLayer(in_channels, out_channels,
                  kernel_size, kernel_size,
                  stride, stride, padding, use_bias)
{
}

std::unique_ptr<Layer> Conv2DLayer::clone() const {
    auto copy = std::make_unique<Conv2DLayer>(
        in_channels_, out_channels_, kernel_h_, kernel_w_,
        stride_h_, stride_w_, padding_, use_bias_);
    copy->set_weights(weights_);
    if (use_bias_) {
        copy->set_bias(bias_);
    }
    return copy;
}

void Conv2DLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 3) {
        // Single sample: (C_in, H, W)
        if (input.shape()[0] != in_channels_) {
            throw std::invalid_argument(
                "Conv2DLayer: expected " + std::to_string(in_channels_) +
                " input channels, got " + std::to_string(input.shape()[0]));
        }

        const size_t H = input.shape()[1];
        const size_t W = input.shape()[2];

        size_t pad_h = 0, pad_w = 0;
        if (padding_ == ops::PaddingMode::SAME) {
            pad_h = ops::compute_same_padding(H, kernel_h_, stride_h_);
            pad_w = ops::compute_same_padding(W, kernel_w_, stride_w_);
        }

        const size_t out_H = ops::conv_output_size(H, kernel_h_, stride_h_, pad_h);
        const size_t out_W = ops::conv_output_size(W, kernel_w_, stride_w_, pad_w);

        // im2col
        ops::im2col(input, col_buf_, kernel_h_, kernel_w_,
                     stride_h_, stride_w_, pad_h, pad_w);

        // GEMM: weights_2d_ (C_out, C_in*kH*kW) @ col_buf_ (C_in*kH*kW, out_H*out_W)
        ops::matmul(weights_2d_, col_buf_, gemm_buf_);

        // Reshape result to (C_out, out_H, out_W)
        std::vector<size_t> out_shape = {out_channels_, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }
        std::memcpy(output.data(), gemm_buf_.data(),
                     out_channels_ * out_H * out_W * sizeof(float));

        // Add bias per-channel
        if (use_bias_) {
            for (size_t c = 0; c < out_channels_; ++c) {
                float b = bias_.data()[c];
                float* out_ch = output.data() + c * out_H * out_W;
                for (size_t i = 0; i < out_H * out_W; ++i) {
                    out_ch[i] += b;
                }
            }
        }

    } else if (input.ndim() == 4) {
        // Batched: (N, C_in, H, W)
        if (input.shape()[1] != in_channels_) {
            throw std::invalid_argument(
                "Conv2DLayer: expected " + std::to_string(in_channels_) +
                " input channels, got " + std::to_string(input.shape()[1]));
        }

        const size_t N = input.shape()[0];
        const size_t H = input.shape()[2];
        const size_t W = input.shape()[3];

        size_t pad_h = 0, pad_w = 0;
        if (padding_ == ops::PaddingMode::SAME) {
            pad_h = ops::compute_same_padding(H, kernel_h_, stride_h_);
            pad_w = ops::compute_same_padding(W, kernel_w_, stride_w_);
        }

        const size_t out_H = ops::conv_output_size(H, kernel_h_, stride_h_, pad_h);
        const size_t out_W = ops::conv_output_size(W, kernel_w_, stride_w_, pad_w);

        std::vector<size_t> out_shape = {N, out_channels_, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }

        const size_t in_sample_size = in_channels_ * H * W;
        const size_t out_sample_size = out_channels_ * out_H * out_W;

        for (size_t n = 0; n < N; ++n) {
            // Create 3D view for this sample
            Tensor sample({in_channels_, H, W});
            std::memcpy(sample.data(), input.data() + n * in_sample_size,
                        in_sample_size * sizeof(float));

            // im2col + GEMM
            ops::im2col(sample, col_buf_, kernel_h_, kernel_w_,
                         stride_h_, stride_w_, pad_h, pad_w);
            ops::matmul(weights_2d_, col_buf_, gemm_buf_);

            // Copy to output
            std::memcpy(output.data() + n * out_sample_size,
                        gemm_buf_.data(),
                        out_sample_size * sizeof(float));

            // Add bias
            if (use_bias_) {
                float* out_n = output.data() + n * out_sample_size;
                for (size_t c = 0; c < out_channels_; ++c) {
                    float b = bias_.data()[c];
                    float* out_ch = out_n + c * out_H * out_W;
                    for (size_t i = 0; i < out_H * out_W; ++i) {
                        out_ch[i] += b;
                    }
                }
            }
        }

    } else {
        throw std::invalid_argument(
            "Conv2DLayer: expected 3D or 4D input, got " +
            std::to_string(input.ndim()) + "D");
    }
}

std::string Conv2DLayer::name() const {
    return "Conv2D(" + std::to_string(in_channels_) + ", " +
           std::to_string(out_channels_) + ", " +
           std::to_string(kernel_h_) + "x" + std::to_string(kernel_w_) + ")";
}

size_t Conv2DLayer::parameter_count() const {
    size_t count = out_channels_ * in_channels_ * kernel_h_ * kernel_w_;
    if (use_bias_) {
        count += out_channels_;
    }
    return count;
}

std::vector<size_t> Conv2DLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 3) {
        size_t H = input_shape[1];
        size_t W = input_shape[2];
        size_t pad_h = 0, pad_w = 0;
        if (padding_ == ops::PaddingMode::SAME) {
            pad_h = ops::compute_same_padding(H, kernel_h_, stride_h_);
            pad_w = ops::compute_same_padding(W, kernel_w_, stride_w_);
        }
        return {out_channels_,
                ops::conv_output_size(H, kernel_h_, stride_h_, pad_h),
                ops::conv_output_size(W, kernel_w_, stride_w_, pad_w)};
    } else if (input_shape.size() == 4) {
        size_t H = input_shape[2];
        size_t W = input_shape[3];
        size_t pad_h = 0, pad_w = 0;
        if (padding_ == ops::PaddingMode::SAME) {
            pad_h = ops::compute_same_padding(H, kernel_h_, stride_h_);
            pad_w = ops::compute_same_padding(W, kernel_w_, stride_w_);
        }
        return {input_shape[0], out_channels_,
                ops::conv_output_size(H, kernel_h_, stride_h_, pad_h),
                ops::conv_output_size(W, kernel_w_, stride_w_, pad_w)};
    }
    throw std::invalid_argument(
        "Conv2DLayer::output_shape: expected 3D or 4D, got " +
        std::to_string(input_shape.size()) + "D");
}

void Conv2DLayer::set_weights(const Tensor& weights) {
    std::vector<size_t> expected = {out_channels_, in_channels_, kernel_h_, kernel_w_};
    if (weights.shape() != expected) {
        throw std::invalid_argument("Conv2DLayer::set_weights: shape mismatch");
    }
    weights_ = weights;
    // Update cached 2D view
    size_t fan_in = in_channels_ * kernel_h_ * kernel_w_;
    weights_2d_ = Tensor({out_channels_, fan_in});
    std::memcpy(weights_2d_.data(), weights_.data(),
                out_channels_ * fan_in * sizeof(float));
}

void Conv2DLayer::set_bias(const Tensor& bias) {
    if (!use_bias_) {
        throw std::invalid_argument(
            "Conv2DLayer::set_bias: bias is disabled");
    }
    std::vector<size_t> expected = {out_channels_};
    if (bias.shape() != expected) {
        throw std::invalid_argument("Conv2DLayer::set_bias: shape mismatch");
    }
    bias_ = bias;
}

} // namespace layers
} // namespace titaninfer
