#include "titaninfer/layers/pooling_layers.hpp"
#include "titaninfer/ops/conv_ops.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace titaninfer {
namespace layers {

// ========================================
// MaxPool2DLayer
// ========================================

MaxPool2DLayer::MaxPool2DLayer(size_t kernel_size, size_t stride, size_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
    if (kernel_size == 0) {
        throw std::invalid_argument("MaxPool2DLayer: kernel_size must be > 0");
    }
}

std::unique_ptr<Layer> MaxPool2DLayer::clone() const {
    return std::make_unique<MaxPool2DLayer>(kernel_size_, stride_, padding_);
}

void MaxPool2DLayer::pool_single(const float* in_data, float* out_data,
                                  size_t C, size_t H, size_t W,
                                  size_t out_H, size_t out_W) const {
    for (size_t c = 0; c < C; ++c) {
        const float* ch_in = in_data + c * H * W;
        float* ch_out = out_data + c * out_H * out_W;

        for (size_t oh = 0; oh < out_H; ++oh) {
            for (size_t ow = 0; ow < out_W; ++ow) {
                float max_val = -std::numeric_limits<float>::infinity();

                for (size_t kh = 0; kh < kernel_size_; ++kh) {
                    for (size_t kw = 0; kw < kernel_size_; ++kw) {
                        size_t ih = oh * stride_ + kh;
                        size_t iw = ow * stride_ + kw;

                        // Handle padding
                        if (ih >= padding_ && iw >= padding_ &&
                            (ih - padding_) < H && (iw - padding_) < W) {
                            float val = ch_in[(ih - padding_) * W + (iw - padding_)];
                            max_val = std::max(max_val, val);
                        }
                        // Padded positions are -inf (not included in max)
                    }
                }

                ch_out[oh * out_W + ow] = max_val;
            }
        }
    }
}

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 3) {
        const size_t C = input.shape()[0];
        const size_t H = input.shape()[1];
        const size_t W = input.shape()[2];
        const size_t out_H = ops::conv_output_size(H, kernel_size_, stride_, padding_);
        const size_t out_W = ops::conv_output_size(W, kernel_size_, stride_, padding_);

        std::vector<size_t> out_shape = {C, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }

        pool_single(input.data(), output.data(), C, H, W, out_H, out_W);

    } else if (input.ndim() == 4) {
        const size_t N = input.shape()[0];
        const size_t C = input.shape()[1];
        const size_t H = input.shape()[2];
        const size_t W = input.shape()[3];
        const size_t out_H = ops::conv_output_size(H, kernel_size_, stride_, padding_);
        const size_t out_W = ops::conv_output_size(W, kernel_size_, stride_, padding_);

        std::vector<size_t> out_shape = {N, C, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }

        const size_t in_sample = C * H * W;
        const size_t out_sample = C * out_H * out_W;

        for (size_t n = 0; n < N; ++n) {
            pool_single(input.data() + n * in_sample,
                        output.data() + n * out_sample,
                        C, H, W, out_H, out_W);
        }

    } else {
        throw std::invalid_argument(
            "MaxPool2DLayer: expected 3D or 4D input, got " +
            std::to_string(input.ndim()) + "D");
    }
}

std::string MaxPool2DLayer::name() const {
    return "MaxPool2D(" + std::to_string(kernel_size_) + ")";
}

std::vector<size_t> MaxPool2DLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 3) {
        size_t H = input_shape[1], W = input_shape[2];
        return {input_shape[0],
                ops::conv_output_size(H, kernel_size_, stride_, padding_),
                ops::conv_output_size(W, kernel_size_, stride_, padding_)};
    } else if (input_shape.size() == 4) {
        size_t H = input_shape[2], W = input_shape[3];
        return {input_shape[0], input_shape[1],
                ops::conv_output_size(H, kernel_size_, stride_, padding_),
                ops::conv_output_size(W, kernel_size_, stride_, padding_)};
    }
    throw std::invalid_argument(
        "MaxPool2DLayer::output_shape: expected 3D or 4D");
}

// ========================================
// AvgPool2DLayer
// ========================================

AvgPool2DLayer::AvgPool2DLayer(size_t kernel_size, size_t stride, size_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
    if (kernel_size == 0) {
        throw std::invalid_argument("AvgPool2DLayer: kernel_size must be > 0");
    }
}

std::unique_ptr<Layer> AvgPool2DLayer::clone() const {
    return std::make_unique<AvgPool2DLayer>(kernel_size_, stride_, padding_);
}

void AvgPool2DLayer::pool_single(const float* in_data, float* out_data,
                                  size_t C, size_t H, size_t W,
                                  size_t out_H, size_t out_W) const {
    const float inv_area = 1.0f / static_cast<float>(kernel_size_ * kernel_size_);

    for (size_t c = 0; c < C; ++c) {
        const float* ch_in = in_data + c * H * W;
        float* ch_out = out_data + c * out_H * out_W;

        for (size_t oh = 0; oh < out_H; ++oh) {
            for (size_t ow = 0; ow < out_W; ++ow) {
                float sum = 0.0f;

                for (size_t kh = 0; kh < kernel_size_; ++kh) {
                    for (size_t kw = 0; kw < kernel_size_; ++kw) {
                        size_t ih = oh * stride_ + kh;
                        size_t iw = ow * stride_ + kw;

                        if (ih >= padding_ && iw >= padding_ &&
                            (ih - padding_) < H && (iw - padding_) < W) {
                            sum += ch_in[(ih - padding_) * W + (iw - padding_)];
                        }
                        // Padded positions contribute 0
                    }
                }

                ch_out[oh * out_W + ow] = sum * inv_area;
            }
        }
    }
}

void AvgPool2DLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() == 3) {
        const size_t C = input.shape()[0];
        const size_t H = input.shape()[1];
        const size_t W = input.shape()[2];
        const size_t out_H = ops::conv_output_size(H, kernel_size_, stride_, padding_);
        const size_t out_W = ops::conv_output_size(W, kernel_size_, stride_, padding_);

        std::vector<size_t> out_shape = {C, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }

        pool_single(input.data(), output.data(), C, H, W, out_H, out_W);

    } else if (input.ndim() == 4) {
        const size_t N = input.shape()[0];
        const size_t C = input.shape()[1];
        const size_t H = input.shape()[2];
        const size_t W = input.shape()[3];
        const size_t out_H = ops::conv_output_size(H, kernel_size_, stride_, padding_);
        const size_t out_W = ops::conv_output_size(W, kernel_size_, stride_, padding_);

        std::vector<size_t> out_shape = {N, C, out_H, out_W};
        if (output.shape() != out_shape) {
            output = Tensor(out_shape);
        }

        const size_t in_sample = C * H * W;
        const size_t out_sample = C * out_H * out_W;

        for (size_t n = 0; n < N; ++n) {
            pool_single(input.data() + n * in_sample,
                        output.data() + n * out_sample,
                        C, H, W, out_H, out_W);
        }

    } else {
        throw std::invalid_argument(
            "AvgPool2DLayer: expected 3D or 4D input, got " +
            std::to_string(input.ndim()) + "D");
    }
}

std::string AvgPool2DLayer::name() const {
    return "AvgPool2D(" + std::to_string(kernel_size_) + ")";
}

std::vector<size_t> AvgPool2DLayer::output_shape(
    const std::vector<size_t>& input_shape) const {
    if (input_shape.size() == 3) {
        size_t H = input_shape[1], W = input_shape[2];
        return {input_shape[0],
                ops::conv_output_size(H, kernel_size_, stride_, padding_),
                ops::conv_output_size(W, kernel_size_, stride_, padding_)};
    } else if (input_shape.size() == 4) {
        size_t H = input_shape[2], W = input_shape[3];
        return {input_shape[0], input_shape[1],
                ops::conv_output_size(H, kernel_size_, stride_, padding_),
                ops::conv_output_size(W, kernel_size_, stride_, padding_)};
    }
    throw std::invalid_argument(
        "AvgPool2DLayer::output_shape: expected 3D or 4D");
}

} // namespace layers
} // namespace titaninfer
