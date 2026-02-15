#pragma once

#include "titaninfer/layers/layer.hpp"
#include "titaninfer/ops/conv_ops.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief 2D Convolution layer using im2col + GEMM
 *
 * Input: 3D (C_in, H, W) or 4D (N, C_in, H, W)
 * Weights: (C_out, C_in, kH, kW)
 * Output: 3D (C_out, out_H, out_W) or 4D (N, C_out, out_H, out_W)
 */
class Conv2DLayer : public Layer {
public:
    Conv2DLayer(size_t in_channels, size_t out_channels,
                size_t kernel_h, size_t kernel_w,
                size_t stride_h = 1, size_t stride_w = 1,
                ops::PaddingMode padding = ops::PaddingMode::VALID,
                bool use_bias = true);

    /// Convenience: square kernel and stride
    Conv2DLayer(size_t in_channels, size_t out_channels,
                size_t kernel_size, size_t stride = 1,
                ops::PaddingMode padding = ops::PaddingMode::VALID,
                bool use_bias = true);

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    size_t parameter_count() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    void set_weights(const Tensor& weights);
    void set_bias(const Tensor& bias);

    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }
    size_t kernel_h() const { return kernel_h_; }
    size_t kernel_w() const { return kernel_w_; }
    size_t stride_h() const { return stride_h_; }
    size_t stride_w() const { return stride_w_; }
    ops::PaddingMode padding() const { return padding_; }
    bool has_bias() const { return use_bias_; }

private:
    void forward_single(const float* input_data, size_t H, size_t W,
                        float* output_data, size_t out_H, size_t out_W);

    size_t in_channels_, out_channels_;
    size_t kernel_h_, kernel_w_;
    size_t stride_h_, stride_w_;
    ops::PaddingMode padding_;
    bool use_bias_;

    Tensor weights_;     // (C_out, C_in, kH, kW)
    Tensor bias_;        // (C_out,)
    Tensor col_buf_;     // im2col buffer, lazily allocated
    Tensor weights_2d_;  // (C_out, C_in*kH*kW), cached reshape
    Tensor gemm_buf_;    // matmul result buffer
};

} // namespace layers
} // namespace titaninfer
