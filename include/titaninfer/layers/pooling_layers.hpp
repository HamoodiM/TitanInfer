#pragma once

#include "titaninfer/layers/layer.hpp"

namespace titaninfer {
namespace layers {

/**
 * @brief 2D Max Pooling layer
 *
 * Input: 3D (C, H, W) or 4D (N, C, H, W)
 * Slides a kernel_size x kernel_size window, takes maximum in each window.
 */
class MaxPool2DLayer : public Layer {
public:
    /**
     * @param kernel_size Pooling window size
     * @param stride Stride (0 = kernel_size, the standard default)
     * @param padding Zero-padding applied to input
     */
    MaxPool2DLayer(size_t kernel_size, size_t stride = 0,
                   size_t padding = 0);

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    size_t kernel_size() const { return kernel_size_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }

private:
    void pool_single(const float* in_data, float* out_data,
                     size_t C, size_t H, size_t W,
                     size_t out_H, size_t out_W) const;

    size_t kernel_size_, stride_, padding_;
};

/**
 * @brief 2D Average Pooling layer
 */
class AvgPool2DLayer : public Layer {
public:
    AvgPool2DLayer(size_t kernel_size, size_t stride = 0,
                   size_t padding = 0);

    std::unique_ptr<Layer> clone() const override;
    void forward(const Tensor& input, Tensor& output) override;
    std::string name() const override;
    std::vector<size_t> output_shape(
        const std::vector<size_t>& input_shape) const override;

    size_t kernel_size() const { return kernel_size_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }

private:
    void pool_single(const float* in_data, float* out_data,
                     size_t C, size_t H, size_t W,
                     size_t out_H, size_t out_W) const;

    size_t kernel_size_, stride_, padding_;
};

} // namespace layers
} // namespace titaninfer
