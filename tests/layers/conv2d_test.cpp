#include <gtest/gtest.h>
#include "titaninfer/layers/conv2d_layer.hpp"
#include "titaninfer/ops/conv_ops.hpp"

#include <cmath>

using namespace titaninfer;
using namespace titaninfer::layers;

TEST(ConvOpsTest, ConvOutputSizeValid) {
    // 5x5 input, 3x3 kernel, stride 1, no padding
    EXPECT_EQ(ops::conv_output_size(5, 3, 1, 0), 3u);
    // 7x7 input, 3x3 kernel, stride 2, no padding
    EXPECT_EQ(ops::conv_output_size(7, 3, 2, 0), 3u);
    // 28x28 input, 5x5 kernel, stride 1, no padding
    EXPECT_EQ(ops::conv_output_size(28, 5, 1, 0), 24u);
}

TEST(ConvOpsTest, ConvOutputSizeSame) {
    // SAME padding: output = ceil(input / stride)
    size_t pad = ops::compute_same_padding(5, 3, 1);
    EXPECT_EQ(ops::conv_output_size(5, 3, 1, pad), 5u);

    pad = ops::compute_same_padding(28, 3, 1);
    EXPECT_EQ(ops::conv_output_size(28, 3, 1, pad), 28u);
}

TEST(ConvOpsTest, Im2colBasic) {
    // 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding
    Tensor input({1, 3, 3});
    for (size_t i = 0; i < 9; ++i) {
        input.data()[i] = static_cast<float>(i + 1);
    }
    // Input: [[1,2,3],[4,5,6],[7,8,9]]

    Tensor col({1});
    ops::im2col(input, col, 2, 2, 1, 1, 0, 0);

    // Output shape: (1*2*2, 2*2) = (4, 4)
    EXPECT_EQ(col.shape()[0], 4u);
    EXPECT_EQ(col.shape()[1], 4u);

    // Check first column (top-left 2x2 patch): [1,2,4,5]
    EXPECT_FLOAT_EQ(col.data()[0 * 4 + 0], 1.0f);  // c=0,kh=0,kw=0 at (0,0)
    EXPECT_FLOAT_EQ(col.data()[1 * 4 + 0], 2.0f);  // c=0,kh=0,kw=1 at (0,0)
    EXPECT_FLOAT_EQ(col.data()[2 * 4 + 0], 4.0f);  // c=0,kh=1,kw=0 at (0,0)
    EXPECT_FLOAT_EQ(col.data()[3 * 4 + 0], 5.0f);  // c=0,kh=1,kw=1 at (0,0)
}

TEST(Conv2DTest, Construction) {
    Conv2DLayer conv(3, 16, 3, 1, ops::PaddingMode::VALID, true);
    EXPECT_EQ(conv.in_channels(), 3u);
    EXPECT_EQ(conv.out_channels(), 16u);
    EXPECT_EQ(conv.kernel_h(), 3u);
    EXPECT_EQ(conv.kernel_w(), 3u);
    EXPECT_EQ(conv.stride_h(), 1u);
    EXPECT_EQ(conv.stride_w(), 1u);
    EXPECT_TRUE(conv.has_bias());
    EXPECT_EQ(conv.padding(), ops::PaddingMode::VALID);
}

TEST(Conv2DTest, ParameterCount) {
    Conv2DLayer conv(3, 16, 3, 1, ops::PaddingMode::VALID, true);
    EXPECT_EQ(conv.parameter_count(), 3u * 16 * 3 * 3 + 16);

    Conv2DLayer conv_no_bias(3, 16, 3, 1, ops::PaddingMode::VALID, false);
    EXPECT_EQ(conv_no_bias.parameter_count(), 3u * 16 * 3 * 3);
}

TEST(Conv2DTest, OutputShapeValid) {
    Conv2DLayer conv(1, 4, 3, 1, ops::PaddingMode::VALID);
    auto shape = conv.output_shape({1, 5, 5});
    EXPECT_EQ(shape[0], 4u);  // out_channels
    EXPECT_EQ(shape[1], 3u);  // 5 - 3 + 1
    EXPECT_EQ(shape[2], 3u);
}

TEST(Conv2DTest, OutputShapeSame) {
    Conv2DLayer conv(1, 4, 3, 1, ops::PaddingMode::SAME);
    auto shape = conv.output_shape({1, 5, 5});
    EXPECT_EQ(shape[0], 4u);
    EXPECT_EQ(shape[1], 5u);  // SAME preserves spatial dims
    EXPECT_EQ(shape[2], 5u);
}

TEST(Conv2DTest, OutputShapeBatched) {
    Conv2DLayer conv(3, 8, 3, 1, ops::PaddingMode::VALID);
    auto shape = conv.output_shape({2, 3, 10, 10});
    EXPECT_EQ(shape[0], 2u);  // batch
    EXPECT_EQ(shape[1], 8u);  // out_channels
    EXPECT_EQ(shape[2], 8u);  // 10 - 3 + 1
    EXPECT_EQ(shape[3], 8u);
}

TEST(Conv2DTest, Forward1x1Kernel) {
    // 1x1 convolution is equivalent to per-pixel FC
    Conv2DLayer conv(2, 1, 1, 1, ops::PaddingMode::VALID, false);

    // Set weights: shape (1, 2, 1, 1) = [1.0, 2.0]
    Tensor w({1, 2, 1, 1});
    w.data()[0] = 1.0f;
    w.data()[1] = 2.0f;
    conv.set_weights(w);

    // Input: (2, 2, 2) = 2 channels, 2x2
    Tensor input({2, 2, 2});
    // Channel 0: [[1,2],[3,4]]
    input.data()[0] = 1.0f; input.data()[1] = 2.0f;
    input.data()[2] = 3.0f; input.data()[3] = 4.0f;
    // Channel 1: [[5,6],[7,8]]
    input.data()[4] = 5.0f; input.data()[5] = 6.0f;
    input.data()[6] = 7.0f; input.data()[7] = 8.0f;

    Tensor output({1});
    conv.forward(input, output);

    // Output: 1*ch0 + 2*ch1 at each position
    EXPECT_EQ(output.shape().size(), 3u);
    EXPECT_EQ(output.shape()[0], 1u);
    EXPECT_EQ(output.shape()[1], 2u);
    EXPECT_EQ(output.shape()[2], 2u);

    EXPECT_FLOAT_EQ(output.data()[0], 1.0f * 1.0f + 2.0f * 5.0f);  // 11
    EXPECT_FLOAT_EQ(output.data()[1], 1.0f * 2.0f + 2.0f * 6.0f);  // 14
    EXPECT_FLOAT_EQ(output.data()[2], 1.0f * 3.0f + 2.0f * 7.0f);  // 17
    EXPECT_FLOAT_EQ(output.data()[3], 1.0f * 4.0f + 2.0f * 8.0f);  // 20
}

TEST(Conv2DTest, ForwardWithBias) {
    Conv2DLayer conv(1, 1, 1, 1, ops::PaddingMode::VALID, true);

    Tensor w({1, 1, 1, 1});
    w.data()[0] = 2.0f;
    conv.set_weights(w);

    Tensor b({1});
    b.data()[0] = 0.5f;
    conv.set_bias(b);

    Tensor input({1, 3, 3});
    input.fill(1.0f);

    Tensor output({1});
    conv.forward(input, output);

    // Each output: 2.0 * 1.0 + 0.5 = 2.5
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], 2.5f);
    }
}

TEST(Conv2DTest, ForwardStride2) {
    Conv2DLayer conv(1, 1, 2, 2, ops::PaddingMode::VALID, false);

    Tensor w({1, 1, 2, 2});
    w.fill(1.0f);
    conv.set_weights(w);

    Tensor input({1, 4, 4});
    input.fill(1.0f);

    Tensor output({1});
    conv.forward(input, output);

    // 4x4 input, 2x2 kernel, stride 2 -> 2x2 output
    EXPECT_EQ(output.shape()[1], 2u);
    EXPECT_EQ(output.shape()[2], 2u);
    // Each output: sum of 2x2 ones = 4.0
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], 4.0f);
    }
}

TEST(Conv2DTest, ForwardBatched) {
    Conv2DLayer conv(1, 1, 1, 1, ops::PaddingMode::VALID, false);

    Tensor w({1, 1, 1, 1});
    w.data()[0] = 3.0f;
    conv.set_weights(w);

    Tensor input({2, 1, 2, 2});
    input.fill(1.0f);

    Tensor output({1});
    conv.forward(input, output);

    EXPECT_EQ(output.shape()[0], 2u);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], 3.0f);
    }
}

TEST(Conv2DTest, SetWeightsShapeMismatch) {
    Conv2DLayer conv(3, 16, 3);
    Tensor bad_w({16, 3, 5, 5}); // wrong kernel size
    EXPECT_THROW(conv.set_weights(bad_w), std::invalid_argument);
}

TEST(Conv2DTest, InvalidInputDimension) {
    Conv2DLayer conv(3, 16, 3);
    Tensor input_1d({10});
    Tensor output({1});
    EXPECT_THROW(conv.forward(input_1d, output), std::invalid_argument);
}

TEST(Conv2DTest, Name) {
    Conv2DLayer conv(3, 16, 3, 1, ops::PaddingMode::VALID);
    EXPECT_EQ(conv.name(), "Conv2D(3, 16, 3x3)");
}

TEST(Conv2DTest, Clone) {
    Conv2DLayer conv(1, 1, 3, 1, ops::PaddingMode::SAME, true);
    Tensor w({1, 1, 3, 3});
    w.fill(1.0f);
    conv.set_weights(w);

    auto cloned = conv.clone();
    auto* conv_clone = dynamic_cast<Conv2DLayer*>(cloned.get());
    ASSERT_NE(conv_clone, nullptr);
    EXPECT_EQ(conv_clone->in_channels(), 1u);
    EXPECT_EQ(conv_clone->out_channels(), 1u);
    EXPECT_EQ(conv_clone->kernel_h(), 3u);
    EXPECT_EQ(conv_clone->padding(), ops::PaddingMode::SAME);
}
