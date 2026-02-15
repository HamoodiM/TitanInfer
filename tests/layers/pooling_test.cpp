#include <gtest/gtest.h>
#include "titaninfer/layers/pooling_layers.hpp"

using namespace titaninfer;
using namespace titaninfer::layers;

TEST(MaxPool2DTest, Construction) {
    MaxPool2DLayer pool(2);
    EXPECT_EQ(pool.kernel_size(), 2u);
    EXPECT_EQ(pool.stride(), 2u);  // defaults to kernel_size
    EXPECT_EQ(pool.padding(), 0u);
    EXPECT_EQ(pool.parameter_count(), 0u);
}

TEST(MaxPool2DTest, OutputShape) {
    MaxPool2DLayer pool(2, 2);
    auto shape = pool.output_shape({1, 4, 4});
    EXPECT_EQ(shape[0], 1u);
    EXPECT_EQ(shape[1], 2u);
    EXPECT_EQ(shape[2], 2u);
}

TEST(MaxPool2DTest, Forward2x2Stride2) {
    MaxPool2DLayer pool(2, 2);

    // 1 channel, 4x4 input
    Tensor input({1, 4, 4});
    float vals[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    for (size_t i = 0; i < 16; ++i) {
        input.data()[i] = vals[i];
    }

    Tensor output({1});
    pool.forward(input, output);

    EXPECT_EQ(output.shape()[0], 1u);
    EXPECT_EQ(output.shape()[1], 2u);
    EXPECT_EQ(output.shape()[2], 2u);

    EXPECT_FLOAT_EQ(output.data()[0], 6.0f);   // max(1,2,5,6)
    EXPECT_FLOAT_EQ(output.data()[1], 8.0f);   // max(3,4,7,8)
    EXPECT_FLOAT_EQ(output.data()[2], 14.0f);  // max(9,10,13,14)
    EXPECT_FLOAT_EQ(output.data()[3], 16.0f);  // max(11,12,15,16)
}

TEST(MaxPool2DTest, Forward3x3Stride1) {
    MaxPool2DLayer pool(3, 1);

    Tensor input({1, 4, 4});
    for (size_t i = 0; i < 16; ++i) {
        input.data()[i] = static_cast<float>(i);
    }

    Tensor output({1});
    pool.forward(input, output);

    // 4x4, kernel 3, stride 1 -> 2x2 output
    EXPECT_EQ(output.shape()[1], 2u);
    EXPECT_EQ(output.shape()[2], 2u);

    // Top-left 3x3: max(0..10) = 10
    EXPECT_FLOAT_EQ(output.data()[0], 10.0f);
    // Top-right 3x3: max(1..11) = 11
    EXPECT_FLOAT_EQ(output.data()[1], 11.0f);
}

TEST(MaxPool2DTest, ForwardBatched) {
    MaxPool2DLayer pool(2, 2);

    Tensor input({2, 1, 2, 2});
    // Sample 0: [1,2,3,4]
    input.data()[0] = 1; input.data()[1] = 2;
    input.data()[2] = 3; input.data()[3] = 4;
    // Sample 1: [5,6,7,8]
    input.data()[4] = 5; input.data()[5] = 6;
    input.data()[6] = 7; input.data()[7] = 8;

    Tensor output({1});
    pool.forward(input, output);

    EXPECT_EQ(output.shape()[0], 2u);
    EXPECT_FLOAT_EQ(output.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 8.0f);
}

TEST(MaxPool2DTest, MultiChannel) {
    MaxPool2DLayer pool(2, 2);

    Tensor input({2, 2, 2});
    // Ch 0: [1,2,3,4], Ch 1: [5,6,7,8]
    float vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < 8; ++i) input.data()[i] = vals[i];

    Tensor output({1});
    pool.forward(input, output);

    EXPECT_FLOAT_EQ(output.data()[0], 4.0f);  // max of ch0
    EXPECT_FLOAT_EQ(output.data()[1], 8.0f);  // max of ch1
}

// ========================================
// AvgPool2D Tests
// ========================================

TEST(AvgPool2DTest, Construction) {
    AvgPool2DLayer pool(2);
    EXPECT_EQ(pool.kernel_size(), 2u);
    EXPECT_EQ(pool.stride(), 2u);
    EXPECT_EQ(pool.parameter_count(), 0u);
}

TEST(AvgPool2DTest, Forward2x2Stride2) {
    AvgPool2DLayer pool(2, 2);

    Tensor input({1, 4, 4});
    float vals[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    for (size_t i = 0; i < 16; ++i) {
        input.data()[i] = vals[i];
    }

    Tensor output({1});
    pool.forward(input, output);

    EXPECT_FLOAT_EQ(output.data()[0], (1+2+5+6)/4.0f);    // 3.5
    EXPECT_FLOAT_EQ(output.data()[1], (3+4+7+8)/4.0f);    // 5.5
    EXPECT_FLOAT_EQ(output.data()[2], (9+10+13+14)/4.0f);  // 11.5
    EXPECT_FLOAT_EQ(output.data()[3], (11+12+15+16)/4.0f); // 13.5
}

TEST(AvgPool2DTest, ForwardBatched) {
    AvgPool2DLayer pool(2, 2);

    Tensor input({2, 1, 2, 2});
    input.fill(4.0f);

    Tensor output({1});
    pool.forward(input, output);

    EXPECT_EQ(output.shape()[0], 2u);
    EXPECT_FLOAT_EQ(output.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 4.0f);
}

TEST(AvgPool2DTest, Name) {
    AvgPool2DLayer pool(3);
    EXPECT_EQ(pool.name(), "AvgPool2D(3)");
}
