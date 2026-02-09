#include <gtest/gtest.h>
#include "titaninfer/ops/activations.hpp"
#include <cmath>
#include <numeric>

using namespace titaninfer;
using namespace titaninfer::ops;

// ========================================
// ReLU Tests
// ========================================

TEST(ActivationsTest, ReluBasic) {
    Tensor input({5});
    input.data()[0] = -2.0f;
    input.data()[1] = -0.5f;
    input.data()[2] =  0.0f;
    input.data()[3] =  0.5f;
    input.data()[4] =  2.0f;

    Tensor output({1}); // wrong shape, should auto-allocate
    relu(input, output);

    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 0.5f);
    EXPECT_FLOAT_EQ(output.data()[4], 2.0f);
}

TEST(ActivationsTest, ReluInplace) {
    Tensor t({4});
    t.data()[0] = -3.0f;
    t.data()[1] =  0.0f;
    t.data()[2] =  1.0f;
    t.data()[3] = -0.1f;

    relu_inplace(t);

    EXPECT_FLOAT_EQ(t.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 0.0f);
}

TEST(ActivationsTest, Relu2D) {
    Tensor input({2, 3});
    input.data()[0] = -1.0f; input.data()[1] = 2.0f; input.data()[2] = -3.0f;
    input.data()[3] =  4.0f; input.data()[4] = -5.0f; input.data()[5] = 6.0f;

    Tensor output({2, 3});
    relu(input, output);

    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 4.0f);
    EXPECT_FLOAT_EQ(output.data()[4], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[5], 6.0f);
}

// ========================================
// Sigmoid Tests
// ========================================

TEST(ActivationsTest, SigmoidBasic) {
    Tensor input({3});
    input.data()[0] = 0.0f;
    input.data()[1] = 1.0f;
    input.data()[2] = -1.0f;

    Tensor output({3});
    sigmoid(input, output);

    EXPECT_NEAR(output.data()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(output.data()[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
    EXPECT_NEAR(output.data()[2], 1.0f / (1.0f + std::exp(1.0f)), 1e-6f);
}

TEST(ActivationsTest, SigmoidBounds) {
    Tensor input({4});
    input.data()[0] =  100.0f;
    input.data()[1] = -100.0f;
    input.data()[2] =  50.0f;
    input.data()[3] = -50.0f;

    Tensor output({4});
    sigmoid(input, output);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GE(output.data()[i], 0.0f);
        EXPECT_LE(output.data()[i], 1.0f);
    }
    // Large positive -> close to 1
    EXPECT_NEAR(output.data()[0], 1.0f, 1e-5f);
    // Large negative -> close to 0
    EXPECT_NEAR(output.data()[1], 0.0f, 1e-5f);
}

TEST(ActivationsTest, SigmoidInplace) {
    Tensor t({2});
    t.data()[0] = 0.0f;
    t.data()[1] = 2.0f;

    sigmoid_inplace(t);

    EXPECT_NEAR(t.data()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(t.data()[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
}

// ========================================
// Tanh Tests
// ========================================

TEST(ActivationsTest, TanhBasic) {
    Tensor input({3});
    input.data()[0] = 0.0f;
    input.data()[1] = 1.0f;
    input.data()[2] = -1.0f;

    Tensor output({3});
    tanh_activation(input, output);

    EXPECT_NEAR(output.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(output.data()[1], std::tanh(1.0f), 1e-6f);
    EXPECT_NEAR(output.data()[2], std::tanh(-1.0f), 1e-6f);
}

TEST(ActivationsTest, TanhBounds) {
    Tensor input({4});
    input.data()[0] =  100.0f;
    input.data()[1] = -100.0f;
    input.data()[2] =  50.0f;
    input.data()[3] = -50.0f;

    Tensor output({4});
    tanh_activation(input, output);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GE(output.data()[i], -1.0f);
        EXPECT_LE(output.data()[i], 1.0f);
    }
    EXPECT_NEAR(output.data()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(output.data()[1], -1.0f, 1e-5f);
}

TEST(ActivationsTest, TanhInplace) {
    Tensor t({2});
    t.data()[0] = 0.0f;
    t.data()[1] = 0.5f;

    tanh_inplace(t);

    EXPECT_NEAR(t.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(t.data()[1], std::tanh(0.5f), 1e-6f);
}

// ========================================
// Softmax Tests
// ========================================

TEST(ActivationsTest, SoftmaxBasic1D) {
    Tensor input({3});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;

    Tensor output({3});
    softmax(input, output);

    // All outputs should be positive
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_GT(output.data()[i], 0.0f);
    }

    // Should sum to 1
    float sum = output.data()[0] + output.data()[1] + output.data()[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);

    // Monotonic: larger input -> larger output
    EXPECT_LT(output.data()[0], output.data()[1]);
    EXPECT_LT(output.data()[1], output.data()[2]);
}

TEST(ActivationsTest, SoftmaxUniform) {
    const size_t n = 4;
    Tensor input({n});
    input.fill(5.0f); // all same value

    Tensor output({n});
    softmax(input, output);

    // Uniform input -> uniform output = 1/N
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(output.data()[i], 1.0f / static_cast<float>(n), 1e-6f);
    }
}

TEST(ActivationsTest, SoftmaxNumericalStabilityLargePositive) {
    Tensor input({3});
    input.data()[0] = 1000.0f;
    input.data()[1] = 1000.0f;
    input.data()[2] = 1000.0f;

    Tensor output({3});
    softmax(input, output);

    // Should not produce NaN or Inf
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(output.data()[i]));
        EXPECT_FALSE(std::isinf(output.data()[i]));
    }

    float sum = output.data()[0] + output.data()[1] + output.data()[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(ActivationsTest, SoftmaxLargeNegative) {
    Tensor input({3});
    input.data()[0] = -1000.0f;
    input.data()[1] = -1000.0f;
    input.data()[2] = -1000.0f;

    Tensor output({3});
    softmax(input, output);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(output.data()[i]));
        EXPECT_FALSE(std::isinf(output.data()[i]));
        EXPECT_NEAR(output.data()[i], 1.0f / 3.0f, 1e-6f);
    }
}

TEST(ActivationsTest, SoftmaxMixedLargeValues) {
    Tensor input({3});
    input.data()[0] = 1000.0f;
    input.data()[1] = -1000.0f;
    input.data()[2] = 0.0f;

    Tensor output({3});
    softmax(input, output);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(output.data()[i]));
        EXPECT_FALSE(std::isinf(output.data()[i]));
    }

    // The largest input should dominate
    EXPECT_NEAR(output.data()[0], 1.0f, 1e-5f);

    float sum = output.data()[0] + output.data()[1] + output.data()[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(ActivationsTest, Softmax2DRowWise) {
    Tensor input({2, 3});
    // Row 0: [1, 2, 3]
    input.data()[0] = 1.0f; input.data()[1] = 2.0f; input.data()[2] = 3.0f;
    // Row 1: [1, 1, 1]
    input.data()[3] = 1.0f; input.data()[4] = 1.0f; input.data()[5] = 1.0f;

    Tensor output({2, 3});
    softmax(input, output);

    // Each row should sum to 1
    float row0_sum = output.data()[0] + output.data()[1] + output.data()[2];
    float row1_sum = output.data()[3] + output.data()[4] + output.data()[5];
    EXPECT_NEAR(row0_sum, 1.0f, 1e-6f);
    EXPECT_NEAR(row1_sum, 1.0f, 1e-6f);

    // Row 1 uniform -> 1/3 each
    EXPECT_NEAR(output.data()[3], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(output.data()[4], 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(output.data()[5], 1.0f / 3.0f, 1e-6f);
}

TEST(ActivationsTest, SoftmaxInvalid3D) {
    Tensor input({2, 3, 4});
    Tensor output({1});

    EXPECT_THROW(softmax(input, output), std::invalid_argument);
}

TEST(ActivationsTest, SoftmaxSingleElement) {
    Tensor input({1});
    input.data()[0] = 42.0f;

    Tensor output({1});
    softmax(input, output);

    // Single element softmax is always 1.0
    EXPECT_NEAR(output.data()[0], 1.0f, 1e-6f);
}

// ========================================
// Auto-allocation Test
// ========================================

TEST(ActivationsTest, OutputAutoAllocation) {
    Tensor input({4});
    input.data()[0] = -1.0f;
    input.data()[1] = 0.0f;
    input.data()[2] = 1.0f;
    input.data()[3] = 2.0f;

    // Output has wrong shape — should be reallocated
    Tensor output({2, 2});
    relu(input, output);

    ASSERT_EQ(output.shape(), input.shape());
    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 2.0f);
}
