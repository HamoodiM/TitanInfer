#include <gtest/gtest.h>
#include "titaninfer/layers/flatten_layer.hpp"

using namespace titaninfer;
using namespace titaninfer::layers;

TEST(FlattenTest, OutputShape3D) {
    FlattenLayer flatten;
    auto shape = flatten.output_shape({3, 4, 5});
    ASSERT_EQ(shape.size(), 1u);
    EXPECT_EQ(shape[0], 60u);
}

TEST(FlattenTest, OutputShape4D) {
    FlattenLayer flatten;
    auto shape = flatten.output_shape({2, 3, 4, 5});
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 2u);
    EXPECT_EQ(shape[1], 60u);
}

TEST(FlattenTest, OutputShape2DPassthrough) {
    FlattenLayer flatten;
    auto shape = flatten.output_shape({4, 10});
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4u);
    EXPECT_EQ(shape[1], 10u);
}

TEST(FlattenTest, ForwardDataPreservation) {
    FlattenLayer flatten;

    Tensor input({2, 3, 4});
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(i);
    }

    Tensor output({1});
    flatten.forward(input, output);

    ASSERT_EQ(output.shape().size(), 1u);
    EXPECT_EQ(output.shape()[0], 24u);

    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], static_cast<float>(i));
    }
}

TEST(FlattenTest, Forward4D) {
    FlattenLayer flatten;

    Tensor input({2, 3, 2, 2});
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(i);
    }

    Tensor output({1});
    flatten.forward(input, output);

    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 2u);
    EXPECT_EQ(output.shape()[1], 12u);

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], static_cast<float>(i));
    }
}

TEST(FlattenTest, ParameterCount) {
    FlattenLayer flatten;
    EXPECT_EQ(flatten.parameter_count(), 0u);
}

TEST(FlattenTest, Name) {
    FlattenLayer flatten;
    EXPECT_EQ(flatten.name(), "Flatten");
}

TEST(FlattenTest, Clone) {
    FlattenLayer flatten;
    auto cloned = flatten.clone();
    EXPECT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->name(), "Flatten");
}
