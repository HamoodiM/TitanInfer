#include <gtest/gtest.h>
#include "titaninfer/quantized_tensor.hpp"
#include "titaninfer/ops/quantized_ops.hpp"
#include "titaninfer/layers/quantized_dense_layer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/ops/matrix_ops.hpp"

#include <cmath>

using namespace titaninfer;

// ========================================
// QuantizedTensor Tests
// ========================================

TEST(QuantizedTensorTest, Construction) {
    QuantizedTensor qt({4, 8});
    EXPECT_EQ(qt.shape()[0], 4u);
    EXPECT_EQ(qt.shape()[1], 8u);
    EXPECT_EQ(qt.size(), 32u);
    EXPECT_EQ(qt.ndim(), 2u);
}

TEST(QuantizedTensorTest, QuantizeDequantizeRoundTrip) {
    Tensor fp32({8});
    for (size_t i = 0; i < 8; ++i) {
        fp32.data()[i] = static_cast<float>(i) - 3.5f;
    }

    QuantizedTensor qt = QuantizedTensor::quantize(fp32);
    Tensor recovered = qt.dequantize();

    EXPECT_EQ(recovered.shape(), fp32.shape());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(recovered.data()[i], fp32.data()[i], 0.05f);
    }
}

TEST(QuantizedTensorTest, QuantizeUniform) {
    Tensor fp32({256});
    for (size_t i = 0; i < 256; ++i) {
        fp32.data()[i] = static_cast<float>(i) / 255.0f;  // [0, 1]
    }

    QuantizedTensor qt = QuantizedTensor::quantize(fp32);
    Tensor recovered = qt.dequantize();

    for (size_t i = 0; i < 256; ++i) {
        EXPECT_NEAR(recovered.data()[i], fp32.data()[i], 0.01f);
    }
}

TEST(QuantizedTensorTest, QuantizeAllSame) {
    Tensor fp32({10});
    fp32.fill(3.14f);

    QuantizedTensor qt = QuantizedTensor::quantize(fp32);
    Tensor recovered = qt.dequantize();

    for (size_t i = 0; i < 10; ++i) {
        // All values should be the same after dequantize
        EXPECT_FLOAT_EQ(recovered.data()[i], recovered.data()[0]);
    }
}

TEST(QuantizedTensorTest, CopySemantics) {
    Tensor fp32({4});
    fp32.data()[0] = -1.0f; fp32.data()[1] = 0.0f;
    fp32.data()[2] = 0.5f;  fp32.data()[3] = 1.0f;

    QuantizedTensor qt = QuantizedTensor::quantize(fp32);
    QuantizedTensor copy(qt);

    EXPECT_EQ(copy.shape(), qt.shape());
    EXPECT_EQ(copy.scale(), qt.scale());
    EXPECT_EQ(copy.zero_point(), qt.zero_point());
    for (size_t i = 0; i < qt.size(); ++i) {
        EXPECT_EQ(copy.data()[i], qt.data()[i]);
    }
}

TEST(QuantizedTensorTest, MoveSemantics) {
    Tensor fp32({4});
    fp32.fill(1.0f);

    QuantizedTensor qt = QuantizedTensor::quantize(fp32);
    size_t orig_size = qt.size();

    QuantizedTensor moved(std::move(qt));
    EXPECT_EQ(moved.size(), orig_size);
    EXPECT_EQ(qt.size(), 0u);
}

// ========================================
// INT8 GEMM Tests
// ========================================

TEST(GemmInt8Test, SmallMatrix) {
    // 2x3 * 3x2, compare against FP32 matmul
    Tensor A_fp({2, 3});
    A_fp.data()[0] = 1; A_fp.data()[1] = 2; A_fp.data()[2] = 3;
    A_fp.data()[3] = 4; A_fp.data()[4] = 5; A_fp.data()[5] = 6;

    Tensor B_fp({3, 2});
    B_fp.data()[0] = 7;  B_fp.data()[1] = 8;
    B_fp.data()[2] = 9;  B_fp.data()[3] = 10;
    B_fp.data()[4] = 11; B_fp.data()[5] = 12;

    // FP32 reference
    Tensor C_ref({1});
    ops::matmul(A_fp, B_fp, C_ref);

    // INT8
    QuantizedTensor A_q = QuantizedTensor::quantize(A_fp);
    QuantizedTensor B_q = QuantizedTensor::quantize(B_fp);
    Tensor C_int8({1});
    ops::gemm_int8(A_q, B_q, C_int8);

    EXPECT_EQ(C_int8.shape()[0], 2u);
    EXPECT_EQ(C_int8.shape()[1], 2u);

    for (size_t i = 0; i < C_ref.size(); ++i) {
        EXPECT_NEAR(C_int8.data()[i], C_ref.data()[i],
                     std::abs(C_ref.data()[i]) * 0.15f + 1.0f);
    }
}

TEST(GemmInt8Test, LargerMatrix) {
    Tensor A_fp({16, 16});
    Tensor B_fp({16, 16});

    for (size_t i = 0; i < 256; ++i) {
        A_fp.data()[i] = static_cast<float>(i % 10) - 5.0f;
        B_fp.data()[i] = static_cast<float>(i % 7) - 3.0f;
    }

    Tensor C_ref({1});
    ops::matmul(A_fp, B_fp, C_ref);

    QuantizedTensor A_q = QuantizedTensor::quantize(A_fp);
    QuantizedTensor B_q = QuantizedTensor::quantize(B_fp);
    Tensor C_int8({1});
    ops::gemm_int8(A_q, B_q, C_int8);

    for (size_t i = 0; i < C_ref.size(); ++i) {
        EXPECT_NEAR(C_int8.data()[i], C_ref.data()[i],
                     std::abs(C_ref.data()[i]) * 0.2f + 2.0f);
    }
}

TEST(GemmInt8Test, DimensionMismatch) {
    QuantizedTensor A({2, 3});
    QuantizedTensor B({4, 2});  // inner dim mismatch: 3 != 4
    Tensor C({1});
    EXPECT_THROW(ops::gemm_int8(A, B, C), std::invalid_argument);
}

// ========================================
// QuantizedDenseLayer Tests
// ========================================

TEST(QuantizedDenseTest, Construction) {
    layers::DenseLayer dense(4, 3, true);
    Tensor w({3, 4});
    for (size_t i = 0; i < 12; ++i) w.data()[i] = static_cast<float>(i) * 0.1f;
    dense.set_weights(w);

    layers::QuantizedDenseLayer qdense(dense);
    EXPECT_EQ(qdense.in_features(), 4u);
    EXPECT_EQ(qdense.out_features(), 3u);
    EXPECT_EQ(qdense.parameter_count(), 4u * 3 + 3);
}

TEST(QuantizedDenseTest, Forward1D) {
    layers::DenseLayer dense(4, 2, true);

    Tensor w({2, 4});
    for (size_t i = 0; i < 8; ++i) w.data()[i] = static_cast<float>(i) - 3.0f;
    dense.set_weights(w);

    Tensor b({2});
    b.data()[0] = 0.1f; b.data()[1] = 0.2f;
    dense.set_bias(b);

    Tensor input({4});
    input.data()[0] = 1.0f; input.data()[1] = 2.0f;
    input.data()[2] = 3.0f; input.data()[3] = 4.0f;

    // FP32 reference
    Tensor ref_out({1});
    dense.forward(input, ref_out);

    // Quantized
    layers::QuantizedDenseLayer qdense(dense);
    Tensor q_out({1});
    qdense.forward(input, q_out);

    EXPECT_EQ(q_out.shape(), ref_out.shape());
    for (size_t i = 0; i < ref_out.size(); ++i) {
        EXPECT_NEAR(q_out.data()[i], ref_out.data()[i],
                     std::abs(ref_out.data()[i]) * 0.2f + 1.0f);
    }
}

TEST(QuantizedDenseTest, Forward2D) {
    layers::DenseLayer dense(4, 2, false);

    Tensor w({2, 4});
    for (size_t i = 0; i < 8; ++i) w.data()[i] = static_cast<float>(i) * 0.5f;
    dense.set_weights(w);

    Tensor input({3, 4});
    for (size_t i = 0; i < 12; ++i) input.data()[i] = static_cast<float>(i) * 0.1f;

    Tensor ref_out({1});
    dense.forward(input, ref_out);

    layers::QuantizedDenseLayer qdense(dense);
    Tensor q_out({1});
    qdense.forward(input, q_out);

    EXPECT_EQ(q_out.shape(), ref_out.shape());
    for (size_t i = 0; i < ref_out.size(); ++i) {
        EXPECT_NEAR(q_out.data()[i], ref_out.data()[i],
                     std::abs(ref_out.data()[i]) * 0.25f + 1.0f);
    }
}

TEST(QuantizedDenseTest, OutputShape) {
    layers::DenseLayer dense(10, 5);
    layers::QuantizedDenseLayer qdense(dense);

    auto shape1 = qdense.output_shape({10});
    EXPECT_EQ(shape1[0], 5u);

    auto shape2 = qdense.output_shape({4, 10});
    EXPECT_EQ(shape2[0], 4u);
    EXPECT_EQ(shape2[1], 5u);
}

TEST(QuantizedDenseTest, Name) {
    layers::DenseLayer dense(10, 5);
    layers::QuantizedDenseLayer qdense(dense);
    EXPECT_EQ(qdense.name(), "QuantizedDense(10, 5)");
}
