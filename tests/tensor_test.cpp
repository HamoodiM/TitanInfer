#include <gtest/gtest.h>
#include "titaninfer/tensor.hpp"
#include <cstdint>

using namespace titaninfer;

// ========================================
// Memory Alignment Tests
// ========================================

TEST(TensorTest, MemoryAlignment) {
    Tensor t({32, 64});
    auto ptr = reinterpret_cast<std::uintptr_t>(t.data());
    EXPECT_EQ(ptr % 32, 0) << "Memory must be 32-byte aligned for AVX2";
}

TEST(TensorTest, MultipleAllocationsAligned) {
    for (int i = 0; i < 100; ++i) {
        Tensor t({10, 20, 30});
        auto ptr = reinterpret_cast<std::uintptr_t>(t.data());
        EXPECT_EQ(ptr % 32, 0);
    }
}

// ========================================
// Construction Tests
// ========================================

TEST(TensorTest, ConstructWithVector) {
    std::vector<size_t> shape = {2, 3, 4};
    Tensor t(shape);
    
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_FALSE(t.empty());
}

TEST(TensorTest, ConstructWithInitializerList) {
    Tensor t({5, 10});
    
    EXPECT_EQ(t.size(), 50);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 5);
    EXPECT_EQ(t.shape()[1], 10);
}

TEST(TensorTest, ZeroInitialization) {
    Tensor t({100});
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST(TensorTest, InvalidShapeEmpty) {
    EXPECT_THROW(Tensor t(std::vector<size_t>{}), std::invalid_argument);
}

TEST(TensorTest, InvalidShapeZeroDimension) {
    EXPECT_THROW(Tensor t({3, 0, 5}), std::invalid_argument);
}

// ========================================
// Copy & Move Semantics Tests
// ========================================

TEST(TensorTest, CopyConstructor) {
    Tensor t1({2, 3});
    t1.fill(42.0f);
    
    Tensor t2(t1);
    
    EXPECT_EQ(t2.shape(), t1.shape());
    EXPECT_EQ(t2.size(), t1.size());
    EXPECT_NE(t2.data(), t1.data()) << "Data must be deep-copied";
    
    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_FLOAT_EQ(t2[i], 42.0f);
    }
}

TEST(TensorTest, MoveConstructor) {
    Tensor t1({3, 4});
    t1.fill(3.14f);
    float* original_ptr = t1.data();
    
    Tensor t2(std::move(t1));
    
    EXPECT_EQ(t2.size(), 12);
    EXPECT_EQ(t2.data(), original_ptr) << "Data must be moved, not copied";
    EXPECT_EQ(t1.data(), nullptr) << "Source must be nullified";
    EXPECT_EQ(t1.size(), 0);
}

TEST(TensorTest, CopyAssignment) {
    Tensor t1({2, 2});
    t1.fill(1.5f);
    
    Tensor t2({10});
    t2 = t1;
    
    EXPECT_EQ(t2.shape(), t1.shape());
    EXPECT_NE(t2.data(), t1.data());
    EXPECT_FLOAT_EQ(t2[0], 1.5f);
}

TEST(TensorTest, MoveAssignment) {
    Tensor t1({5, 5});
    t1.fill(2.71f);
    float* original_ptr = t1.data();
    
    Tensor t2({1});
    t2 = std::move(t1);
    
    EXPECT_EQ(t2.size(), 25);
    EXPECT_EQ(t2.data(), original_ptr);
    EXPECT_EQ(t1.data(), nullptr);
}

TEST(TensorTest, SelfAssignment) {
    Tensor t({3, 3});
    t.fill(7.0f);
    
    t = t;  // Self-assignment
    
    EXPECT_EQ(t.size(), 9);
    EXPECT_FLOAT_EQ(t[0], 7.0f);
}

// ========================================
// Data Access Tests
// ========================================

TEST(TensorTest, FlatIndexAccess) {
    Tensor t({10});
    
    for (size_t i = 0; i < 10; ++i) {
        t[i] = static_cast<float>(i);
    }
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(t[i], static_cast<float>(i));
    }
}

TEST(TensorTest, MultiDimensionalAccess) {
    Tensor t({2, 3, 4});
    
    t(0, 0, 0) = 1.0f;
    t(1, 2, 3) = 99.0f;
    
    EXPECT_FLOAT_EQ(t(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(1, 2, 3), 99.0f);
}

TEST(TensorTest, RowMajorOrdering) {
    Tensor t({2, 3});  // 2 rows, 3 columns
    
    // Fill: [[1, 2, 3],
    //        [4, 5, 6]]
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t(i, j) = static_cast<float>(i * 3 + j + 1);
        }
    }
    
    EXPECT_FLOAT_EQ(t[0], 1.0f);  // t(0,0)
    EXPECT_FLOAT_EQ(t[1], 2.0f);  // t(0,1)
    EXPECT_FLOAT_EQ(t[2], 3.0f);  // t(0,2)
    EXPECT_FLOAT_EQ(t[3], 4.0f);  // t(1,0)
}

// ========================================
// Debug Mode Tests (Bounds Checking)
// ========================================

#ifdef TITANINFER_DEBUG
TEST(TensorTest, BoundsCheckingFlatIndex) {
    Tensor t({5});
    EXPECT_THROW(t[5], std::out_of_range);
    EXPECT_THROW(t[100], std::out_of_range);
}

TEST(TensorTest, BoundsCheckingMultiDim) {
    Tensor t({2, 3});
    EXPECT_THROW(t(2, 0), std::out_of_range);  // First dim out of range
    EXPECT_THROW(t(0, 3), std::out_of_range);  // Second dim out of range
}

TEST(TensorTest, DimensionMismatch) {
    Tensor t({2, 3, 4});
    EXPECT_THROW(t(0, 0), std::out_of_range);  // Too few indices
}
#endif

// ========================================
// Memory Operations Tests
// ========================================

TEST(TensorTest, FillOperation) {
    Tensor t({20});
    t.fill(3.14159f);
    
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 3.14159f);
    }
}

TEST(TensorTest, ZeroOperation) {
    Tensor t({10});
    t.fill(42.0f);
    t.zero();
    
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

// ========================================
// Edge Cases
// ========================================

TEST(TensorTest, LargeTensor) {
    Tensor t({512, 512});  // 256K elements ≈ 1MB
    EXPECT_EQ(t.size(), 262144);
    
    t.fill(1.0f);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[t.size() - 1], 1.0f);
}

TEST(TensorTest, HighDimensionalTensor) {
    Tensor t({2, 2, 2, 2, 2});  // 5D tensor
    EXPECT_EQ(t.ndim(), 5);
    EXPECT_EQ(t.size(), 32);
    
    t(1, 1, 1, 1, 1) = 777.0f;
    EXPECT_FLOAT_EQ(t(1, 1, 1, 1, 1), 777.0f);
}

TEST(TensorTest, SingleElementTensor) {
    Tensor t({1});
    t[0] = 123.456f;
    EXPECT_FLOAT_EQ(t[0], 123.456f);
}
