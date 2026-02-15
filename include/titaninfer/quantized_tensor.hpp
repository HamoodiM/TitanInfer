#pragma once

#include "titaninfer/tensor.hpp"
#include <cstdint>
#include <vector>

namespace titaninfer {

/**
 * @brief INT8 quantized tensor with per-tensor affine quantization
 *
 * Quantization: q = clamp(round(x / scale) + zero_point, -128, 127)
 * Dequantization: x = (q - zero_point) * scale
 *
 * Uses 32-byte aligned memory for SIMD operations.
 */
class QuantizedTensor {
public:
    explicit QuantizedTensor(const std::vector<size_t>& shape);
    QuantizedTensor(const QuantizedTensor& other);
    QuantizedTensor(QuantizedTensor&& other) noexcept;
    ~QuantizedTensor();

    QuantizedTensor& operator=(const QuantizedTensor& other);
    QuantizedTensor& operator=(QuantizedTensor&& other) noexcept;

    /// Quantize a FP32 tensor to INT8
    static QuantizedTensor quantize(const Tensor& fp32);

    /// Dequantize back to FP32
    Tensor dequantize() const;

    int8_t* data() noexcept { return data_; }
    const int8_t* data() const noexcept { return data_; }

    const std::vector<size_t>& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return size_; }
    size_t ndim() const noexcept { return shape_.size(); }

    float scale() const noexcept { return scale_; }
    int8_t zero_point() const noexcept { return zero_point_; }

    void set_scale(float s) noexcept { scale_ = s; }
    void set_zero_point(int8_t zp) noexcept { zero_point_ = zp; }

private:
    static int8_t* allocate_aligned(size_t num_elements);
    static void deallocate_aligned(int8_t* ptr);

    int8_t* data_;
    std::vector<size_t> shape_;
    size_t size_;
    float scale_;
    int8_t zero_point_;

    static constexpr size_t ALIGNMENT = 32;
};

} // namespace titaninfer
