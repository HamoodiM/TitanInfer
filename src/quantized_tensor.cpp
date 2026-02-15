#include "titaninfer/quantized_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace titaninfer {

int8_t* QuantizedTensor::allocate_aligned(size_t num_elements) {
    if (num_elements == 0) {
        return nullptr;
    }
    // Round up to ALIGNMENT boundary
    size_t bytes = num_elements;
    size_t padded = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

#ifdef _WIN32
    void* ptr = _aligned_malloc(padded, ALIGNMENT);
    if (!ptr) throw std::bad_alloc();
    return static_cast<int8_t*>(ptr);
#elif defined(__APPLE__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, padded) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<int8_t*>(ptr);
#else
    void* ptr = std::aligned_alloc(ALIGNMENT, padded);
    if (!ptr) throw std::bad_alloc();
    return static_cast<int8_t*>(ptr);
#endif
}

void QuantizedTensor::deallocate_aligned(int8_t* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

QuantizedTensor::QuantizedTensor(const std::vector<size_t>& shape)
    : data_(nullptr)
    , shape_(shape)
    , size_(1)
    , scale_(1.0f)
    , zero_point_(0)
{
    for (auto d : shape) {
        size_ *= d;
    }
    data_ = allocate_aligned(size_);
    std::memset(data_, 0, size_);
}

QuantizedTensor::QuantizedTensor(const QuantizedTensor& other)
    : data_(nullptr)
    , shape_(other.shape_)
    , size_(other.size_)
    , scale_(other.scale_)
    , zero_point_(other.zero_point_)
{
    data_ = allocate_aligned(size_);
    std::memcpy(data_, other.data_, size_);
}

QuantizedTensor::QuantizedTensor(QuantizedTensor&& other) noexcept
    : data_(other.data_)
    , shape_(std::move(other.shape_))
    , size_(other.size_)
    , scale_(other.scale_)
    , zero_point_(other.zero_point_)
{
    other.data_ = nullptr;
    other.size_ = 0;
}

QuantizedTensor::~QuantizedTensor() {
    deallocate_aligned(data_);
}

QuantizedTensor& QuantizedTensor::operator=(const QuantizedTensor& other) {
    if (this != &other) {
        deallocate_aligned(data_);
        shape_ = other.shape_;
        size_ = other.size_;
        scale_ = other.scale_;
        zero_point_ = other.zero_point_;
        data_ = allocate_aligned(size_);
        std::memcpy(data_, other.data_, size_);
    }
    return *this;
}

QuantizedTensor& QuantizedTensor::operator=(QuantizedTensor&& other) noexcept {
    if (this != &other) {
        deallocate_aligned(data_);
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        scale_ = other.scale_;
        zero_point_ = other.zero_point_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

QuantizedTensor QuantizedTensor::quantize(const Tensor& fp32) {
    QuantizedTensor qt(fp32.shape());

    if (fp32.size() == 0) {
        return qt;
    }

    const float* src = fp32.data();
    const size_t n = fp32.size();

    // Find min/max, always include zero so zero_point stays in [-128, 127]
    float min_val = src[0];
    float max_val = src[0];
    for (size_t i = 1; i < n; ++i) {
        min_val = std::min(min_val, src[i]);
        max_val = std::max(max_val, src[i]);
    }
    min_val = std::min(min_val, 0.0f);
    max_val = std::max(max_val, 0.0f);

    // Handle degenerate case: all values the same
    if (max_val == min_val) {
        qt.scale_ = 1.0f;
        qt.zero_point_ = static_cast<int8_t>(
            std::max(-128.0f, std::min(127.0f, std::round(min_val))));
        for (size_t i = 0; i < n; ++i) {
            qt.data_[i] = qt.zero_point_;
        }
        return qt;
    }

    // Asymmetric quantization to [-128, 127]
    qt.scale_ = (max_val - min_val) / 255.0f;
    float inv_scale = 1.0f / qt.scale_;

    // zero_point chosen so that min_val maps to -128
    float zp_float = -128.0f - min_val * inv_scale;
    qt.zero_point_ = static_cast<int8_t>(
        std::max(-128.0f, std::min(127.0f, std::round(zp_float))));

    for (size_t i = 0; i < n; ++i) {
        float q = std::round(src[i] * inv_scale + static_cast<float>(qt.zero_point_));
        q = std::max(-128.0f, std::min(127.0f, q));
        qt.data_[i] = static_cast<int8_t>(q);
    }

    return qt;
}

Tensor QuantizedTensor::dequantize() const {
    Tensor fp32(shape_);
    float* dst = fp32.data();

    for (size_t i = 0; i < size_; ++i) {
        dst[i] = (static_cast<float>(data_[i]) - static_cast<float>(zero_point_)) * scale_;
    }

    return fp32;
}

} // namespace titaninfer
