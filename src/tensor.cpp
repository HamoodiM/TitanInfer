#include "titaninfer/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>

// Platform-specific aligned allocation
#ifdef _WIN32
    #include <malloc.h>
#else
    #include <cstdlib>
#endif

namespace titaninfer {

// ========================================
// Constructors & Destructor
// ========================================

Tensor::Tensor(const std::vector<size_t>& shape) 
    : data_(nullptr), shape_(shape), size_(0) {
    
    validate_shape(shape_);
    
    // Compute total size
    size_ = std::accumulate(shape_.begin(), shape_.end(), 
                            size_t(1), std::multiplies<size_t>());
    
    // Allocate aligned memory
    data_ = allocate_aligned(size_);
    
    // Zero-initialize
    std::memset(data_, 0, size_ * sizeof(float));
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(std::vector<size_t>(shape)) {}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr), shape_(other.shape_), size_(other.size_) {
    
    data_ = allocate_aligned(size_);
    std::memcpy(data_, other.data_, size_ * sizeof(float));
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_) {
    
    other.data_ = nullptr;
    other.size_ = 0;
}

Tensor::~Tensor() {
    deallocate_aligned(data_);
}

// ========================================
// Assignment Operators
// ========================================

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Deallocate old memory
        deallocate_aligned(data_);
        
        // Copy metadata
        shape_ = other.shape_;
        size_ = other.size_;
        
        // Allocate and copy data
        data_ = allocate_aligned(size_);
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Deallocate old memory
        deallocate_aligned(data_);
        
        // Move ownership
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        
        // Nullify source
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

// ========================================
// Data Access
// ========================================

float& Tensor::operator[](size_t index) {
#ifdef TITANINFER_DEBUG
    if (index >= size_) {
        throw std::out_of_range(
            "Index " + std::to_string(index) + 
            " out of range for tensor of size " + std::to_string(size_)
        );
    }
#endif
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
#ifdef TITANINFER_DEBUG
    if (index >= size_) {
        throw std::out_of_range(
            "Index " + std::to_string(index) + 
            " out of range for tensor of size " + std::to_string(size_)
        );
    }
#endif
    return data_[index];
}

// ========================================
// Memory Operations
// ========================================

void Tensor::fill(float value) {
    std::fill(data_, data_ + size_, value);
}

void Tensor::zero() {
    std::memset(data_, 0, size_ * sizeof(float));
}

// ========================================
// Private: Memory Management
// ========================================

float* Tensor::allocate_aligned(size_t num_elements) {
    if (num_elements == 0) {
        return nullptr;
    }
    
    size_t byte_size = num_elements * sizeof(float);
    
    // Round up to multiple of alignment for certain allocators
    size_t aligned_size = ((byte_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    
    float* ptr = nullptr;
    
#ifdef _WIN32
    // Windows: _aligned_malloc
    ptr = static_cast<float*>(_aligned_malloc(aligned_size, ALIGNMENT));
    if (!ptr) {
        throw std::bad_alloc();
    }
#elif defined(__APPLE__)
    // macOS: posix_memalign
    if (posix_memalign(reinterpret_cast<void**>(&ptr), ALIGNMENT, aligned_size) != 0) {
        throw std::bad_alloc();
    }
#else
    // Linux: aligned_alloc (C++17)
    ptr = static_cast<float*>(std::aligned_alloc(ALIGNMENT, aligned_size));
    if (!ptr) {
        throw std::bad_alloc();
    }
#endif
    
    return ptr;
}

void Tensor::deallocate_aligned(float* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void Tensor::validate_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            throw std::invalid_argument(
                "Tensor shape dimension " + std::to_string(i) + " cannot be zero"
            );
        }
    }
}

} // namespace titaninfer
