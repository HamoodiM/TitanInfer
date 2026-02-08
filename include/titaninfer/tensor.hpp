#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>
#include <initializer_list>

namespace titaninfer {

/**
 * @brief Core Tensor class with aligned memory for SIMD operations
 * 
 * Provides RAII-managed, 32-byte aligned memory allocation suitable for AVX2.
 * Supports dynamic N-dimensional shapes with zero-copy move semantics.
 */
class Tensor {
public:
    // ========================================
    // Constructors & Destructor
    // ========================================
    
    /**
     * @brief Construct tensor with given shape (zero-initialized)
     * @param shape Dimensions of the tensor (e.g., {32, 64, 128})
     * @throws std::invalid_argument if shape is empty or contains zeros
     */
    explicit Tensor(const std::vector<size_t>& shape);
    
    /**
     * @brief Construct tensor from initializer list
     * @param shape Dimensions (e.g., {2, 3, 4})
     */
    explicit Tensor(std::initializer_list<size_t> shape);
    
    /**
     * @brief Copy constructor (deep copy)
     */
    Tensor(const Tensor& other);
    
    /**
     * @brief Move constructor (zero-copy)
     */
    Tensor(Tensor&& other) noexcept;
    
    /**
     * @brief Destructor (releases aligned memory)
     */
    ~Tensor();
    
    // ========================================
    // Assignment Operators
    // ========================================
    
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // ========================================
    // Data Access
    // ========================================
    
    /**
     * @brief Access element by flat index
     * @param index Linear index into data array
     * @return Reference to element
     * @note Bounds checking enabled in Debug builds (TITANINFER_DEBUG)
     */
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    
    /**
     * @brief Access element by multi-dimensional indices
     * @param indices Variable number of dimension indices
     * @return Reference to element
     * @throws std::out_of_range if indices mismatch shape dimensionality (debug mode)
     */
    template<typename... Indices>
    float& operator()(Indices... indices);
    
    template<typename... Indices>
    const float& operator()(Indices... indices) const;
    
    /**
     * @brief Raw pointer access (for SIMD operations)
     * @return Aligned pointer to data
     */
    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    
    // ========================================
    // Shape & Size Queries
    // ========================================
    
    /**
     * @brief Get tensor shape
     */
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    
    /**
     * @brief Total number of elements
     */
    size_t size() const noexcept { return size_; }
    
    /**
     * @brief Number of dimensions
     */
    size_t ndim() const noexcept { return shape_.size(); }
    
    /**
     * @brief Check if tensor is empty
     */
    bool empty() const noexcept { return size_ == 0; }
    
    // ========================================
    // Memory Operations
    // ========================================
    
    /**
     * @brief Fill tensor with a scalar value
     */
    void fill(float value);
    
    /**
     * @brief Zero out all elements
     */
    void zero();
    
private:
    // ========================================
    // Memory Management
    // ========================================
    
    /**
     * @brief Allocate 32-byte aligned memory (cross-platform)
     */
    static float* allocate_aligned(size_t num_elements);
    
    /**
     * @brief Free aligned memory (cross-platform)
     */
    static void deallocate_aligned(float* ptr);
    
    /**
     * @brief Compute flat index from multi-dimensional indices
     */
    template<typename... Indices>
    size_t compute_flat_index(Indices... indices) const;
    
    /**
     * @brief Validate shape vector
     */
    static void validate_shape(const std::vector<size_t>& shape);
    
    // ========================================
    // Member Variables
    // ========================================
    
    float* data_;                    // 32-byte aligned data pointer
    std::vector<size_t> shape_;      // Tensor dimensions
    size_t size_;                    // Total number of elements
    
    static constexpr size_t ALIGNMENT = 32;  // AVX2 alignment requirement
};

// ========================================
// Template Implementations
// ========================================

template<typename... Indices>
float& Tensor::operator()(Indices... indices) {
    size_t flat_idx = compute_flat_index(indices...);
    return data_[flat_idx];
}

template<typename... Indices>
const float& Tensor::operator()(Indices... indices) const {
    size_t flat_idx = compute_flat_index(indices...);
    return data_[flat_idx];
}

template<typename... Indices>
size_t Tensor::compute_flat_index(Indices... indices) const {
#ifdef TITANINFER_DEBUG
    if (sizeof...(indices) != shape_.size()) {
        throw std::out_of_range(
            "Index dimension mismatch: expected " + std::to_string(shape_.size()) +
            ", got " + std::to_string(sizeof...(indices))
        );
    }
#endif
    
    size_t idx_array[] = {static_cast<size_t>(indices)...};
    size_t flat_idx = 0;
    size_t stride = 1;
    
    // Row-major order computation
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
#ifdef TITANINFER_DEBUG
        if (idx_array[i] >= shape_[i]) {
            throw std::out_of_range(
                "Index " + std::to_string(idx_array[i]) + 
                " out of range for dimension " + std::to_string(i) + 
                " with size " + std::to_string(shape_[i])
            );
        }
#endif
        flat_idx += idx_array[i] * stride;
        stride *= shape_[i];
    }
    
    return flat_idx;
}

} // namespace titaninfer
