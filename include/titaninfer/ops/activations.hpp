#pragma once

#include "titaninfer/tensor.hpp"

namespace titaninfer {
namespace ops {

/**
 * @brief ReLU activation: output[i] = max(0, input[i])
 *
 * @param input Input tensor (any shape)
 * @param output Output tensor (auto-allocated if shape mismatches)
 */
void relu(const Tensor& input, Tensor& output);

/**
 * @brief In-place ReLU: input[i] = max(0, input[i])
 */
void relu_inplace(Tensor& input);

/**
 * @brief Sigmoid activation: output[i] = 1 / (1 + exp(-input[i]))
 *
 * @param input Input tensor (any shape)
 * @param output Output tensor (auto-allocated if shape mismatches)
 */
void sigmoid(const Tensor& input, Tensor& output);

/**
 * @brief In-place sigmoid
 */
void sigmoid_inplace(Tensor& input);

/**
 * @brief Tanh activation: output[i] = tanh(input[i])
 *
 * Named tanh_activation to avoid collision with std::tanh.
 *
 * @param input Input tensor (any shape)
 * @param output Output tensor (auto-allocated if shape mismatches)
 */
void tanh_activation(const Tensor& input, Tensor& output);

/**
 * @brief In-place tanh
 */
void tanh_inplace(Tensor& input);

/**
 * @brief Softmax activation (numerically stable via log-sum-exp)
 *
 * - 1D tensor: softmax over all elements
 * - 2D tensor: row-wise softmax (axis=-1)
 * - 3D+ tensor: throws std::invalid_argument
 *
 * No in-place variant — the log-sum-exp trick requires reading all
 * input values before writing any output.
 *
 * @param input Input tensor (1D or 2D)
 * @param output Output tensor (auto-allocated if shape mismatches)
 * @throws std::invalid_argument if input has 3+ dimensions
 */
void softmax(const Tensor& input, Tensor& output);

} // namespace ops
} // namespace titaninfer
