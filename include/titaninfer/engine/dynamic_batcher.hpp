#pragma once

#include "titaninfer/tensor.hpp"
#include "titaninfer/layers/sequential.hpp"

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace titaninfer {
namespace engine {

struct BatcherConfig {
    size_t max_batch_size = 32;
    size_t max_wait_ms = 10;
};

/**
 * @brief Dynamic batcher for grouping concurrent inference requests
 *
 * Queues individual predict() requests and groups them into optimal
 * batch sizes. Uses a dedicated background thread to form batches.
 */
class DynamicBatcher {
public:
    /**
     * @param model Borrowed reference to Sequential model (caller keeps alive)
     * @param input_shape Shape of a single input sample
     * @param config Batching configuration
     */
    DynamicBatcher(layers::Sequential& model,
                   const std::vector<size_t>& input_shape,
                   const BatcherConfig& config = {});
    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    /// Submit a single input for inference, returns future for result
    std::future<Tensor> submit(Tensor input);

private:
    void batcher_loop();

    struct Request {
        Tensor input;
        std::promise<Tensor> promise;
    };

    layers::Sequential& model_;
    std::vector<size_t> input_shape_;
    BatcherConfig config_;

    std::queue<Request> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
    std::thread thread_;
};

} // namespace engine
} // namespace titaninfer
