#include "titaninfer/engine/dynamic_batcher.hpp"

#include <cstring>

namespace titaninfer {
namespace engine {

DynamicBatcher::DynamicBatcher(layers::Sequential& model,
                               const std::vector<size_t>& input_shape,
                               const BatcherConfig& config)
    : model_(model)
    , input_shape_(input_shape)
    , config_(config)
    , stop_(false)
    , thread_(&DynamicBatcher::batcher_loop, this)
{
}

DynamicBatcher::~DynamicBatcher() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
}

std::future<Tensor> DynamicBatcher::submit(Tensor input) {
    std::promise<Tensor> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("DynamicBatcher: submit on stopped batcher")));
            return future;
        }
        queue_.push(Request{std::move(input), std::move(promise)});
    }
    cv_.notify_one();

    return future;
}

void DynamicBatcher::batcher_loop() {
    for (;;) {
        std::vector<Request> batch;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for at least one request or stop signal
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });

            if (stop_ && queue_.empty()) {
                return;
            }

            // Wait up to max_wait_ms to collect more requests
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config_.max_wait_ms);

            while (batch.size() < config_.max_batch_size) {
                if (!queue_.empty()) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();
                } else {
                    // Wait for more or timeout
                    if (cv_.wait_until(lock, deadline,
                            [this] { return stop_ || !queue_.empty(); })) {
                        if (stop_ && queue_.empty()) {
                            break;
                        }
                    } else {
                        break; // Timeout
                    }
                }
            }
        }

        if (batch.empty()) {
            continue;
        }

        // Process batch
        try {
            if (batch.size() == 1) {
                // Single request — no need to form batch tensor
                Tensor result = model_.forward(batch[0].input);
                batch[0].promise.set_value(std::move(result));
            } else {
                // Stack inputs into batched tensor
                const size_t N = batch.size();
                size_t sample_size = 1;
                for (auto d : input_shape_) {
                    sample_size *= d;
                }

                std::vector<size_t> batch_shape = {N};
                batch_shape.insert(batch_shape.end(),
                                   input_shape_.begin(), input_shape_.end());
                Tensor batched_input(batch_shape);

                for (size_t i = 0; i < N; ++i) {
                    std::memcpy(batched_input.data() + i * sample_size,
                                batch[i].input.data(),
                                sample_size * sizeof(float));
                }

                Tensor batched_output = model_.forward(batched_input);

                // Split output
                size_t out_sample_size = batched_output.size() / N;
                std::vector<size_t> single_out_shape(
                    batched_output.shape().begin() + 1,
                    batched_output.shape().end());

                for (size_t i = 0; i < N; ++i) {
                    Tensor result(single_out_shape);
                    std::memcpy(result.data(),
                                batched_output.data() + i * out_sample_size,
                                out_sample_size * sizeof(float));
                    batch[i].promise.set_value(std::move(result));
                }
            }
        } catch (...) {
            // Propagate exception to all pending requests
            auto ex = std::current_exception();
            for (auto& req : batch) {
                req.promise.set_exception(ex);
            }
        }
    }
}

} // namespace engine
} // namespace titaninfer
