#include "titaninfer/engine/thread_pool.hpp"

#include <algorithm>

namespace titaninfer {
namespace engine {

ThreadPool::ThreadPool(size_t num_threads)
    : stop_(false)
{
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 1;
        }
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
        worker.join();
    }
}

} // namespace engine
} // namespace titaninfer
