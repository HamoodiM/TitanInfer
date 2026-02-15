#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace titaninfer {
namespace engine {

/**
 * @brief Fixed-size thread pool for parallel task execution
 *
 * Uses a task queue with mutex + condition variable synchronization.
 * Non-copyable, non-movable (owns threads and mutex).
 */
class ThreadPool {
public:
    /**
     * @brief Construct thread pool
     * @param num_threads Number of worker threads (0 = hardware_concurrency, min 1)
     */
    explicit ThreadPool(size_t num_threads = 0);

    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Submit a callable for async execution
     * @return std::future for the result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using R = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<R> result = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool: submit on stopped pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();

        return result;
    }

    size_t thread_count() const noexcept { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_;
};

} // namespace engine
} // namespace titaninfer
