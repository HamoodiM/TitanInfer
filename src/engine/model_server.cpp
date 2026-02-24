#include "titaninfer/engine/model_server.hpp"
#include "titaninfer/engine/inference_engine.hpp"
#include "titaninfer/engine/thread_pool.hpp"
#include "titaninfer/exceptions.hpp"
#include "titaninfer/logger.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <utility>

namespace titaninfer::engine {

// ===========================================================================
// Anonymous namespace — internal components
// ===========================================================================
namespace {

// ---------------------------------------------------------------------------
// Request ID generator
// ---------------------------------------------------------------------------
std::string generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "req-%016llx-%08llx",
                  static_cast<unsigned long long>(now),
                  static_cast<unsigned long long>(id));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// EnginePool — pool of InferenceEngine instances for one model-version
// ---------------------------------------------------------------------------
class EnginePool {
public:
    EnginePool(const std::string& model_path, size_t pool_size,
               bool profiling)
    {
        engines_.reserve(pool_size);
        in_use_.resize(pool_size, false);
        for (size_t i = 0; i < pool_size; ++i) {
            auto engine = InferenceEngine::Builder()
                .setModelPath(model_path)
                .enableProfiling(profiling)
                .build();
            if (i == 0) {
                input_shape_ = engine.expected_input_shape();
            }
            engines_.push_back(std::move(engine));
        }
    }

    EnginePool(const EnginePool&) = delete;
    EnginePool& operator=(const EnginePool&) = delete;

    class Lease {
    public:
        Lease(EnginePool& pool, size_t index)
            : pool_(&pool), index_(index) {}

        ~Lease() {
            if (pool_) {
                pool_->release(index_);
            }
        }

        Lease(Lease&& other) noexcept
            : pool_(other.pool_), index_(other.index_) {
            other.pool_ = nullptr;
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                if (pool_) pool_->release(index_);
                pool_ = other.pool_;
                index_ = other.index_;
                other.pool_ = nullptr;
            }
            return *this;
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        InferenceEngine& engine() { return pool_->engines_[index_]; }

    private:
        EnginePool* pool_;
        size_t index_;
    };

    Lease acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            for (size_t i = 0; i < in_use_.size(); ++i) {
                if (!in_use_[i]) return true;
            }
            return false;
        });
        for (size_t i = 0; i < in_use_.size(); ++i) {
            if (!in_use_[i]) {
                in_use_[i] = true;
                return Lease(*this, i);
            }
        }
        // Should never reach here due to the wait predicate
        throw ServerException("EnginePool: no available engine",
                              ErrorCode::INTERNAL_ERROR);
    }

    size_t pool_size() const noexcept { return engines_.size(); }
    const std::vector<size_t>& input_shape() const { return input_shape_; }

private:
    void release(size_t index) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            in_use_[index] = false;
        }
        cv_.notify_one();
    }

    std::vector<InferenceEngine> engines_;
    std::vector<bool> in_use_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<size_t> input_shape_;
};

// ---------------------------------------------------------------------------
// ModelCache — LRU cache of loaded EnginePool instances
// ---------------------------------------------------------------------------
using CacheKey = std::pair<std::string, uint32_t>;

struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const {
        size_t h1 = std::hash<std::string>{}(k.first);
        size_t h2 = std::hash<uint32_t>{}(k.second);
        return h1 ^ (h2 * 2654435761u);
    }
};

class ModelCache {
public:
    ModelCache(size_t max_loaded, size_t pool_size, bool profiling)
        : max_loaded_(max_loaded), pool_size_(pool_size),
          profiling_(profiling) {}

    std::shared_ptr<EnginePool> get_or_load(
        const ModelVersionInfo& info,
        const std::function<bool(const CacheKey&)>& is_pinned)
    {
        CacheKey key{info.name, info.version};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pools_.find(key);
            if (it != pools_.end()) {
                touch(key);
                return it->second;
            }
        }

        // Load outside the lock (slow I/O)
        TITANINFER_LOG_INFO("Loading model '" + info.name +
                            "' v" + std::to_string(info.version) +
                            " from " + info.file_path);
        auto pool = std::make_shared<EnginePool>(
            info.file_path, pool_size_, profiling_);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Double-check: another thread may have loaded it
            auto it = pools_.find(key);
            if (it != pools_.end()) {
                touch(key);
                return it->second;
            }

            // Evict if at capacity
            while (pools_.size() >= max_loaded_ && !lru_order_.empty()) {
                evict_lru(is_pinned);
            }

            pools_[key] = pool;
            lru_order_.push_front(key);
            lru_map_[key] = lru_order_.begin();
        }

        TITANINFER_LOG_INFO("Loaded model '" + info.name +
                            "' v" + std::to_string(info.version) +
                            " (pool size: " + std::to_string(pool_size_) + ")");
        return pool;
    }

    void evict(const CacheKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto map_it = lru_map_.find(key);
        if (map_it != lru_map_.end()) {
            lru_order_.erase(map_it->second);
            lru_map_.erase(map_it);
        }
        pools_.erase(key);
        TITANINFER_LOG_INFO("Evicted model '" + key.first +
                            "' v" + std::to_string(key.second));
    }

    void replace(const CacheKey& key, std::shared_ptr<EnginePool> pool) {
        std::lock_guard<std::mutex> lock(mutex_);
        pools_[key] = std::move(pool);
        touch(key);
    }

    size_t loaded_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pools_.size();
    }

private:
    void touch(const CacheKey& key) {
        auto it = lru_map_.find(key);
        if (it != lru_map_.end()) {
            lru_order_.erase(it->second);
            lru_order_.push_front(key);
            it->second = lru_order_.begin();
        }
    }

    void evict_lru(const std::function<bool(const CacheKey&)>& is_pinned) {
        // Walk from back (least recently used) to find an evictable entry
        for (auto rit = lru_order_.rbegin(); rit != lru_order_.rend(); ++rit) {
            if (!is_pinned(*rit)) {
                CacheKey victim = *rit;
                lru_map_.erase(victim);
                lru_order_.erase(std::next(rit).base());
                pools_.erase(victim);
                TITANINFER_LOG_INFO("LRU evicted model '" + victim.first +
                                    "' v" + std::to_string(victim.second));
                return;
            }
        }
        // All models are pinned — cannot evict
        TITANINFER_LOG_WARNING("Cannot evict: all loaded models are pinned");
    }

    size_t max_loaded_;
    size_t pool_size_;
    bool profiling_;

    std::list<CacheKey> lru_order_;
    std::unordered_map<CacheKey, std::list<CacheKey>::iterator, CacheKeyHash> lru_map_;
    std::unordered_map<CacheKey, std::shared_ptr<EnginePool>, CacheKeyHash> pools_;
    mutable std::mutex mutex_;
};

// ---------------------------------------------------------------------------
// RateLimiter — token bucket per tenant
// ---------------------------------------------------------------------------
class RateLimiter {
public:
    void set_quota(const std::string& tenant_id, const TenantQuota& quota) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& state = tenants_[tenant_id];
        state.quota = quota;
        state.tokens = quota.max_qps;
        state.last_refill = std::chrono::steady_clock::now();
        state.concurrent.store(0, std::memory_order_relaxed);
    }

    void remove_quota(const std::string& tenant_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        tenants_.erase(tenant_id);
    }

    // Returns true if request is allowed
    bool try_acquire(const std::string& tenant_id) {
        if (tenant_id.empty()) return true;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tenants_.find(tenant_id);
        if (it == tenants_.end()) return true; // No quota = unlimited

        auto& state = it->second;
        refill_tokens(state);

        if (state.tokens < 1.0) return false;

        if (state.concurrent.load(std::memory_order_relaxed) >=
            state.quota.max_concurrent) {
            return false;
        }

        state.tokens -= 1.0;
        state.concurrent.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void release(const std::string& tenant_id) {
        if (tenant_id.empty()) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tenants_.find(tenant_id);
        if (it != tenants_.end()) {
            auto val = it->second.concurrent.load(std::memory_order_relaxed);
            if (val > 0) {
                it->second.concurrent.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

private:
    struct TenantState {
        TenantQuota quota;
        double tokens = 0.0;
        std::chrono::steady_clock::time_point last_refill;
        std::atomic<size_t> concurrent{0};
    };

    void refill_tokens(TenantState& state) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(
            now - state.last_refill).count();
        state.tokens = std::min(state.quota.max_qps,
                                state.tokens + elapsed * state.quota.max_qps);
        state.last_refill = now;
    }

    std::unordered_map<std::string, TenantState> tenants_;
    std::mutex mutex_;
};

// ---------------------------------------------------------------------------
// TrafficSplitter — weighted random version selection
// ---------------------------------------------------------------------------
class TrafficSplitter {
public:
    void set_rules(const std::string& model_name,
                   const std::vector<TrafficRule>& rules) {
        std::lock_guard<std::mutex> lock(mutex_);
        rules_[model_name] = rules;
    }

    void remove_rules(const std::string& model_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        rules_.erase(model_name);
    }

    // Returns 0 if no rules set (caller should use default routing)
    uint32_t select_version(const std::string& model_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = rules_.find(model_name);
        if (it == rules_.end() || it->second.empty()) return 0;

        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);

        double cumulative = 0.0;
        for (const auto& rule : it->second) {
            cumulative += rule.weight;
            if (r <= cumulative) return rule.version;
        }
        return it->second.back().version;
    }

private:
    std::unordered_map<std::string, std::vector<TrafficRule>> rules_;
    std::mutex mutex_;
};

// ---------------------------------------------------------------------------
// Path parser
// ---------------------------------------------------------------------------
struct ParsedRoute {
    std::string model_name;
    uint32_t version = 0;    // 0 = use default routing
    bool valid = false;
};

ParsedRoute parse_path(const std::string& path) {
    ParsedRoute result;

    // Split by '/'
    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string segment;
    while (std::getline(iss, segment, '/')) {
        if (!segment.empty()) parts.push_back(segment);
    }

    // /v1/models/{name}/predict -> parts: [v1, models, name, predict]
    // /v1/models/{name}/versions/{ver}/predict -> parts: [v1, models, name, versions, ver, predict]

    if (parts.size() < 4) return result;
    if (parts[0] != "v1" || parts[1] != "models") return result;

    result.model_name = parts[2];

    if (parts.size() == 4 && parts[3] == "predict") {
        result.valid = true;
    } else if (parts.size() == 6 && parts[3] == "versions" && parts[5] == "predict") {
        try {
            result.version = static_cast<uint32_t>(std::stoul(parts[4]));
        } catch (...) {
            return result;
        }
        result.valid = true;
    }

    return result;
}

} // anonymous namespace

// ===========================================================================
// ModelServer::Impl
// ===========================================================================

struct ModelServer::Impl {
    ModelServerConfig config;
    std::unique_ptr<ThreadPool> thread_pool;

    // Registry: name -> (version -> info)
    std::unordered_map<std::string,
        std::map<uint32_t, ModelVersionInfo>> registry;
    mutable std::shared_mutex registry_mutex;

    std::unique_ptr<ModelCache> cache;
    RateLimiter rate_limiter;
    TrafficSplitter traffic_splitter;
    std::atomic<uint64_t> request_counter{0};

    explicit Impl(const ModelServerConfig& cfg)
        : config(cfg)
    {
        size_t threads = cfg.worker_threads > 0
            ? cfg.worker_threads
            : std::max(size_t{1}, static_cast<size_t>(
                std::thread::hardware_concurrency()));
        size_t per_model = cfg.engines_per_model > 0
            ? cfg.engines_per_model : threads;

        thread_pool = std::make_unique<ThreadPool>(threads);
        cache = std::make_unique<ModelCache>(
            cfg.max_loaded_models, per_model, cfg.enable_profiling);
    }

    // Check if a cache key corresponds to a pinned model
    bool is_pinned(const CacheKey& key) const {
        // Called under cache lock but needs registry read
        std::shared_lock<std::shared_mutex> lock(registry_mutex);
        auto name_it = registry.find(key.first);
        if (name_it == registry.end()) return false;
        auto ver_it = name_it->second.find(key.second);
        if (ver_it == name_it->second.end()) return false;
        return ver_it->second.pinned;
    }

    // Find the ModelVersionInfo for a resolved route
    ModelVersionInfo find_version(const std::string& name,
                                  uint32_t version) const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex);
        auto name_it = registry.find(name);
        if (name_it == registry.end()) {
            throw ServerException("Model '" + name + "' not found",
                                  ErrorCode::MODEL_NOT_FOUND);
        }
        if (version == 0) {
            // Default: highest version number
            if (name_it->second.empty()) {
                throw ServerException("Model '" + name + "' has no versions",
                                      ErrorCode::VERSION_NOT_FOUND);
            }
            return name_it->second.rbegin()->second;
        }
        auto ver_it = name_it->second.find(version);
        if (ver_it == name_it->second.end()) {
            throw ServerException("Model '" + name + "' version " +
                                  std::to_string(version) + " not found",
                                  ErrorCode::VERSION_NOT_FOUND);
        }
        return ver_it->second;
    }

    Response do_predict(const std::string& model_name,
                        const Tensor& input,
                        const std::string& tenant_id,
                        const std::string& req_id)
    {
        auto start = std::chrono::steady_clock::now();
        std::string request_id = req_id.empty() ? generate_request_id() : req_id;

        Response response;
        response.headers["X-Request-Id"] = request_id;

        // Rate limiting
        if (!rate_limiter.try_acquire(tenant_id)) {
            response.status_code = 429;
            response.error_message = "Quota exceeded for tenant '" +
                                     tenant_id + "'";
            TITANINFER_LOG_WARNING("[" + request_id + "] " +
                                  response.error_message);
            return response;
        }

        // RAII guard for releasing quota on all exit paths
        struct QuotaGuard {
            RateLimiter& rl;
            const std::string& tid;
            ~QuotaGuard() { rl.release(tid); }
        } guard{rate_limiter, tenant_id};

        try {
            // Route: traffic splitting or default
            uint32_t version = traffic_splitter.select_version(model_name);

            ModelVersionInfo info = find_version(model_name, version);

            // Load or fetch from cache
            auto pool = cache->get_or_load(
                info, [this](const CacheKey& k) { return is_pinned(k); });

            // Acquire engine from pool
            auto lease = pool->acquire();

            TITANINFER_LOG_DEBUG("[" + request_id + "] Predicting on '" +
                                model_name + "' v" +
                                std::to_string(info.version));

            // Run inference
            Tensor output = lease.engine().predict(input);

            auto end = std::chrono::steady_clock::now();
            response.status_code = 200;
            response.body = std::move(output);
            response.latency_ms = std::chrono::duration<double, std::milli>(
                end - start).count();
            response.headers["X-Model-Version"] =
                std::to_string(info.version);

        } catch (const ServerException& e) {
            response.status_code =
                (e.error_code() == ErrorCode::MODEL_NOT_FOUND ||
                 e.error_code() == ErrorCode::VERSION_NOT_FOUND) ? 404 : 500;
            response.error_message = e.what();
            TITANINFER_LOG_ERROR("[" + request_id + "] " +
                                response.error_message);
        } catch (const ValidationException& e) {
            response.status_code = 400;
            response.error_message = e.what();
            TITANINFER_LOG_WARNING("[" + request_id + "] " +
                                  response.error_message);
        } catch (const std::invalid_argument& e) {
            response.status_code = 400;
            response.error_message = e.what();
            TITANINFER_LOG_WARNING("[" + request_id + "] " +
                                  response.error_message);
        } catch (const TitanInferException& e) {
            response.status_code = 500;
            response.error_message = e.what();
            TITANINFER_LOG_ERROR("[" + request_id + "] " +
                                response.error_message);
        } catch (const std::exception& e) {
            response.status_code = 500;
            response.error_message = e.what();
            TITANINFER_LOG_ERROR("[" + request_id + "] Internal error: " +
                                std::string(e.what()));
        }

        return response;
    }
};

// ===========================================================================
// ModelServer::Builder
// ===========================================================================

ModelServer::Builder& ModelServer::Builder::setMaxLoadedModels(size_t count) {
    config_.max_loaded_models = count;
    return *this;
}

ModelServer::Builder& ModelServer::Builder::setWorkerThreads(size_t count) {
    config_.worker_threads = count;
    return *this;
}

ModelServer::Builder& ModelServer::Builder::setEnginesPerModel(size_t count) {
    config_.engines_per_model = count;
    return *this;
}

ModelServer::Builder& ModelServer::Builder::enableProfiling(bool enable) {
    config_.enable_profiling = enable;
    return *this;
}

ModelServer ModelServer::Builder::build() {
    return ModelServer(config_);
}

// ===========================================================================
// ModelServer
// ===========================================================================

ModelServer::ModelServer(const ModelServerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ModelServer::~ModelServer() = default;

ModelServer::ModelServer(ModelServer&&) noexcept = default;
ModelServer& ModelServer::operator=(ModelServer&&) noexcept = default;

// ---- Registry ----

void ModelServer::register_model(const std::string& name,
                                  uint32_t version,
                                  const std::string& file_path,
                                  const std::set<std::string>& tags) {
    std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);

    ModelVersionInfo info;
    info.name = name;
    info.version = version;
    info.file_path = file_path;
    info.tags = tags;
    info.pinned = false;

    impl_->registry[name][version] = info;

    TITANINFER_LOG_INFO("Registered model '" + name + "' v" +
                        std::to_string(version) + " at " + file_path);
}

void ModelServer::unregister_model(const std::string& name,
                                    uint32_t version) {
    {
        std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
        auto name_it = impl_->registry.find(name);
        if (name_it != impl_->registry.end()) {
            name_it->second.erase(version);
            if (name_it->second.empty()) {
                impl_->registry.erase(name_it);
            }
        }
    }
    impl_->cache->evict(CacheKey{name, version});
    impl_->traffic_splitter.remove_rules(name);

    TITANINFER_LOG_INFO("Unregistered model '" + name + "' v" +
                        std::to_string(version));
}

void ModelServer::add_tag(const std::string& name, uint32_t version,
                           const std::string& tag) {
    std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
    auto name_it = impl_->registry.find(name);
    if (name_it == impl_->registry.end()) {
        throw ServerException("Model '" + name + "' not found",
                              ErrorCode::MODEL_NOT_FOUND);
    }
    auto ver_it = name_it->second.find(version);
    if (ver_it == name_it->second.end()) {
        throw ServerException("Version " + std::to_string(version) +
                              " not found", ErrorCode::VERSION_NOT_FOUND);
    }
    ver_it->second.tags.insert(tag);
}

void ModelServer::remove_tag(const std::string& name, uint32_t version,
                              const std::string& tag) {
    std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
    auto name_it = impl_->registry.find(name);
    if (name_it == impl_->registry.end()) return;
    auto ver_it = name_it->second.find(version);
    if (ver_it == name_it->second.end()) return;
    ver_it->second.tags.erase(tag);
}

void ModelServer::pin_model(const std::string& name, uint32_t version) {
    std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
    auto name_it = impl_->registry.find(name);
    if (name_it == impl_->registry.end()) {
        throw ServerException("Model '" + name + "' not found",
                              ErrorCode::MODEL_NOT_FOUND);
    }
    auto ver_it = name_it->second.find(version);
    if (ver_it == name_it->second.end()) {
        throw ServerException("Version " + std::to_string(version) +
                              " not found", ErrorCode::VERSION_NOT_FOUND);
    }
    ver_it->second.pinned = true;
}

void ModelServer::unpin_model(const std::string& name, uint32_t version) {
    std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
    auto name_it = impl_->registry.find(name);
    if (name_it == impl_->registry.end()) return;
    auto ver_it = name_it->second.find(version);
    if (ver_it == name_it->second.end()) return;
    ver_it->second.pinned = false;
}

std::vector<ModelVersionInfo> ModelServer::list_versions(
    const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(impl_->registry_mutex);
    auto it = impl_->registry.find(name);
    if (it == impl_->registry.end()) return {};
    std::vector<ModelVersionInfo> result;
    result.reserve(it->second.size());
    for (const auto& pair : it->second) {
        result.push_back(pair.second);
    }
    return result;
}

// ---- Quotas ----

void ModelServer::set_tenant_quota(const std::string& tenant_id,
                                    const TenantQuota& quota) {
    impl_->rate_limiter.set_quota(tenant_id, quota);
    TITANINFER_LOG_INFO("Set quota for tenant '" + tenant_id +
                        "': max_qps=" + std::to_string(quota.max_qps) +
                        ", max_concurrent=" +
                        std::to_string(quota.max_concurrent));
}

void ModelServer::remove_tenant_quota(const std::string& tenant_id) {
    impl_->rate_limiter.remove_quota(tenant_id);
}

// ---- Traffic Splitting ----

void ModelServer::set_traffic_rules(const std::string& model_name,
                                     const std::vector<TrafficRule>& rules) {
    impl_->traffic_splitter.set_rules(model_name, rules);
}

// ---- Inference ----

Response ModelServer::predict(const std::string& model_name,
                               const Tensor& input,
                               const std::string& tenant_id,
                               const std::string& request_id) {
    return impl_->do_predict(model_name, input, tenant_id, request_id);
}

std::future<Response> ModelServer::predict_async(
    const std::string& model_name,
    const Tensor& input,
    const std::string& tenant_id,
    const std::string& request_id)
{
    // Deep copy input for thread safety (Tensor has deep copy ctor)
    Tensor input_copy(input);
    std::string name_copy = model_name;
    std::string tenant_copy = tenant_id;
    std::string req_copy = request_id;

    return impl_->thread_pool->submit(
        [this, name_copy, input_copy, tenant_copy, req_copy]() {
            return impl_->do_predict(name_copy, input_copy,
                                     tenant_copy, req_copy);
        });
}

Response ModelServer::handle_request(const Request& request) {
    if (request.method != HttpMethod::POST) {
        Response response;
        response.status_code = 405;
        response.error_message = "Method not allowed";
        response.headers["X-Request-Id"] =
            request.request_id.empty() ? generate_request_id()
                                       : request.request_id;
        return response;
    }

    auto route = parse_path(request.path);
    if (!route.valid) {
        Response response;
        response.status_code = 404;
        response.error_message = "Invalid path: " + request.path;
        response.headers["X-Request-Id"] =
            request.request_id.empty() ? generate_request_id()
                                       : request.request_id;
        return response;
    }

    // If a specific version is in the URL, use it directly
    if (route.version > 0) {
        // Temporarily bypass traffic splitting by directly finding version
        auto start = std::chrono::steady_clock::now();
        std::string req_id = request.request_id.empty()
            ? generate_request_id() : request.request_id;

        Response response;
        response.headers["X-Request-Id"] = req_id;

        if (!impl_->rate_limiter.try_acquire(request.tenant_id)) {
            response.status_code = 429;
            response.error_message = "Quota exceeded for tenant '" +
                                     request.tenant_id + "'";
            return response;
        }

        struct QuotaGuard {
            RateLimiter& rl;
            const std::string& tid;
            ~QuotaGuard() { rl.release(tid); }
        } guard{impl_->rate_limiter, request.tenant_id};

        try {
            ModelVersionInfo info =
                impl_->find_version(route.model_name, route.version);
            auto pool = impl_->cache->get_or_load(
                info, [this](const CacheKey& k) {
                    return impl_->is_pinned(k);
                });
            auto lease = pool->acquire();
            Tensor output = lease.engine().predict(request.body);

            auto end = std::chrono::steady_clock::now();
            response.status_code = 200;
            response.body = std::move(output);
            response.latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
            response.headers["X-Model-Version"] =
                std::to_string(info.version);
        } catch (const ServerException& e) {
            response.status_code = 404;
            response.error_message = e.what();
        } catch (const ValidationException& e) {
            response.status_code = 400;
            response.error_message = e.what();
        } catch (const std::invalid_argument& e) {
            response.status_code = 400;
            response.error_message = e.what();
        } catch (const std::exception& e) {
            response.status_code = 500;
            response.error_message = e.what();
        }

        return response;
    }

    return impl_->do_predict(route.model_name, request.body,
                             request.tenant_id, request.request_id);
}

std::future<Response> ModelServer::handle_request_async(
    const Request& request)
{
    Request req_copy = request;
    // Deep-copy the tensor body
    req_copy.body = Tensor(request.body);

    return impl_->thread_pool->submit(
        [this, req_copy]() mutable {
            return handle_request(req_copy);
        });
}

// ---- Hot Reload ----

void ModelServer::reload_model(const std::string& name, uint32_t version,
                                const std::string& new_file_path) {
    // Update registry with new path
    {
        std::unique_lock<std::shared_mutex> lock(impl_->registry_mutex);
        auto name_it = impl_->registry.find(name);
        if (name_it == impl_->registry.end()) {
            throw ServerException("Model '" + name + "' not found",
                                  ErrorCode::MODEL_NOT_FOUND);
        }
        auto ver_it = name_it->second.find(version);
        if (ver_it == name_it->second.end()) {
            throw ServerException("Version " + std::to_string(version) +
                                  " not found", ErrorCode::VERSION_NOT_FOUND);
        }
        ver_it->second.file_path = new_file_path;
    }

    TITANINFER_LOG_INFO("Hot-reloading model '" + name + "' v" +
                        std::to_string(version) + " from " + new_file_path);

    // Build new pool in background, then atomically swap
    size_t pool_size = impl_->config.engines_per_model > 0
        ? impl_->config.engines_per_model
        : impl_->thread_pool->thread_count();
    bool profiling = impl_->config.enable_profiling;
    CacheKey key{name, version};

    // Load synchronously for simplicity and testability.
    // The old pool remains alive via shared_ptr until all Leases complete.
    auto new_pool = std::make_shared<EnginePool>(
        new_file_path, pool_size, profiling);
    impl_->cache->replace(key, std::move(new_pool));

    TITANINFER_LOG_INFO("Hot-reload complete for '" + name + "' v" +
                        std::to_string(version));
}

// ---- Stats ----

size_t ModelServer::loaded_model_count() const {
    return impl_->cache->loaded_count();
}

size_t ModelServer::registered_model_count() const {
    std::shared_lock<std::shared_mutex> lock(impl_->registry_mutex);
    size_t count = 0;
    for (const auto& pair : impl_->registry) {
        count += pair.second.size();
    }
    return count;
}

} // namespace titaninfer::engine
