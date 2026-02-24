#pragma once

#include <future>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "titaninfer/tensor.hpp"

namespace titaninfer::engine {

// ---------------------------------------------------------------------------
// HTTP-style request/response types
// ---------------------------------------------------------------------------

enum class HttpMethod { GET, POST, DELETE_METHOD };

struct Request {
    std::string request_id;
    HttpMethod method = HttpMethod::POST;
    std::string path;
    std::string tenant_id;
    Tensor body{std::vector<size_t>{1}};
    std::unordered_map<std::string, std::string> headers;
};

struct Response {
    int status_code = 200;
    Tensor body{std::vector<size_t>{1}};
    std::string error_message;
    std::unordered_map<std::string, std::string> headers;
    double latency_ms = 0.0;
};

// ---------------------------------------------------------------------------
// Model registry types
// ---------------------------------------------------------------------------

struct ModelVersionInfo {
    std::string name;
    uint32_t version = 0;
    std::string file_path;
    std::set<std::string> tags;
    bool pinned = false;
};

struct TenantQuota {
    double max_qps = 100.0;
    size_t max_concurrent = 10;
};

struct TrafficRule {
    uint32_t version = 0;
    double weight = 1.0;
};

struct ModelServerConfig {
    size_t max_loaded_models = 16;
    size_t worker_threads = 0;       // 0 = hardware_concurrency
    size_t engines_per_model = 0;    // 0 = worker_threads count
    bool enable_profiling = false;
};

// ---------------------------------------------------------------------------
// ModelServer
// ---------------------------------------------------------------------------

class ModelServer {
public:
    class Builder {
    public:
        Builder() = default;

        Builder& setMaxLoadedModels(size_t count);
        Builder& setWorkerThreads(size_t count);
        Builder& setEnginesPerModel(size_t count);
        Builder& enableProfiling(bool enable = true);

        ModelServer build();

    private:
        ModelServerConfig config_;
    };

    ~ModelServer();
    ModelServer(const ModelServer&) = delete;
    ModelServer& operator=(const ModelServer&) = delete;
    ModelServer(ModelServer&&) noexcept;
    ModelServer& operator=(ModelServer&&) noexcept;

    // ---- Model Registry ----

    void register_model(const std::string& name,
                        uint32_t version,
                        const std::string& file_path,
                        const std::set<std::string>& tags = {});

    void unregister_model(const std::string& name, uint32_t version);

    void add_tag(const std::string& name, uint32_t version,
                 const std::string& tag);
    void remove_tag(const std::string& name, uint32_t version,
                    const std::string& tag);

    void pin_model(const std::string& name, uint32_t version);
    void unpin_model(const std::string& name, uint32_t version);

    std::vector<ModelVersionInfo> list_versions(const std::string& name) const;

    // ---- Quotas ----

    void set_tenant_quota(const std::string& tenant_id,
                          const TenantQuota& quota);
    void remove_tenant_quota(const std::string& tenant_id);

    // ---- Traffic Splitting ----

    void set_traffic_rules(const std::string& model_name,
                           const std::vector<TrafficRule>& rules);

    // ---- Inference ----

    Response predict(const std::string& model_name,
                     const Tensor& input,
                     const std::string& tenant_id = "",
                     const std::string& request_id = "");

    std::future<Response> predict_async(const std::string& model_name,
                                        const Tensor& input,
                                        const std::string& tenant_id = "",
                                        const std::string& request_id = "");

    Response handle_request(const Request& request);
    std::future<Response> handle_request_async(const Request& request);

    // ---- Hot Reload ----

    void reload_model(const std::string& name, uint32_t version,
                      const std::string& new_file_path);

    // ---- Stats ----

    size_t loaded_model_count() const;
    size_t registered_model_count() const;

private:
    explicit ModelServer(const ModelServerConfig& config);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace titaninfer::engine
