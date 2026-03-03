#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace titaninfer::engine {

// ---------------------------------------------------------------------------
// Metrics reported by each node for each model it serves (via heartbeat)
// ---------------------------------------------------------------------------

struct NodeMetrics {
    double queue_depth     = 0.0;  ///< Pending requests
    double p95_latency_ms  = 0.0;  ///< 95th-percentile latency (ms)
    double cpu_util        = 0.0;  ///< CPU utilization [0, 1]
    double memory_pressure = 0.0;  ///< Memory utilization [0, 1]
    size_t active_requests = 0;    ///< In-flight request count
    std::chrono::steady_clock::time_point timestamp;
};

// ---------------------------------------------------------------------------
// Node registration / state
// ---------------------------------------------------------------------------

struct NodeInfo {
    std::string node_id;
    std::string address;  ///< "host:port" for external routing
    std::unordered_map<std::string, NodeMetrics> model_metrics;
    bool healthy = true;
};

// ---------------------------------------------------------------------------
// Scheduling result
// ---------------------------------------------------------------------------

struct RoutingDecision {
    std::string target_node_id;
    double confidence_score = 0.0;    ///< [0,1] routing quality signal
    std::string trace_id;             ///< 128-bit hex (OTEL-compatible)
    std::optional<std::string> error; ///< Backpressure reason (empty = success)
};

// ---------------------------------------------------------------------------
// Autoscaling recommendation
// ---------------------------------------------------------------------------

struct ScaleRecommendation {
    std::string model_name;
    int delta_replicas    = 0;   ///< +N = scale up, -N = scale down
    std::string justification;
    double trigger_metric = 0.0; ///< The metric that crossed the SLO
};

// ---------------------------------------------------------------------------
// Controller configuration
// ---------------------------------------------------------------------------

struct ClusterControllerConfig {
    size_t heartbeat_timeout_ms      = 5000;  ///< Silence → node unhealthy
    size_t heartbeat_interval_ms     = 2000;  ///< Monitor poll period
    double slo_p95_target_ms         = 100.0; ///< SLO: p95 latency target
    double slo_queue_depth_max       = 8.0;   ///< SLO: max queue depth
    size_t circuit_breaker_failures  = 5;     ///< Failures before open
    size_t circuit_breaker_window_ms = 30000; ///< Failure counting window
};

// ---------------------------------------------------------------------------
// ClusterController
//
// Thread-safe. All public methods may be called concurrently.
// CRDT-style lock-free reads (atomic<shared_ptr<ClusterState>>);
// single-writer mutex for state updates.
// Background std::jthread performs heartbeat-timeout failure detection.
// ---------------------------------------------------------------------------

class ClusterController {
public:
    class Builder {
    public:
        Builder() = default;

        Builder& setHeartbeatTimeoutMs(size_t ms);
        Builder& setHeartbeatIntervalMs(size_t ms);
        Builder& setSloP95TargetMs(double ms);
        Builder& setSloQueueDepthMax(double depth);
        Builder& setCircuitBreakerFailures(size_t count);
        Builder& setCircuitBreakerWindowMs(size_t ms);

        ClusterController build();

    private:
        ClusterControllerConfig config_;
    };

    ~ClusterController();
    ClusterController(ClusterController&&) noexcept;
    ClusterController& operator=(ClusterController&&) noexcept;
    ClusterController(const ClusterController&) = delete;
    ClusterController& operator=(const ClusterController&) = delete;

    // ---- Node Lifecycle ------------------------------------------------

    /// Register a new node. Idempotent: re-registering updates the record.
    void register_node(const NodeInfo& info);

    /// Update per-model metrics for an existing node and mark it healthy.
    /// Silently ignored if node_id is unknown.
    void heartbeat(const std::string& node_id,
                   const std::unordered_map<std::string, NodeMetrics>& metrics);

    /// Remove a node from the registry and clear its circuit breakers.
    void deregister_node(const std::string& node_id);

    // ---- Scheduling ----------------------------------------------------

    /// Power-of-two random choices with tenant-affinity tiebreaking.
    /// Returns a RoutingDecision; decision.error is set on failure.
    RoutingDecision schedule(const std::string& model_name,
                             const std::string& tenant_id = "");

    // ---- Autoscaling ---------------------------------------------------

    /// Compute SLO-violation driven scale signals for all registered models.
    std::vector<ScaleRecommendation> compute_scale_signals() const;

    // ---- Circuit Breaker Feedback --------------------------------------

    void record_success(const std::string& node_id,
                        const std::string& model_name);
    void record_failure(const std::string& node_id,
                        const std::string& model_name);

    // ---- Observability -------------------------------------------------

    /// Prometheus text exposition format (/metrics endpoint body).
    std::string metrics_text() const;

    std::vector<NodeInfo> list_nodes() const;
    size_t healthy_node_count() const;

private:
    explicit ClusterController(const ClusterControllerConfig& config);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace titaninfer::engine
