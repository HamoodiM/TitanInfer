#include "titaninfer/engine/cluster_controller.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace titaninfer::engine {

// ===========================================================================
// Anonymous namespace — internal helpers and CircuitBreaker
// ===========================================================================
namespace {

// Generate a 128-bit trace ID as 32 hex chars (OTEL-compatible).
std::string generate_trace_id() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t hi = dist(rng);
    uint64_t lo = dist(rng);
    char buf[33];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                  static_cast<unsigned long long>(hi),
                  static_cast<unsigned long long>(lo));
    return std::string(buf);
}

// Weighted load score for a node×model pair. Lower = less loaded.
double load_score(const NodeMetrics& m, double slo_p95_target) {
    double latency_ratio =
        (slo_p95_target > 0.0) ? m.p95_latency_ms / slo_p95_target : 0.0;
    return m.queue_depth * 0.4 + m.cpu_util * 0.3 + latency_ratio * 0.3;
}

// ---------------------------------------------------------------------------
// CircuitBreaker — sliding-window failure detector per node×model
// ---------------------------------------------------------------------------
struct CircuitBreaker {
    enum class State { Closed, Open, HalfOpen };

    State  state            = State::Closed;
    bool   probe_in_flight  = false;
    std::chrono::steady_clock::time_point open_since;
    std::deque<std::chrono::steady_clock::time_point> failure_timestamps;

    // Returns true if the circuit should BLOCK this request.
    // May transition Open → HalfOpen if the window has expired.
    bool should_block(size_t threshold, size_t window_ms,
                      std::chrono::steady_clock::time_point now) {
        if (state == State::Closed) return false;

        if (state == State::Open) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - open_since)
                    .count();
            if (static_cast<size_t>(elapsed) >= window_ms) {
                state          = State::HalfOpen;
                probe_in_flight = false;
            }
        }

        if (state == State::HalfOpen) {
            if (probe_in_flight) return true; // already probing
            probe_in_flight = true;           // allow exactly one probe
            return false;
        }

        return true; // state == Open (window not expired)
    }

    void on_failure(size_t threshold, size_t window_ms,
                    std::chrono::steady_clock::time_point now) {
        // Evict timestamps outside the window
        while (!failure_timestamps.empty()) {
            auto age =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - failure_timestamps.front())
                    .count();
            if (static_cast<size_t>(age) > window_ms)
                failure_timestamps.pop_front();
            else
                break;
        }
        failure_timestamps.push_back(now);
        if (failure_timestamps.size() >= threshold) {
            state      = State::Open;
            open_since = now;
        }
    }

    void on_success() {
        state          = State::Closed;
        probe_in_flight = false;
        failure_timestamps.clear();
    }
};

} // anonymous namespace

// ===========================================================================
// Impl
// ===========================================================================
struct ClusterController::Impl {
    ClusterControllerConfig config;

    // ---- CRDT-style cluster state (lock-free reads, single-writer) ----

    struct ClusterState {
        std::unordered_map<std::string, NodeInfo> nodes;
        uint64_t version = 0;
    };

    std::atomic<std::shared_ptr<ClusterState>> state{
        std::make_shared<ClusterState>()};
    std::mutex write_mutex; // guards all state writes

    // ---- Circuit breakers: node_id → model_name → CircuitBreaker ----

    std::unordered_map<std::string,
                       std::unordered_map<std::string, CircuitBreaker>>
        circuit_breakers;
    std::mutex cb_mutex;

    // ---- Background failure-detection thread (C++20 jthread) ----

    std::jthread heartbeat_monitor;

    // ---- Prometheus counters ----

    std::atomic<uint64_t> total_scheduled{0};
    std::atomic<uint64_t> total_rejected{0};

    explicit Impl(const ClusterControllerConfig& cfg) : config(cfg) {}
};

// ===========================================================================
// Builder
// ===========================================================================

ClusterController::Builder&
ClusterController::Builder::setHeartbeatTimeoutMs(size_t ms) {
    config_.heartbeat_timeout_ms = ms;
    return *this;
}

ClusterController::Builder&
ClusterController::Builder::setHeartbeatIntervalMs(size_t ms) {
    config_.heartbeat_interval_ms = ms;
    return *this;
}

ClusterController::Builder&
ClusterController::Builder::setSloP95TargetMs(double ms) {
    config_.slo_p95_target_ms = ms;
    return *this;
}

ClusterController::Builder&
ClusterController::Builder::setSloQueueDepthMax(double depth) {
    config_.slo_queue_depth_max = depth;
    return *this;
}

ClusterController::Builder&
ClusterController::Builder::setCircuitBreakerFailures(size_t count) {
    config_.circuit_breaker_failures = count;
    return *this;
}

ClusterController::Builder&
ClusterController::Builder::setCircuitBreakerWindowMs(size_t ms) {
    config_.circuit_breaker_window_ms = ms;
    return *this;
}

ClusterController ClusterController::Builder::build() {
    return ClusterController{config_};
}

// ===========================================================================
// Construction / destruction
// ===========================================================================

ClusterController::ClusterController(const ClusterControllerConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {
    // Capture raw pointer — safe because impl_ owns the Impl and the jthread
    // lives inside impl_, so the Impl always outlives the jthread lambda.
    Impl* impl_ptr = impl_.get();

    impl_->heartbeat_monitor = std::jthread([impl_ptr](std::stop_token st) {
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                impl_ptr->config.heartbeat_interval_ms));
            if (st.stop_requested()) break;

            auto now     = std::chrono::steady_clock::now();
            auto current = impl_ptr->state.load();

            // Build updated state (COW)
            auto next        = std::make_shared<Impl::ClusterState>(*current);
            bool any_changed = false;

            for (auto& [id, node] : next->nodes) {
                if (node.model_metrics.empty()) continue;

                // Latest heartbeat timestamp across all models
                auto latest = node.model_metrics.begin()->second.timestamp;
                for (auto& [m, metrics] : node.model_metrics) {
                    if (metrics.timestamp > latest) latest = metrics.timestamp;
                }

                auto age =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - latest)
                        .count();
                bool should_be_healthy = static_cast<size_t>(age) <
                                         impl_ptr->config.heartbeat_timeout_ms;

                if (node.healthy != should_be_healthy) {
                    node.healthy = should_be_healthy;
                    any_changed  = true;
                }
            }

            if (any_changed) {
                next->version = current->version + 1;
                std::lock_guard<std::mutex> lk(impl_ptr->write_mutex);
                impl_ptr->state.store(next);
            }
        }
    });
}

ClusterController::~ClusterController() = default;
ClusterController::ClusterController(ClusterController&&) noexcept = default;
ClusterController& ClusterController::operator=(ClusterController&&) noexcept =
    default;

// ===========================================================================
// Node Lifecycle
// ===========================================================================

void ClusterController::register_node(const NodeInfo& info) {
    std::lock_guard<std::mutex> lk(impl_->write_mutex);
    auto current    = impl_->state.load();
    auto next       = std::make_shared<Impl::ClusterState>(*current);
    next->nodes[info.node_id] = info;
    next->version   = current->version + 1;
    impl_->state.store(next);
}

void ClusterController::heartbeat(
    const std::string& node_id,
    const std::unordered_map<std::string, NodeMetrics>& metrics) {
    std::lock_guard<std::mutex> lk(impl_->write_mutex);
    auto current = impl_->state.load();
    if (current->nodes.find(node_id) == current->nodes.end()) return;

    auto next         = std::make_shared<Impl::ClusterState>(*current);
    auto& node        = next->nodes[node_id];
    auto  now         = std::chrono::steady_clock::now();

    for (auto& [model, m] : metrics) {
        auto updated      = m;
        updated.timestamp = now;
        node.model_metrics[model] = updated;
    }
    node.healthy  = true;
    next->version = current->version + 1;
    impl_->state.store(next);
}

void ClusterController::deregister_node(const std::string& node_id) {
    {
        std::lock_guard<std::mutex> lk(impl_->write_mutex);
        auto current = impl_->state.load();
        if (current->nodes.find(node_id) == current->nodes.end()) return;
        auto next    = std::make_shared<Impl::ClusterState>(*current);
        next->nodes.erase(node_id);
        next->version = current->version + 1;
        impl_->state.store(next);
    }
    // Clean up circuit breakers
    std::lock_guard<std::mutex> cb_lk(impl_->cb_mutex);
    impl_->circuit_breakers.erase(node_id);
}

// ===========================================================================
// Scheduling — power-of-two choices + tenant-affinity tiebreaking
// ===========================================================================

RoutingDecision ClusterController::schedule(const std::string& model_name,
                                             const std::string& tenant_id) {
    impl_->total_scheduled.fetch_add(1, std::memory_order_relaxed);

    auto state = impl_->state.load();
    auto now   = std::chrono::steady_clock::now();

    struct Candidate {
        std::string node_id;
        double      score;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(state->nodes.size());

    {
        std::lock_guard<std::mutex> cb_lk(impl_->cb_mutex);
        for (auto& [id, node] : state->nodes) {
            if (!node.healthy) continue;

            auto it = node.model_metrics.find(model_name);
            if (it == node.model_metrics.end()) continue;

            auto& cb = impl_->circuit_breakers[id][model_name];
            if (cb.should_block(impl_->config.circuit_breaker_failures,
                                impl_->config.circuit_breaker_window_ms, now)) {
                continue;
            }

            candidates.push_back(
                {id, load_score(it->second, impl_->config.slo_p95_target_ms)});
        }
    }

    if (candidates.empty()) {
        impl_->total_rejected.fetch_add(1, std::memory_order_relaxed);
        return RoutingDecision{
            "", 0.0, generate_trace_id(),
            "no healthy node available for model '" + model_name + "'"};
    }

    RoutingDecision decision;
    decision.trace_id = generate_trace_id();

    if (candidates.size() == 1) {
        decision.target_node_id  = candidates[0].node_id;
        decision.confidence_score = 1.0;
        return decision;
    }

    // Power-of-two random choices
    thread_local std::mt19937_64 rng{std::random_device{}()};

    size_t idx1 = 0, idx2 = 1;
    if (candidates.size() > 2) {
        std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
        idx1 = dist(rng);
        do { idx2 = dist(rng); } while (idx2 == idx1);
    }

    // Tenant affinity: if tenant is given, use its hash to prefer a candidate
    // when both are within 20% of each other
    if (!tenant_id.empty()) {
        size_t h = std::hash<std::string>{}(tenant_id);
        bool prefer_first = (h % 2 == 0);
        if (!prefer_first) std::swap(idx1, idx2);
        // Keep swap only if scores are close (affinity tiebreaker)
        if (candidates[idx1].score > candidates[idx2].score * 1.2)
            std::swap(idx1, idx2); // second is clearly better; override affinity
    }

    const auto& chosen =
        (candidates[idx1].score <= candidates[idx2].score) ? candidates[idx1]
                                                            : candidates[idx2];

    decision.target_node_id = chosen.node_id;

    double max_score =
        std::max(candidates[idx1].score, candidates[idx2].score);
    decision.confidence_score =
        (max_score > 0.0) ? 1.0 - chosen.score / max_score : 1.0;
    decision.confidence_score =
        std::clamp(decision.confidence_score, 0.0, 1.0);

    return decision;
}

// ===========================================================================
// Autoscaling
// ===========================================================================

std::vector<ScaleRecommendation>
ClusterController::compute_scale_signals() const {
    auto state = impl_->state.load();

    struct ModelAgg {
        double sum_queue    = 0.0;
        double sum_p95      = 0.0;
        size_t count        = 0;
    };
    std::unordered_map<std::string, ModelAgg> agg;

    for (auto& [id, node] : state->nodes) {
        if (!node.healthy) continue;
        for (auto& [model, metrics] : node.model_metrics) {
            auto& a = agg[model];
            a.sum_queue += metrics.queue_depth;
            a.sum_p95   += metrics.p95_latency_ms;
            ++a.count;
        }
    }

    std::vector<ScaleRecommendation> recs;
    for (auto& [model, a] : agg) {
        if (a.count == 0) continue;
        double avg_queue = a.sum_queue / static_cast<double>(a.count);
        double avg_p95   = a.sum_p95   / static_cast<double>(a.count);

        if (avg_queue > impl_->config.slo_queue_depth_max * 0.8) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "avg queue_depth %.2f exceeds 80%% of "
                          "slo_queue_depth_max (%.1f)",
                          avg_queue, impl_->config.slo_queue_depth_max);
            recs.push_back({model, +1, std::string(buf), avg_queue});
        } else if (avg_p95 > impl_->config.slo_p95_target_ms) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "avg p95_latency %.2fms exceeds SLO target %.1fms",
                          avg_p95, impl_->config.slo_p95_target_ms);
            recs.push_back({model, +1, std::string(buf), avg_p95});
        } else if (avg_queue < impl_->config.slo_queue_depth_max * 0.1 &&
                   a.count > 1) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "avg queue_depth %.2f below 10%% of "
                          "slo_queue_depth_max (%.1f) across %zu nodes",
                          avg_queue, impl_->config.slo_queue_depth_max, a.count);
            recs.push_back({model, -1, std::string(buf), avg_queue});
        }
    }

    return recs;
}

// ===========================================================================
// Circuit Breaker Feedback
// ===========================================================================

void ClusterController::record_success(const std::string& node_id,
                                        const std::string& model_name) {
    std::lock_guard<std::mutex> lk(impl_->cb_mutex);
    impl_->circuit_breakers[node_id][model_name].on_success();
}

void ClusterController::record_failure(const std::string& node_id,
                                        const std::string& model_name) {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(impl_->cb_mutex);
    impl_->circuit_breakers[node_id][model_name].on_failure(
        impl_->config.circuit_breaker_failures,
        impl_->config.circuit_breaker_window_ms, now);
}

// ===========================================================================
// Observability
// ===========================================================================

std::string ClusterController::metrics_text() const {
    auto state = impl_->state.load();

    size_t healthy = 0;
    for (auto& [id, node] : state->nodes)
        if (node.healthy) ++healthy;

    std::ostringstream out;

    out << "# HELP cluster_healthy_nodes Number of healthy nodes\n"
           "# TYPE cluster_healthy_nodes gauge\n"
        << "cluster_healthy_nodes " << healthy << "\n\n";

    out << "# HELP cluster_total_nodes Total registered nodes\n"
           "# TYPE cluster_total_nodes gauge\n"
        << "cluster_total_nodes " << state->nodes.size() << "\n\n";

    out << "# HELP cluster_requests_total Total scheduling requests\n"
           "# TYPE cluster_requests_total counter\n"
        << "cluster_requests_total "
        << impl_->total_scheduled.load(std::memory_order_relaxed) << "\n\n";

    out << "# HELP cluster_requests_rejected_total Rejected scheduling requests\n"
           "# TYPE cluster_requests_rejected_total counter\n"
        << "cluster_requests_rejected_total "
        << impl_->total_rejected.load(std::memory_order_relaxed) << "\n\n";

    out << "# HELP node_queue_depth Queue depth per node and model\n"
           "# TYPE node_queue_depth gauge\n";
    for (auto& [id, node] : state->nodes)
        for (auto& [model, m] : node.model_metrics)
            out << "node_queue_depth{node=\"" << id << "\",model=\"" << model
                << "\"} " << m.queue_depth << "\n";
    out << "\n";

    out << "# HELP node_p95_latency_ms P95 latency per node and model\n"
           "# TYPE node_p95_latency_ms gauge\n";
    for (auto& [id, node] : state->nodes)
        for (auto& [model, m] : node.model_metrics)
            out << "node_p95_latency_ms{node=\"" << id << "\",model=\"" << model
                << "\"} " << m.p95_latency_ms << "\n";

    return out.str();
}

std::vector<NodeInfo> ClusterController::list_nodes() const {
    auto state = impl_->state.load();
    std::vector<NodeInfo> result;
    result.reserve(state->nodes.size());
    for (auto& [id, node] : state->nodes)
        result.push_back(node);
    return result;
}

size_t ClusterController::healthy_node_count() const {
    auto state = impl_->state.load();
    size_t count = 0;
    for (auto& [id, node] : state->nodes)
        if (node.healthy) ++count;
    return count;
}

} // namespace titaninfer::engine
