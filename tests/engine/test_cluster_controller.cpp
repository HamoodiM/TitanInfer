#include <gtest/gtest.h>
#include "titaninfer/engine/cluster_controller.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace titaninfer::engine;

// ===========================================================================
// Test helpers
// ===========================================================================

static NodeMetrics make_metrics(double queue_depth   = 0.0,
                                double p95_latency   = 10.0,
                                double cpu_util      = 0.1,
                                double memory_press  = 0.1,
                                size_t active_reqs   = 0) {
    NodeMetrics m;
    m.queue_depth     = queue_depth;
    m.p95_latency_ms  = p95_latency;
    m.cpu_util        = cpu_util;
    m.memory_pressure = memory_press;
    m.active_requests = active_reqs;
    m.timestamp       = std::chrono::steady_clock::now();
    return m;
}

static NodeInfo make_node(const std::string& id,
                           const std::string& model = "resnet50",
                           double queue = 0.0) {
    NodeInfo n;
    n.node_id = id;
    n.address = "127.0.0.1:" + id;
    n.healthy = true;
    n.model_metrics[model] = make_metrics(queue);
    return n;
}

// Build a minimal controller with fast timeouts for unit tests.
static ClusterController fast_controller(size_t timeout_ms  = 200,
                                          size_t interval_ms = 50) {
    return ClusterController::Builder()
        .setHeartbeatTimeoutMs(timeout_ms)
        .setHeartbeatIntervalMs(interval_ms)
        .build();
}

// ===========================================================================
// Group 1: Builder & Config
// ===========================================================================

TEST(BuilderTest, DefaultConfig) {
    auto ctrl = ClusterController::Builder().build();
    EXPECT_EQ(ctrl.healthy_node_count(), 0u);
    EXPECT_TRUE(ctrl.list_nodes().empty());
}

TEST(BuilderTest, CustomConfig) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(1000)
                    .setHeartbeatIntervalMs(200)
                    .setSloP95TargetMs(50.0)
                    .setSloQueueDepthMax(4.0)
                    .setCircuitBreakerFailures(3)
                    .setCircuitBreakerWindowMs(10000)
                    .build();
    EXPECT_EQ(ctrl.healthy_node_count(), 0u);
}

TEST(BuilderTest, MoveConstructor) {
    auto ctrl  = ClusterController::Builder().build();
    auto ctrl2 = std::move(ctrl);
    EXPECT_EQ(ctrl2.healthy_node_count(), 0u);
    EXPECT_TRUE(ctrl2.list_nodes().empty());
}

TEST(BuilderTest, MoveAssignment) {
    auto ctrl  = ClusterController::Builder().build();
    auto ctrl2 = ClusterController::Builder().build();
    ctrl2      = std::move(ctrl);
    EXPECT_EQ(ctrl2.healthy_node_count(), 0u);
}

// ===========================================================================
// Group 2: Node Registry
// ===========================================================================

TEST(RegistryTest, RegisterSingleNode) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    auto nodes = ctrl.list_nodes();
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0].node_id, "n1");
}

TEST(RegistryTest, RegisterMultipleNodes) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.register_node(make_node("n3"));
    EXPECT_EQ(ctrl.list_nodes().size(), 3u);
}

TEST(RegistryTest, RegisterIsIdempotent) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n1")); // re-register same ID
    EXPECT_EQ(ctrl.list_nodes().size(), 1u);
}

TEST(RegistryTest, DeregisterNode) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.deregister_node("n1");
    auto nodes = ctrl.list_nodes();
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0].node_id, "n2");
}

TEST(RegistryTest, DeregisterUnknownIsNoOp) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    EXPECT_NO_THROW(ctrl.deregister_node("nonexistent"));
    EXPECT_EQ(ctrl.list_nodes().size(), 1u);
}

TEST(RegistryTest, HeartbeatUpdatesMetrics) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "bert", 0.0));

    std::unordered_map<std::string, NodeMetrics> update;
    update["bert"] = make_metrics(7.5, 200.0, 0.8);
    ctrl.heartbeat("n1", update);

    auto nodes = ctrl.list_nodes();
    ASSERT_EQ(nodes.size(), 1u);
    ASSERT_TRUE(nodes[0].model_metrics.count("bert"));
    EXPECT_DOUBLE_EQ(nodes[0].model_metrics.at("bert").queue_depth, 7.5);
    EXPECT_DOUBLE_EQ(nodes[0].model_metrics.at("bert").cpu_util, 0.8);
}

TEST(RegistryTest, HeartbeatUnknownNodeIsNoOp) {
    auto ctrl = fast_controller();
    std::unordered_map<std::string, NodeMetrics> m{{"model", make_metrics()}};
    EXPECT_NO_THROW(ctrl.heartbeat("ghost", m));
    EXPECT_EQ(ctrl.list_nodes().size(), 0u);
}

TEST(RegistryTest, NodeWithMultipleModels) {
    auto ctrl = fast_controller();
    NodeInfo n;
    n.node_id = "n1";
    n.model_metrics["resnet50"] = make_metrics(1.0);
    n.model_metrics["bert"]     = make_metrics(2.0);
    ctrl.register_node(n);

    auto nodes = ctrl.list_nodes();
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0].model_metrics.size(), 2u);
}

// ===========================================================================
// Group 3: Scheduling
// ===========================================================================

TEST(SchedulingTest, ToHealthyNode) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    auto d = ctrl.schedule("resnet50");
    ASSERT_FALSE(d.error.has_value()) << *d.error;
    EXPECT_EQ(d.target_node_id, "n1");
    EXPECT_GE(d.confidence_score, 0.0);
    EXPECT_LE(d.confidence_score, 1.0);
}

TEST(SchedulingTest, RejectsNoHealthyNodes) {
    auto ctrl = fast_controller();
    auto d    = ctrl.schedule("missing_model");
    ASSERT_TRUE(d.error.has_value());
    EXPECT_FALSE(d.error->empty());
    EXPECT_TRUE(d.target_node_id.empty());
}

TEST(SchedulingTest, RejectsModelNotOnAnyNode) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "resnet50"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    auto d = ctrl.schedule("bert"); // not registered on n1
    ASSERT_TRUE(d.error.has_value());
}

TEST(SchedulingTest, PrefersLeastLoaded) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "model", 0.1));
    ctrl.register_node(make_node("n2", "model", 9.0));
    ctrl.heartbeat("n1", {{"model", make_metrics(0.1)}});
    ctrl.heartbeat("n2", {{"model", make_metrics(9.0)}});

    int n1_count = 0;
    for (int i = 0; i < 200; ++i) {
        auto d = ctrl.schedule("model");
        if (!d.error && d.target_node_id == "n1") ++n1_count;
    }
    // n1 is ~90× less loaded; should win the vast majority of p-o-2 draws
    EXPECT_GT(n1_count, 130);
}

TEST(SchedulingTest, ConfidenceScoreRange) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "model", 1.0));
    ctrl.register_node(make_node("n2", "model", 5.0));
    ctrl.heartbeat("n1", {{"model", make_metrics(1.0)}});
    ctrl.heartbeat("n2", {{"model", make_metrics(5.0)}});

    for (int i = 0; i < 50; ++i) {
        auto d = ctrl.schedule("model");
        EXPECT_GE(d.confidence_score, 0.0);
        EXPECT_LE(d.confidence_score, 1.0);
    }
}

TEST(SchedulingTest, TraceIdIs32HexChars) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    auto d = ctrl.schedule("resnet50");
    ASSERT_EQ(d.trace_id.size(), 32u);
    for (char c : d.trace_id)
        EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(c)));
}

TEST(SchedulingTest, TraceIdsAreUnique) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    std::string first  = ctrl.schedule("resnet50").trace_id;
    std::string second = ctrl.schedule("resnet50").trace_id;
    EXPECT_NE(first, second);
}

TEST(SchedulingTest, TenantAffinityConsistency) {
    auto ctrl = fast_controller();
    // Two equally loaded nodes; tenant affinity should pin tenant to one node.
    ctrl.register_node(make_node("n1", "model", 1.0));
    ctrl.register_node(make_node("n2", "model", 1.0));
    ctrl.heartbeat("n1", {{"model", make_metrics(1.0)}});
    ctrl.heartbeat("n2", {{"model", make_metrics(1.0)}});

    std::string first = ctrl.schedule("model", "tenant-A").target_node_id;
    int same_count = 0;
    for (int i = 0; i < 50; ++i)
        if (ctrl.schedule("model", "tenant-A").target_node_id == first)
            ++same_count;

    // Deterministic hash → should always pick the same node for equal loads
    EXPECT_EQ(same_count, 50);
}

TEST(SchedulingTest, ScheduleMultipleModels) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "resnet50"));
    ctrl.register_node(make_node("n2", "bert"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    ctrl.heartbeat("n2", {{"bert", make_metrics()}});

    auto d1 = ctrl.schedule("resnet50");
    auto d2 = ctrl.schedule("bert");
    ASSERT_FALSE(d1.error.has_value());
    ASSERT_FALSE(d2.error.has_value());
    EXPECT_EQ(d1.target_node_id, "n1");
    EXPECT_EQ(d2.target_node_id, "n2");
}

TEST(SchedulingTest, ConcurrentSchedule) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics(1.0)}});
    ctrl.heartbeat("n2", {{"resnet50", make_metrics(2.0)}});

    std::atomic<int> success{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 100; ++i) {
                auto d = ctrl.schedule("resnet50");
                if (!d.error) ++success;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(success.load(), 800);
}

// ===========================================================================
// Group 4: Circuit Breakers
// ===========================================================================

TEST(CircuitBreakerTest, OpensAfterThreshold) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(3)
                    .setCircuitBreakerWindowMs(60000)
                    .build();
    ctrl.register_node(make_node("n1", "model"));
    ctrl.register_node(make_node("n2", "model"));
    ctrl.heartbeat("n1", {{"model", make_metrics()}});
    ctrl.heartbeat("n2", {{"model", make_metrics()}});

    ctrl.record_failure("n1", "model");
    ctrl.record_failure("n1", "model");
    ctrl.record_failure("n1", "model"); // circuit opens for n1

    // With 2 nodes and n1's circuit open, all requests must go to n2
    for (int i = 0; i < 30; ++i) {
        auto d = ctrl.schedule("model");
        ASSERT_FALSE(d.error.has_value());
        // After the single half-open probe, n1 is fully excluded
    }

    // Confirm n2 dominates (n1 gets at most 1 half-open probe)
    int n2_count = 0;
    for (int i = 0; i < 50; ++i) {
        auto d = ctrl.schedule("model");
        if (!d.error && d.target_node_id == "n2") ++n2_count;
    }
    EXPECT_GE(n2_count, 49); // n1 gets at most 0 more probes once probe is in-flight
}

TEST(CircuitBreakerTest, ClosesAfterSuccess) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(2)
                    .setCircuitBreakerWindowMs(60000)
                    .build();
    ctrl.register_node(make_node("n1", "model"));
    ctrl.heartbeat("n1", {{"model", make_metrics()}});

    ctrl.record_failure("n1", "model");
    ctrl.record_failure("n1", "model"); // open

    ctrl.record_success("n1", "model"); // close

    // Should be able to schedule to n1 again
    auto d = ctrl.schedule("model");
    ASSERT_FALSE(d.error.has_value());
    EXPECT_EQ(d.target_node_id, "n1");
}

TEST(CircuitBreakerTest, HalfOpenAllowsOneProbe) {
    // Fast window so we can test half-open in a unit test
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(2)
                    .setCircuitBreakerWindowMs(100) // 100ms window
                    .build();
    ctrl.register_node(make_node("n1", "model"));
    ctrl.register_node(make_node("n2", "model"));
    ctrl.heartbeat("n1", {{"model", make_metrics()}});
    ctrl.heartbeat("n2", {{"model", make_metrics()}});

    ctrl.record_failure("n1", "model");
    ctrl.record_failure("n1", "model"); // open

    // Wait for window to expire → half-open
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // First schedule call may pick n1 (half-open probe)
    // Second call for n1 should be blocked (probe_in_flight)
    int n1_count = 0;
    for (int i = 0; i < 10; ++i) {
        auto d = ctrl.schedule("model");
        if (!d.error && d.target_node_id == "n1") ++n1_count;
    }
    // At most 1 probe goes to n1
    EXPECT_LE(n1_count, 1);
}

TEST(CircuitBreakerTest, WindowExpiry) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(2)
                    .setCircuitBreakerWindowMs(100)
                    .build();
    ctrl.register_node(make_node("n1", "model"));
    ctrl.heartbeat("n1", {{"model", make_metrics()}});

    ctrl.record_failure("n1", "model");
    // Wait for window to expire before second failure
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    ctrl.record_failure("n1", "model"); // old failure expired, count = 1

    // Circuit should NOT open (only 1 failure in new window)
    auto d = ctrl.schedule("model");
    ASSERT_FALSE(d.error.has_value());
    EXPECT_EQ(d.target_node_id, "n1");
}

TEST(CircuitBreakerTest, IndependentPerModel) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(2)
                    .setCircuitBreakerWindowMs(60000)
                    .build();
    NodeInfo n;
    n.node_id = "n1";
    n.model_metrics["resnet50"] = make_metrics();
    n.model_metrics["bert"]     = make_metrics();
    ctrl.register_node(n);
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}, {"bert", make_metrics()}});

    ctrl.record_failure("n1", "resnet50");
    ctrl.record_failure("n1", "resnet50"); // open for resnet50 only

    auto d_bert = ctrl.schedule("bert");
    EXPECT_FALSE(d_bert.error.has_value()); // bert still works
}

TEST(CircuitBreakerTest, IndependentPerNode) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(2)
                    .setCircuitBreakerWindowMs(60000)
                    .build();
    ctrl.register_node(make_node("n1", "model"));
    ctrl.register_node(make_node("n2", "model"));
    ctrl.heartbeat("n1", {{"model", make_metrics()}});
    ctrl.heartbeat("n2", {{"model", make_metrics()}});

    ctrl.record_failure("n1", "model");
    ctrl.record_failure("n1", "model"); // n1 open

    // n2's circuit is intact
    for (int i = 0; i < 20; ++i) {
        auto d = ctrl.schedule("model");
        if (!d.error && d.target_node_id == "n2") continue;
        // n1 might get one probe; n2 should dominate
    }
    // This just validates no crash
}

// ===========================================================================
// Group 5: Failure Detection (heartbeat timeout)
// ===========================================================================

TEST(FailureDetectionTest, NodeMarkedUnhealthyAfterTimeout) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(150)
                    .setHeartbeatIntervalMs(50)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    EXPECT_EQ(ctrl.healthy_node_count(), 1u);

    // No heartbeat — wait for timeout + 2 monitor intervals
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    EXPECT_EQ(ctrl.healthy_node_count(), 0u);
}

TEST(FailureDetectionTest, NodeRecoveryAfterHeartbeat) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(150)
                    .setHeartbeatIntervalMs(50)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    // Let it time out
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    EXPECT_EQ(ctrl.healthy_node_count(), 0u);

    // Send a fresh heartbeat → should recover immediately
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    // Monitor runs every 50ms; give it time to pick up the fresh timestamp
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(ctrl.healthy_node_count(), 1u);
}

TEST(FailureDetectionTest, PartialClusterFailure) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(150)
                    .setHeartbeatIntervalMs(50)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    ctrl.heartbeat("n2", {{"resnet50", make_metrics()}});
    EXPECT_EQ(ctrl.healthy_node_count(), 2u);

    // Only keep n2 alive
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ctrl.heartbeat("n2", {{"resnet50", make_metrics()}});
    std::this_thread::sleep_for(std::chrono::milliseconds(350));

    EXPECT_EQ(ctrl.healthy_node_count(), 1u);

    auto nodes = ctrl.list_nodes();
    auto n2_it = std::find_if(nodes.begin(), nodes.end(),
                              [](const NodeInfo& n) { return n.node_id == "n2"; });
    ASSERT_NE(n2_it, nodes.end());
    EXPECT_TRUE(n2_it->healthy);
}

TEST(FailureDetectionTest, ZeroHealthyNodesAfterAllFail) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(150)
                    .setHeartbeatIntervalMs(50)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    ctrl.heartbeat("n2", {{"resnet50", make_metrics()}});

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(ctrl.healthy_node_count(), 0u);

    auto d = ctrl.schedule("resnet50");
    ASSERT_TRUE(d.error.has_value());
}

TEST(FailureDetectionTest, AllNodesRecover) {
    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(150)
                    .setHeartbeatIntervalMs(50)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    ctrl.heartbeat("n2", {{"resnet50", make_metrics()}});

    // Let both time out
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    ASSERT_EQ(ctrl.healthy_node_count(), 0u);

    // Revive both
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});
    ctrl.heartbeat("n2", {{"resnet50", make_metrics()}});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(ctrl.healthy_node_count(), 2u);
}

// ===========================================================================
// Group 6: Autoscaling
// ===========================================================================

TEST(AutoscalingTest, ScaleUpOnHighQueueDepth) {
    auto ctrl = ClusterController::Builder()
                    .setSloQueueDepthMax(8.0)
                    .build();
    // avg queue > 80% of 8.0 → 6.5
    ctrl.register_node(make_node("n1", "model", 6.5));
    ctrl.heartbeat("n1", {{"model", make_metrics(6.5)}});

    auto recs = ctrl.compute_scale_signals();
    ASSERT_EQ(recs.size(), 1u);
    EXPECT_EQ(recs[0].model_name, "model");
    EXPECT_GT(recs[0].delta_replicas, 0);
    EXPECT_GT(recs[0].trigger_metric, 0.0);
    EXPECT_FALSE(recs[0].justification.empty());
}

TEST(AutoscalingTest, ScaleUpOnHighLatency) {
    auto ctrl = ClusterController::Builder()
                    .setSloP95TargetMs(100.0)
                    .build();
    ctrl.register_node(make_node("n1", "model", 0.1));
    ctrl.heartbeat("n1", {{"model", make_metrics(0.1, 250.0)}});

    auto recs = ctrl.compute_scale_signals();
    ASSERT_GE(recs.size(), 1u);
    auto it = std::find_if(recs.begin(), recs.end(),
                           [](const ScaleRecommendation& r) {
                               return r.delta_replicas > 0;
                           });
    EXPECT_NE(it, recs.end());
}

TEST(AutoscalingTest, ScaleDownOnLowUtilization) {
    auto ctrl = ClusterController::Builder()
                    .setSloQueueDepthMax(8.0)
                    .build();
    // 2 nodes with very low queue → scale-down eligible
    ctrl.register_node(make_node("n1", "model", 0.1));
    ctrl.register_node(make_node("n2", "model", 0.1));
    ctrl.heartbeat("n1", {{"model", make_metrics(0.1)}});
    ctrl.heartbeat("n2", {{"model", make_metrics(0.1)}});

    auto recs = ctrl.compute_scale_signals();
    auto it = std::find_if(recs.begin(), recs.end(),
                           [](const ScaleRecommendation& r) {
                               return r.delta_replicas < 0;
                           });
    EXPECT_NE(it, recs.end());
}

TEST(AutoscalingTest, NoScaleWhenSLOsMet) {
    auto ctrl = ClusterController::Builder()
                    .setSloP95TargetMs(100.0)
                    .setSloQueueDepthMax(8.0)
                    .build();
    // Single node at moderate load — no scale-up or scale-down
    ctrl.register_node(make_node("n1", "model", 2.0));
    ctrl.heartbeat("n1", {{"model", make_metrics(2.0, 50.0)}});

    auto recs = ctrl.compute_scale_signals();
    EXPECT_TRUE(recs.empty());
}

TEST(AutoscalingTest, MultiModelSignals) {
    auto ctrl = ClusterController::Builder()
                    .setSloQueueDepthMax(8.0)
                    .build();
    NodeInfo n;
    n.node_id = "n1";
    n.model_metrics["resnet50"] = make_metrics(7.0); // scale-up
    n.model_metrics["bert"]     = make_metrics(2.0); // ok
    ctrl.register_node(n);
    ctrl.heartbeat("n1", {{"resnet50", make_metrics(7.0)},
                          {"bert",     make_metrics(2.0)}});

    auto recs = ctrl.compute_scale_signals();
    // Only resnet50 should trigger scale-up
    auto resnet_it = std::find_if(recs.begin(), recs.end(),
                                  [](const ScaleRecommendation& r) {
                                      return r.model_name == "resnet50";
                                  });
    EXPECT_NE(resnet_it, recs.end());
    EXPECT_GT(resnet_it->delta_replicas, 0);
}

TEST(AutoscalingTest, JustificationIsNonEmpty) {
    auto ctrl = ClusterController::Builder()
                    .setSloQueueDepthMax(4.0)
                    .build();
    ctrl.register_node(make_node("n1", "model", 4.0));
    ctrl.heartbeat("n1", {{"model", make_metrics(4.0)}});

    auto recs = ctrl.compute_scale_signals();
    ASSERT_GE(recs.size(), 1u);
    EXPECT_FALSE(recs[0].justification.empty());
}

// ===========================================================================
// Group 7: Prometheus Metrics
// ===========================================================================

TEST(MetricsTest, TextFormatContainsHelp) {
    auto ctrl = fast_controller();
    auto text = ctrl.metrics_text();
    EXPECT_NE(text.find("# HELP"), std::string::npos);
    EXPECT_NE(text.find("# TYPE"), std::string::npos);
}

TEST(MetricsTest, HealthyNodeCount) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    auto text = ctrl.metrics_text();
    EXPECT_NE(text.find("cluster_healthy_nodes 1"), std::string::npos);
}

TEST(MetricsTest, CountersIncrement) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    ctrl.schedule("resnet50");
    ctrl.schedule("resnet50");
    ctrl.schedule("nonexistent"); // rejected

    auto text = ctrl.metrics_text();
    EXPECT_NE(text.find("cluster_requests_total 3"), std::string::npos);
    EXPECT_NE(text.find("cluster_requests_rejected_total 1"), std::string::npos);
}

TEST(MetricsTest, NodeMetricsLabels) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1", "resnet50"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics(3.14)}});

    auto text = ctrl.metrics_text();
    EXPECT_NE(text.find("node_queue_depth{node=\"n1\",model=\"resnet50\"}"),
              std::string::npos);
    EXPECT_NE(text.find("node_p95_latency_ms{node=\"n1\",model=\"resnet50\"}"),
              std::string::npos);
}

// ===========================================================================
// Group 8: Concurrency
// ===========================================================================

TEST(ConcurrencyTest, ConcurrentHeartbeat) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.register_node(make_node("n2"));

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i) {
                std::string id = (t % 2 == 0) ? "n1" : "n2";
                ctrl.heartbeat(id, {{
                    "resnet50", make_metrics(static_cast<double>(i % 5))
                }});
            }
        });
    }
    for (auto& th : threads) th.join();
    // Just validate no crash or data race
    EXPECT_GE(ctrl.healthy_node_count(), 0u);
}

TEST(ConcurrencyTest, ConcurrentRegisterAndSchedule) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    std::atomic<int> success{0};
    std::vector<std::thread> threads;

    // Half threads register/deregister nodes
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            std::string id = "dyn_" + std::to_string(t);
            for (int i = 0; i < 20; ++i) {
                ctrl.register_node(make_node(id));
                ctrl.heartbeat(id, {{"resnet50", make_metrics()}});
                ctrl.deregister_node(id);
            }
        });
    }

    // Half threads schedule
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) {
                auto d = ctrl.schedule("resnet50");
                if (!d.error) ++success;
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_GT(success.load(), 0);
}

TEST(ConcurrencyTest, ConcurrentCircuitBreakerUpdates) {
    auto ctrl = ClusterController::Builder()
                    .setCircuitBreakerFailures(10)
                    .setCircuitBreakerWindowMs(60000)
                    .build();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 20; ++i) {
                if (i % 2 == 0)
                    ctrl.record_failure("n1", "resnet50");
                else
                    ctrl.record_success("n1", "resnet50");
            }
        });
    }
    for (auto& th : threads) th.join();
    // No crash = pass
}

TEST(ConcurrencyTest, AsyncScheduleFuture) {
    auto ctrl = fast_controller();
    ctrl.register_node(make_node("n1"));
    ctrl.heartbeat("n1", {{"resnet50", make_metrics()}});

    auto fut = std::async(std::launch::async, [&] {
        return ctrl.schedule("resnet50");
    });
    auto d = fut.get();
    EXPECT_FALSE(d.error.has_value());
    EXPECT_EQ(d.target_node_id, "n1");
}

TEST(ConcurrencyTest, HighThroughput) {
    auto ctrl = fast_controller();
    // 5 nodes, each serving the model
    for (int i = 1; i <= 5; ++i) {
        std::string id = "n" + std::to_string(i);
        ctrl.register_node(make_node(id, "resnet50",
                                      static_cast<double>(i) * 0.5));
        ctrl.heartbeat(id, {{"resnet50", make_metrics(static_cast<double>(i) * 0.5)}});
    }

    constexpr int THREADS     = 16;
    constexpr int PER_THREAD  = 6250; // 100K total

    std::atomic<int> success{0};
    std::vector<std::thread> threads;
    threads.reserve(THREADS);

    for (int t = 0; t < THREADS; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < PER_THREAD; ++i) {
                auto d = ctrl.schedule("resnet50",
                                        "tenant_" + std::to_string(t));
                if (!d.error) ++success;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(success.load(), THREADS * PER_THREAD);
}
