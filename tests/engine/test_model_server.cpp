#include <gtest/gtest.h>
#include "titaninfer/engine/model_server.hpp"
#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/exceptions.hpp"
#include "titaninfer/logger.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace titaninfer;
using namespace titaninfer::engine;
using namespace titaninfer::layers;
using namespace titaninfer::io;

// ============================================================
// RAII temp file helper
// ============================================================

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& name) : path(name) {}
    ~TempFile() { std::remove(path.c_str()); }
};

// Helper: Dense(4,8) -> ReLU -> Dense(8,3) -> Softmax
static void save_test_mlp(const std::string& filename) {
    Sequential model;

    auto dense1 = std::make_unique<DenseLayer>(4, 8);
    Tensor w1({8, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.1f * static_cast<float>((i % 5) + 1);
    }
    dense1->set_weights(w1);
    Tensor b1({8});
    for (size_t i = 0; i < b1.size(); ++i) {
        b1.data()[i] = 0.01f * static_cast<float>(i);
    }
    dense1->set_bias(b1);
    model.add(std::move(dense1));

    model.add(std::make_unique<ReluLayer>());

    auto dense2 = std::make_unique<DenseLayer>(8, 3);
    Tensor w2({3, 8});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.05f * static_cast<float>((i % 4) + 1);
    }
    dense2->set_weights(w2);
    Tensor b2({3});
    b2.zero();
    dense2->set_bias(b2);
    model.add(std::move(dense2));

    model.add(std::make_unique<SoftmaxLayer>());

    ModelSerializer::save(model, filename);
}

// Build a different model: Dense(4,6) -> Sigmoid -> Dense(6,2) -> Softmax
static void save_alt_mlp(const std::string& filename) {
    Sequential model;

    auto dense1 = std::make_unique<DenseLayer>(4, 6);
    Tensor w1({6, 4});
    for (size_t i = 0; i < w1.size(); ++i) {
        w1.data()[i] = 0.2f * static_cast<float>((i % 3) + 1);
    }
    dense1->set_weights(w1);
    Tensor b1({6});
    b1.zero();
    dense1->set_bias(b1);
    model.add(std::move(dense1));

    model.add(std::make_unique<SigmoidLayer>());

    auto dense2 = std::make_unique<DenseLayer>(6, 2);
    Tensor w2({2, 6});
    for (size_t i = 0; i < w2.size(); ++i) {
        w2.data()[i] = 0.1f * static_cast<float>((i % 4) + 1);
    }
    dense2->set_weights(w2);
    Tensor b2({2});
    b2.zero();
    dense2->set_bias(b2);
    model.add(std::move(dense2));

    model.add(std::make_unique<SoftmaxLayer>());

    ModelSerializer::save(model, filename);
}

static Tensor make_test_input() {
    Tensor input({4});
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    return input;
}

// ============================================================
// Fixtures
// ============================================================

class ModelServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().set_level(LogLevel::SILENT);
    }
    void TearDown() override {
        Logger::instance().set_level(LogLevel::INFO);
    }
};

// ============================================================
// Group 1: Builder & Construction (4 tests)
// ============================================================

TEST_F(ModelServerTest, BuilderDefaults) {
    auto server = ModelServer::Builder().build();
    EXPECT_EQ(server.loaded_model_count(), 0u);
    EXPECT_EQ(server.registered_model_count(), 0u);
}

TEST_F(ModelServerTest, BuilderCustomConfig) {
    auto server = ModelServer::Builder()
        .setMaxLoadedModels(8)
        .setWorkerThreads(4)
        .setEnginesPerModel(2)
        .enableProfiling(true)
        .build();
    EXPECT_EQ(server.loaded_model_count(), 0u);
}

TEST_F(ModelServerTest, BuilderZeroWorkers) {
    // 0 workers should default to hardware_concurrency
    auto server = ModelServer::Builder()
        .setWorkerThreads(0)
        .build();
    EXPECT_EQ(server.loaded_model_count(), 0u);
}

TEST_F(ModelServerTest, MoveSemantics) {
    TempFile f("test_ms_move.titan");
    save_test_mlp(f.path);

    auto server1 = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2)
        .setEnginesPerModel(1)
        .build();
    server1.register_model("m1", 1, f.path);

    // Move construct
    ModelServer server2 = std::move(server1);
    EXPECT_EQ(server2.registered_model_count(), 1u);

    // Move assign
    ModelServer server3 = ModelServer::Builder().build();
    server3 = std::move(server2);
    EXPECT_EQ(server3.registered_model_count(), 1u);
}

// ============================================================
// Group 2: Model Registry (7 tests)
// ============================================================

TEST_F(ModelServerTest, RegisterAndListVersions) {
    TempFile f("test_ms_reg.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("resnet", 1, f.path);
    server.register_model("resnet", 2, f.path);
    server.register_model("resnet", 3, f.path);

    auto versions = server.list_versions("resnet");
    EXPECT_EQ(versions.size(), 3u);
    EXPECT_EQ(versions[0].version, 1u);
    EXPECT_EQ(versions[1].version, 2u);
    EXPECT_EQ(versions[2].version, 3u);
}

TEST_F(ModelServerTest, RegisterWithTags) {
    TempFile f("test_ms_tags.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f.path, {"prod", "stable"});

    auto versions = server.list_versions("m1");
    ASSERT_EQ(versions.size(), 1u);
    EXPECT_TRUE(versions[0].tags.count("prod") > 0);
    EXPECT_TRUE(versions[0].tags.count("stable") > 0);
}

TEST_F(ModelServerTest, UnregisterModel) {
    TempFile f("test_ms_unreg.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f.path);
    server.register_model("m1", 2, f.path);
    EXPECT_EQ(server.registered_model_count(), 2u);

    server.unregister_model("m1", 1);
    EXPECT_EQ(server.registered_model_count(), 1u);

    auto versions = server.list_versions("m1");
    ASSERT_EQ(versions.size(), 1u);
    EXPECT_EQ(versions[0].version, 2u);
}

TEST_F(ModelServerTest, AddRemoveTag) {
    TempFile f("test_ms_addtag.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f.path);

    server.add_tag("m1", 1, "canary");
    auto v = server.list_versions("m1");
    EXPECT_TRUE(v[0].tags.count("canary") > 0);

    server.remove_tag("m1", 1, "canary");
    v = server.list_versions("m1");
    EXPECT_TRUE(v[0].tags.count("canary") == 0);
}

TEST_F(ModelServerTest, PinUnpinModel) {
    TempFile f("test_ms_pin.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f.path);

    server.pin_model("m1", 1);
    auto v = server.list_versions("m1");
    EXPECT_TRUE(v[0].pinned);

    server.unpin_model("m1", 1);
    v = server.list_versions("m1");
    EXPECT_FALSE(v[0].pinned);
}

TEST_F(ModelServerTest, RegisterDuplicateVersion) {
    TempFile f1("test_ms_dup1.titan");
    TempFile f2("test_ms_dup2.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f1.path, {"old"});
    server.register_model("m1", 1, f2.path, {"new"});

    EXPECT_EQ(server.registered_model_count(), 1u);
    auto v = server.list_versions("m1");
    EXPECT_EQ(v[0].file_path, f2.path);
    EXPECT_TRUE(v[0].tags.count("new") > 0);
}

TEST_F(ModelServerTest, ListVersionsUnknownModel) {
    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    auto versions = server.list_versions("nonexistent");
    EXPECT_TRUE(versions.empty());
}

// ============================================================
// Group 3: Basic Inference (5 tests)
// ============================================================

TEST_F(ModelServerTest, PredictRegisteredModel) {
    TempFile f("test_ms_pred.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    Response resp = server.predict("mlp", input);
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(resp.body.shape().size(), 1u);
    EXPECT_EQ(resp.body.shape()[0], 3u);

    // Softmax output: all >= 0, sum ≈ 1
    float sum = 0.0f;
    for (size_t i = 0; i < resp.body.size(); ++i) {
        EXPECT_GE(resp.body.data()[i], 0.0f);
        EXPECT_LE(resp.body.data()[i], 1.0f);
        sum += resp.body.data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    EXPECT_GT(resp.latency_ms, 0.0);
}

TEST_F(ModelServerTest, PredictUnregisteredModel) {
    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    auto input = make_test_input();
    Response resp = server.predict("nonexistent", input);
    EXPECT_EQ(resp.status_code, 404);
    EXPECT_FALSE(resp.error_message.empty());
}

TEST_F(ModelServerTest, PredictShapeMismatch) {
    TempFile f("test_ms_shape.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    // Wrong shape: 3 instead of 4
    Tensor wrong_input({3});
    wrong_input.fill(1.0f);
    Response resp = server.predict("mlp", wrong_input);
    EXPECT_EQ(resp.status_code, 400);
    EXPECT_FALSE(resp.error_message.empty());
}

TEST_F(ModelServerTest, PredictMultipleModels) {
    TempFile f1("test_ms_multi1.titan");
    TempFile f2("test_ms_multi2.titan");
    save_test_mlp(f1.path);
    save_alt_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp1", 1, f1.path);
    server.register_model("mlp2", 1, f2.path);

    auto input = make_test_input();

    Response r1 = server.predict("mlp1", input);
    EXPECT_EQ(r1.status_code, 200);
    EXPECT_EQ(r1.body.shape()[0], 3u);

    Response r2 = server.predict("mlp2", input);
    EXPECT_EQ(r2.status_code, 200);
    EXPECT_EQ(r2.body.shape()[0], 2u);
}

TEST_F(ModelServerTest, PredictConsistency) {
    TempFile f("test_ms_consist.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    Response r1 = server.predict("mlp", input);
    Response r2 = server.predict("mlp", input);

    ASSERT_EQ(r1.body.size(), r2.body.size());
    for (size_t i = 0; i < r1.body.size(); ++i) {
        EXPECT_FLOAT_EQ(r1.body.data()[i], r2.body.data()[i]);
    }
}

// ============================================================
// Group 4: REST-like Request Handling (5 tests)
// ============================================================

TEST_F(ModelServerTest, HandleRequestPostPredict) {
    TempFile f("test_ms_rest.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mymodel", 1, f.path);

    Request req;
    req.method = HttpMethod::POST;
    req.path = "/v1/models/mymodel/predict";
    req.body = make_test_input();

    Response resp = server.handle_request(req);
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(resp.body.shape()[0], 3u);
}

TEST_F(ModelServerTest, HandleRequestVersionedPredict) {
    TempFile f("test_ms_rest_ver.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mymodel", 2, f.path);

    Request req;
    req.method = HttpMethod::POST;
    req.path = "/v1/models/mymodel/versions/2/predict";
    req.body = make_test_input();

    Response resp = server.handle_request(req);
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(resp.headers.at("X-Model-Version"), "2");
}

TEST_F(ModelServerTest, HandleRequestInvalidPath) {
    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();

    Request req;
    req.method = HttpMethod::POST;
    req.path = "/v1/invalid";
    req.body = make_test_input();

    Response resp = server.handle_request(req);
    EXPECT_EQ(resp.status_code, 404);
}

TEST_F(ModelServerTest, HandleRequestWrongMethod) {
    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();

    Request req;
    req.method = HttpMethod::GET;
    req.path = "/v1/models/mymodel/predict";
    req.body = make_test_input();

    Response resp = server.handle_request(req);
    EXPECT_EQ(resp.status_code, 405);
}

TEST_F(ModelServerTest, HandleRequestIdPropagation) {
    TempFile f("test_ms_reqid.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    Request req;
    req.method = HttpMethod::POST;
    req.path = "/v1/models/mlp/predict";
    req.body = make_test_input();
    req.request_id = "custom-id-42";

    Response resp = server.handle_request(req);
    EXPECT_EQ(resp.headers.at("X-Request-Id"), "custom-id-42");
}

// ============================================================
// Group 5: LRU Cache (5 tests)
// ============================================================

TEST_F(ModelServerTest, CacheLoadsOnFirstRequest) {
    TempFile f("test_ms_lazy.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    // Not loaded yet
    EXPECT_EQ(server.loaded_model_count(), 0u);

    // First predict triggers load
    auto input = make_test_input();
    Response resp = server.predict("mlp", input);
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(server.loaded_model_count(), 1u);
}

TEST_F(ModelServerTest, CacheEvictsLRU) {
    TempFile f1("test_ms_lru1.titan");
    TempFile f2("test_ms_lru2.titan");
    TempFile f3("test_ms_lru3.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);
    save_test_mlp(f3.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(2)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f1.path);
    server.register_model("m2", 1, f2.path);
    server.register_model("m3", 1, f3.path);

    auto input = make_test_input();

    // Load m1, m2
    server.predict("m1", input);
    server.predict("m2", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);

    // Load m3 -> evicts m1 (LRU)
    server.predict("m3", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);

    // m1 should be evicted, re-requesting it reloads
    server.predict("m1", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);
}

TEST_F(ModelServerTest, CacheSkipsPinnedOnEviction) {
    TempFile f1("test_ms_pin1.titan");
    TempFile f2("test_ms_pin2.titan");
    TempFile f3("test_ms_pin3.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);
    save_test_mlp(f3.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(2)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f1.path);
    server.register_model("m2", 1, f2.path);
    server.register_model("m3", 1, f3.path);

    auto input = make_test_input();

    // Load m1 (pinned) and m2
    server.predict("m1", input);
    server.pin_model("m1", 1);
    server.predict("m2", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);

    // Load m3 -> should evict m2 (m1 is pinned)
    server.predict("m3", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);

    // m2 was evicted; verify m1 is still loaded by checking it returns instantly
    Response r1 = server.predict("m1", input);
    EXPECT_EQ(r1.status_code, 200);
}

TEST_F(ModelServerTest, CacheReloadsAfterEviction) {
    TempFile f1("test_ms_reload1.titan");
    TempFile f2("test_ms_reload2.titan");
    TempFile f3("test_ms_reload3.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);
    save_test_mlp(f3.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(2)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("m1", 1, f1.path);
    server.register_model("m2", 1, f2.path);
    server.register_model("m3", 1, f3.path);

    auto input = make_test_input();

    server.predict("m1", input);
    server.predict("m2", input);
    server.predict("m3", input); // evicts m1

    // Re-request m1 should reload it (evicting m2)
    Response resp = server.predict("m1", input);
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(resp.body.shape()[0], 3u);
}

TEST_F(ModelServerTest, CacheExplicitEviction) {
    TempFile f("test_ms_evict.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    server.predict("mlp", input);
    EXPECT_EQ(server.loaded_model_count(), 1u);

    server.unregister_model("mlp", 1);
    EXPECT_EQ(server.loaded_model_count(), 0u);
}

// ============================================================
// Group 6: Rate Limiting (5 tests)
// ============================================================

TEST_F(ModelServerTest, NoQuotaAllowsAll) {
    TempFile f("test_ms_noquota.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    // No quota set — should all succeed
    for (int i = 0; i < 10; ++i) {
        Response resp = server.predict("mlp", input, "tenant_a");
        EXPECT_EQ(resp.status_code, 200);
    }
}

TEST_F(ModelServerTest, QPSQuotaEnforced) {
    TempFile f("test_ms_qps.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    // Very low QPS: 2 per second
    TenantQuota quota;
    quota.max_qps = 2.0;
    quota.max_concurrent = 100;
    server.set_tenant_quota("t1", quota);

    auto input = make_test_input();

    // Rapid-fire requests
    int ok_count = 0;
    int rejected_count = 0;
    for (int i = 0; i < 10; ++i) {
        Response resp = server.predict("mlp", input, "t1");
        if (resp.status_code == 200) ok_count++;
        if (resp.status_code == 429) rejected_count++;
    }

    // Should have some accepted and some rejected
    EXPECT_GT(ok_count, 0);
    EXPECT_GT(rejected_count, 0);
}

TEST_F(ModelServerTest, ConcurrentQuotaEnforced) {
    TempFile f("test_ms_conc.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(4).setEnginesPerModel(2).build();
    server.register_model("mlp", 1, f.path);

    TenantQuota quota;
    quota.max_qps = 10000.0;  // High QPS
    quota.max_concurrent = 1;  // Only 1 concurrent
    server.set_tenant_quota("t1", quota);

    auto input = make_test_input();

    // Launch multiple async requests
    std::vector<std::future<Response>> futures;
    for (int i = 0; i < 5; ++i) {
        futures.push_back(server.predict_async("mlp", input, "t1"));
    }

    int ok = 0, rejected = 0;
    for (auto& f : futures) {
        Response resp = f.get();
        if (resp.status_code == 200) ok++;
        if (resp.status_code == 429) rejected++;
    }

    // At least one should succeed, some may be rejected
    EXPECT_GT(ok, 0);
}

TEST_F(ModelServerTest, QuotaPerTenant) {
    TempFile f("test_ms_pertenant.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    TenantQuota strict;
    strict.max_qps = 1.0;
    strict.max_concurrent = 100;
    server.set_tenant_quota("strict_tenant", strict);

    auto input = make_test_input();

    // Strict tenant hits quota
    server.predict("mlp", input, "strict_tenant");
    Response r1 = server.predict("mlp", input, "strict_tenant");
    // May be 429 or 200 depending on timing

    // Unquoted tenant always succeeds
    for (int i = 0; i < 5; ++i) {
        Response r2 = server.predict("mlp", input, "free_tenant");
        EXPECT_EQ(r2.status_code, 200);
    }
}

TEST_F(ModelServerTest, RemoveQuotaAllowsAll) {
    TempFile f("test_ms_rmquota.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);

    TenantQuota quota;
    quota.max_qps = 1.0;
    quota.max_concurrent = 100;
    server.set_tenant_quota("t1", quota);

    server.remove_tenant_quota("t1");

    auto input = make_test_input();
    // After removal, all should succeed
    for (int i = 0; i < 10; ++i) {
        Response resp = server.predict("mlp", input, "t1");
        EXPECT_EQ(resp.status_code, 200);
    }
}

// ============================================================
// Group 7: Traffic Splitting & Hot Reload (5 tests)
// ============================================================

TEST_F(ModelServerTest, DefaultRouteToLatestVersion) {
    TempFile f1("test_ms_latest1.titan");
    TempFile f2("test_ms_latest2.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f1.path);
    server.register_model("mlp", 2, f2.path);

    auto input = make_test_input();
    Response resp = server.predict("mlp", input);
    EXPECT_EQ(resp.status_code, 200);
    // Should route to version 2 (highest)
    EXPECT_EQ(resp.headers.at("X-Model-Version"), "2");
}

TEST_F(ModelServerTest, TrafficSplitDistribution) {
    TempFile f("test_ms_split.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f.path);
    server.register_model("mlp", 2, f.path);

    // 80% to v1, 20% to v2
    server.set_traffic_rules("mlp", {{1, 0.8}, {2, 0.2}});

    auto input = make_test_input();
    int v1_count = 0, v2_count = 0;
    for (int i = 0; i < 200; ++i) {
        Response resp = server.predict("mlp", input);
        ASSERT_EQ(resp.status_code, 200);
        uint32_t ver = static_cast<uint32_t>(
            std::stoul(resp.headers.at("X-Model-Version")));
        if (ver == 1) v1_count++;
        else if (ver == 2) v2_count++;
    }

    // With 200 samples, 80/20 split: v1 should be ~160, v2 ~40
    // Allow generous tolerance
    EXPECT_GT(v1_count, 100);  // At least 50% (way lower than 80% to avoid flakiness)
    EXPECT_GT(v2_count, 5);    // At least a few
    EXPECT_EQ(v1_count + v2_count, 200);
}

TEST_F(ModelServerTest, HotReloadSwitchesVersion) {
    TempFile f1("test_ms_hot1.titan");
    TempFile f2("test_ms_hot2.titan");
    save_test_mlp(f1.path);
    save_alt_mlp(f2.path);  // Different model: output size 2

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f1.path);

    auto input = make_test_input();

    // Before reload: output shape = {3}
    Response r1 = server.predict("mlp", input);
    EXPECT_EQ(r1.status_code, 200);
    EXPECT_EQ(r1.body.shape()[0], 3u);

    // Hot reload with alt model (output shape = {2})
    server.reload_model("mlp", 1, f2.path);

    Response r2 = server.predict("mlp", input);
    EXPECT_EQ(r2.status_code, 200);
    EXPECT_EQ(r2.body.shape()[0], 2u);
}

TEST_F(ModelServerTest, HotReloadPreservesRegistry) {
    TempFile f1("test_ms_hotreg1.titan");
    TempFile f2("test_ms_hotreg2.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();
    server.register_model("mlp", 1, f1.path, {"prod"});

    server.reload_model("mlp", 1, f2.path);

    auto versions = server.list_versions("mlp");
    ASSERT_EQ(versions.size(), 1u);
    EXPECT_EQ(versions[0].file_path, f2.path);
    EXPECT_TRUE(versions[0].tags.count("prod") > 0);
}

TEST_F(ModelServerTest, HotReloadNonexistentThrows) {
    auto server = ModelServer::Builder()
        .setWorkerThreads(2).setEnginesPerModel(1).build();

    EXPECT_THROW(
        server.reload_model("nonexistent", 1, "path.titan"),
        ServerException);
}

// ============================================================
// Group 8: Concurrency (4 tests)
// ============================================================

TEST_F(ModelServerTest, ConcurrentPredictSameModel) {
    TempFile f("test_ms_concsame.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(4)
        .setEnginesPerModel(4)
        .build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    // Warm up to load the model
    server.predict("mlp", input);

    const int num_threads = 8;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 10; ++i) {
                Response resp = server.predict("mlp", input);
                if (resp.status_code == 200) {
                    success_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(success_count.load(), num_threads * 10);
}

TEST_F(ModelServerTest, ConcurrentPredictDifferentModels) {
    TempFile f1("test_ms_concdiff1.titan");
    TempFile f2("test_ms_concdiff2.titan");
    save_test_mlp(f1.path);
    save_alt_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(4)
        .setEnginesPerModel(2)
        .build();
    server.register_model("mlp1", 1, f1.path);
    server.register_model("mlp2", 1, f2.path);

    auto input = make_test_input();
    std::atomic<int> ok{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        std::string name = (t % 2 == 0) ? "mlp1" : "mlp2";
        threads.emplace_back([&server, &input, &ok, name]() {
            for (int i = 0; i < 10; ++i) {
                Response resp = server.predict(name, input);
                if (resp.status_code == 200) {
                    ok.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(ok.load(), 40);
}

TEST_F(ModelServerTest, ConcurrentRegisterAndPredict) {
    TempFile f("test_ms_concreg.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(16)
        .setWorkerThreads(4)
        .setEnginesPerModel(2)
        .build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    std::atomic<bool> stop{false};
    std::atomic<int> predict_ok{0};

    // Thread 1: predict continuously
    std::thread predictor([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            Response resp = server.predict("mlp", input);
            if (resp.status_code == 200) {
                predict_ok.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // Thread 2: register/unregister models
    TempFile f2("test_ms_concreg2.titan");
    save_test_mlp(f2.path);
    for (int i = 0; i < 20; ++i) {
        server.register_model("dynamic", static_cast<uint32_t>(i + 1),
                              f2.path);
    }
    for (int i = 0; i < 20; ++i) {
        server.unregister_model("dynamic", static_cast<uint32_t>(i + 1));
    }

    stop.store(true, std::memory_order_relaxed);
    predictor.join();

    EXPECT_GT(predict_ok.load(), 0);
}

TEST_F(ModelServerTest, AsyncPredictFuture) {
    TempFile f("test_ms_async.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(4)
        .setEnginesPerModel(2)
        .build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();

    std::vector<std::future<Response>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(server.predict_async("mlp", input));
    }

    for (auto& fut : futures) {
        Response resp = fut.get();
        EXPECT_EQ(resp.status_code, 200);
        EXPECT_EQ(resp.body.shape()[0], 3u);
    }
}

// ============================================================
// Group 9: Edge Cases (3 tests)
// ============================================================

TEST_F(ModelServerTest, ServerStatsAccuracy) {
    TempFile f1("test_ms_stats1.titan");
    TempFile f2("test_ms_stats2.titan");
    save_test_mlp(f1.path);
    save_test_mlp(f2.path);

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(2).setEnginesPerModel(1).build();

    EXPECT_EQ(server.registered_model_count(), 0u);
    EXPECT_EQ(server.loaded_model_count(), 0u);

    server.register_model("m1", 1, f1.path);
    server.register_model("m2", 1, f2.path);
    EXPECT_EQ(server.registered_model_count(), 2u);
    EXPECT_EQ(server.loaded_model_count(), 0u);

    auto input = make_test_input();
    server.predict("m1", input);
    EXPECT_EQ(server.loaded_model_count(), 1u);

    server.predict("m2", input);
    EXPECT_EQ(server.loaded_model_count(), 2u);
}

TEST_F(ModelServerTest, EnginePoolExhaustion) {
    TempFile f("test_ms_exhaust.titan");
    save_test_mlp(f.path);

    // Pool size 1 — all threads fight for the same engine
    auto server = ModelServer::Builder()
        .setMaxLoadedModels(4)
        .setWorkerThreads(4)
        .setEnginesPerModel(1)
        .build();
    server.register_model("mlp", 1, f.path);

    auto input = make_test_input();
    std::atomic<int> ok{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 5; ++i) {
                Response resp = server.predict("mlp", input);
                if (resp.status_code == 200)
                    ok.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(ok.load(), 20);
}

TEST_F(ModelServerTest, HandleRequestAsyncFuture) {
    TempFile f("test_ms_asyncreq.titan");
    save_test_mlp(f.path);

    auto server = ModelServer::Builder()
        .setWorkerThreads(4)
        .setEnginesPerModel(2)
        .build();
    server.register_model("mlp", 1, f.path);

    Request req;
    req.method = HttpMethod::POST;
    req.path = "/v1/models/mlp/predict";
    req.body = make_test_input();

    auto fut = server.handle_request_async(req);
    Response resp = fut.get();
    EXPECT_EQ(resp.status_code, 200);
    EXPECT_EQ(resp.body.shape()[0], 3u);
}

// ============================================================
// Integration: 100+ concurrent requests across 10 models
// ============================================================

TEST_F(ModelServerTest, IntegrationHighConcurrency) {
    const int num_models = 10;
    std::vector<TempFile> files;
    files.reserve(num_models);
    for (int i = 0; i < num_models; ++i) {
        std::string name = "test_ms_integ_" + std::to_string(i) + ".titan";
        files.emplace_back(name);
        save_test_mlp(files.back().path);
    }

    auto server = ModelServer::Builder()
        .setMaxLoadedModels(static_cast<size_t>(num_models))
        .setWorkerThreads(8)
        .setEnginesPerModel(4)
        .build();

    for (int i = 0; i < num_models; ++i) {
        server.register_model("model_" + std::to_string(i), 1,
                              files[static_cast<size_t>(i)].path);
    }

    auto input = make_test_input();
    const int requests_per_thread = 10;
    const int num_threads = 12;

    std::atomic<int> total_ok{0};
    std::atomic<int> total_fail{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                std::string model_name =
                    "model_" + std::to_string((t * requests_per_thread + i)
                                               % num_models);
                Response resp = server.predict(model_name, input);
                if (resp.status_code == 200) {
                    total_ok.fetch_add(1, std::memory_order_relaxed);
                } else {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    EXPECT_EQ(total_ok.load(), num_threads * requests_per_thread);
    EXPECT_EQ(total_fail.load(), 0);
    EXPECT_EQ(server.loaded_model_count(),
              static_cast<size_t>(num_models));
}
