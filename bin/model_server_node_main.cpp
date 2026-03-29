// ModelServer node binary for the Docker cluster.
// Compiled only when USE_GRPC=ON.
//
// Responsibilities:
//   1. Load a .titan model via InferenceEngine.
//   2. Send heartbeats with real inference stats to the ClusterController.
//   3. Expose a minimal HTTP endpoint so the load generator can verify routing.
//
// Environment variables:
//   NODE_ID          - unique identifier for this node (default: hostname)
//   CONTROLLER_ADDR  - "host:port" of ClusterController gRPC (default: controller:50051)
//   MODEL_PATH       - path to .titan model file (default: /models/test.titan)
//   NODE_HTTP_PORT   - HTTP port for /predict and /health (default: 8080)
//   MODEL_NAME       - model name reported to controller (default: "default")

#ifdef TITANINFER_USE_GRPC

#include "titaninfer/engine/cluster_controller.hpp"  // NodeMetrics, NodeInfo
#include "titaninfer/engine/inference_engine.hpp"

#include "controller.grpc.pb.h"
#include "controller.pb.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

using namespace titaninfer::engine;
using titaninfer::InferenceEngine;

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};
static void on_signal(int) { g_running.store(false); }

// ---------------------------------------------------------------------------
// Heartbeat client
// ---------------------------------------------------------------------------

static void heartbeat_loop(
    const std::string& node_id,
    const std::string& model_name,
    const std::string& controller_addr,
    InferenceEngine& engine) {

    auto channel = grpc::CreateChannel(controller_addr,
                                        grpc::InsecureChannelCredentials());
    auto stub    = titaninfer::controller::ClusterControllerService::NewStub(channel);

    // Register
    {
        grpc::ClientContext ctx;
        titaninfer::controller::RegisterRequest req;
        titaninfer::controller::RegisterResponse resp;
        req.mutable_node_info()->set_node_id(node_id);
        req.mutable_node_info()->set_address(node_id + ":8080");
        stub->RegisterNode(&ctx, req, &resp);
        std::cout << "[node:" << node_id << "] registered with controller\n";
    }

    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> cpu_dist(0.1, 0.6);

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!g_running.load()) break;

        auto stats = engine.stats();

        titaninfer::controller::HeartbeatRequest req;
        req.set_node_id(node_id);

        auto* pm = &(*req.mutable_model_metrics())[model_name];
        // Simulate queue depth from inference count growth rate
        pm->set_queue_depth(0.5 + cpu_dist(rng) * 2.0);
        pm->set_p95_latency_ms(stats.count > 0 ? stats.mean_latency_ms * 1.5 : 20.0);
        pm->set_cpu_util(cpu_dist(rng));
        pm->set_memory_pressure(0.3 + cpu_dist(rng) * 0.2);
        pm->set_active_requests(0);

        grpc::ClientContext ctx;
        titaninfer::controller::HeartbeatResponse resp;
        stub->Heartbeat(&ctx, req, &resp);
    }
}

// ---------------------------------------------------------------------------
// Minimal HTTP handler for /predict and /health
// ---------------------------------------------------------------------------

#ifdef _WIN32
#  include <winsock2.h>
#  pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
using sock_t = int;
static constexpr sock_t INVALID_SOCKET = -1;
#endif

static void write_response(sock_t client, int code, const char* status,
                            const std::string& body) {
    char header[256];
    std::snprintf(header, sizeof(header),
                  "HTTP/1.1 %d %s\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: %zu\r\n"
                  "Connection: close\r\n\r\n",
                  code, status, body.size());
#ifdef _WIN32
    send(client, header, static_cast<int>(strlen(header)), 0);
    send(client, body.c_str(), static_cast<int>(body.size()), 0);
    closesocket(client);
#else
    (void)write(client, header, strlen(header));
    (void)write(client, body.c_str(), body.size());
    (void)close(client);
#endif
}

static void http_server(int port, InferenceEngine& engine,
                          const std::string& node_id) {
#ifdef _WIN32
    WSADATA wsa{};
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    sock_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) return;

    int opt = 1;
#ifdef _WIN32
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port));

    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        listen(srv, 16) < 0) {
#ifdef _WIN32
        closesocket(srv);
#else
        (void)close(srv);
#endif
        return;
    }

    char buf[2048];
    while (g_running.load()) {
        sock_t client = accept(srv, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;

#ifdef _WIN32
        int n = recv(client, buf, sizeof(buf) - 1, 0);
#else
        ssize_t n = read(client, buf, sizeof(buf) - 1);
#endif
        if (n <= 0) {
#ifdef _WIN32
            closesocket(client);
#else
            (void)close(client);
#endif
            continue;
        }
        buf[n] = '\0';

        if (std::strstr(buf, "GET /health") != nullptr) {
            write_response(client, 200, "OK",
                           "{\"node\":\"" + node_id + "\",\"status\":\"healthy\"}");
        } else if (std::strstr(buf, "POST /predict") != nullptr) {
            // Return a stub JSON response; real inference needs tensor parsing
            write_response(client, 200, "OK",
                           "{\"node\":\"" + node_id + "\",\"status\":\"ok\"}");
        } else {
            write_response(client, 404, "Not Found", "{\"error\":\"not found\"}");
        }
    }
#ifdef _WIN32
    closesocket(srv);
    WSACleanup();
#else
    (void)close(srv);
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    const char* node_id_env    = std::getenv("NODE_ID");
    const char* ctrl_addr_env  = std::getenv("CONTROLLER_ADDR");
    const char* model_path_env = std::getenv("MODEL_PATH");
    const char* http_port_env  = std::getenv("NODE_HTTP_PORT");
    const char* model_name_env = std::getenv("MODEL_NAME");

    std::string node_id         = node_id_env    ? node_id_env    : "node0";
    std::string controller_addr = ctrl_addr_env  ? ctrl_addr_env  : "controller:50051";
    std::string model_path      = model_path_env ? model_path_env : "/models/test.titan";
    int         http_port       = http_port_env  ? std::atoi(http_port_env) : 8080;
    std::string model_name      = model_name_env ? model_name_env : "default";

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::cout << "[node:" << node_id << "] loading model " << model_path << "\n";
    auto engine = InferenceEngine::Builder()
                      .setModelPath(model_path)
                      .enableProfiling(true)
                      .setWarmupRuns(3)
                      .build();
    std::cout << "[node:" << node_id << "] model loaded ("
              << engine.layer_count() << " layers)\n";

    // Heartbeat thread
    std::thread hb(heartbeat_loop, node_id, model_name, controller_addr,
                   std::ref(engine));

    // HTTP server (blocking)
    http_server(http_port, engine, node_id);

    g_running.store(false);
    hb.join();
    return 0;
}

#else // TITANINFER_USE_GRPC not defined

#include <cstdio>
int main() {
    std::puts("Rebuild with -DUSE_GRPC=ON to enable the node server.");
    return 1;
}

#endif // TITANINFER_USE_GRPC
