// ClusterController gRPC server binary.
// Compiled only when USE_GRPC=ON (Docker cluster scenario).
// Wraps the zero-dependency ClusterController library with a real gRPC
// transport so that ModelServer nodes can register, heartbeat, and receive
// routing decisions over the network.
//
// Usage:
//   cluster_controller_server [--port=50051] [--metrics-port=9090]
//   CONTROLLER_PORT=50051 METRICS_PORT=9090 ./cluster_controller_server

#ifdef TITANINFER_USE_GRPC

#include "titaninfer/engine/cluster_controller.hpp"

// Generated protobuf / gRPC stubs (produced by bin/CMakeLists.txt)
#include "controller.grpc.pb.h"
#include "controller.pb.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

using namespace titaninfer::engine;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

// ---------------------------------------------------------------------------
// Minimal hand-rolled HTTP server for /metrics (Prometheus scrape)
// ---------------------------------------------------------------------------
// We expose a single TCP socket on metrics_port that accepts one connection,
// writes the Prometheus text body, and closes. This avoids any HTTP library.

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

static void run_metrics_server(int port,
                                 const ClusterController* ctrl,
                                 std::atomic<bool>* running) {
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
        listen(srv, 5) < 0) {
#ifdef _WIN32
        closesocket(srv);
#else
        (void)close(srv);
#endif
        return;
    }

    while (running->load()) {
        sock_t client = accept(srv, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;

        std::string body = ctrl->metrics_text();
        std::ostringstream resp;
        resp << "HTTP/1.1 200 OK\r\n"
             << "Content-Type: text/plain; version=0.0.4\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Connection: close\r\n\r\n"
             << body;
        std::string resp_str = resp.str();
#ifdef _WIN32
        send(client, resp_str.c_str(),
             static_cast<int>(resp_str.size()), 0);
        closesocket(client);
#else
        (void)write(client, resp_str.c_str(), resp_str.size());
        (void)close(client);
#endif
    }
#ifdef _WIN32
    closesocket(srv);
    WSACleanup();
#else
    (void)close(srv);
#endif
}

// ---------------------------------------------------------------------------
// gRPC service implementation
// ---------------------------------------------------------------------------

class ClusterControllerServiceImpl final
    : public titaninfer::controller::ClusterControllerService::Service {
public:
    explicit ClusterControllerServiceImpl(ClusterController& ctrl)
        : ctrl_(ctrl) {}

    Status RegisterNode(
        ServerContext* /*ctx*/,
        const titaninfer::controller::RegisterRequest* req,
        titaninfer::controller::RegisterResponse* resp) override {
        NodeInfo info;
        info.node_id = req->node_info().node_id();
        info.address = req->node_info().address();
        for (auto& [model, pm] : req->node_info().model_metrics()) {
            NodeMetrics m;
            m.queue_depth     = pm.queue_depth();
            m.p95_latency_ms  = pm.p95_latency_ms();
            m.cpu_util        = pm.cpu_util();
            m.memory_pressure = pm.memory_pressure();
            m.active_requests = static_cast<size_t>(pm.active_requests());
            m.timestamp       = std::chrono::steady_clock::now();
            info.model_metrics[model] = m;
        }
        ctrl_.register_node(info);
        resp->set_accepted(true);
        resp->set_cluster_version(0);
        return Status::OK;
    }

    Status Heartbeat(
        ServerContext* /*ctx*/,
        const titaninfer::controller::HeartbeatRequest* req,
        titaninfer::controller::HeartbeatResponse* resp) override {
        std::unordered_map<std::string, NodeMetrics> metrics;
        for (auto& [model, pm] : req->model_metrics()) {
            NodeMetrics m;
            m.queue_depth     = pm.queue_depth();
            m.p95_latency_ms  = pm.p95_latency_ms();
            m.cpu_util        = pm.cpu_util();
            m.memory_pressure = pm.memory_pressure();
            m.active_requests = static_cast<size_t>(pm.active_requests());
            m.timestamp       = std::chrono::steady_clock::now();
            metrics[model]    = m;
        }
        ctrl_.heartbeat(req->node_id(), metrics);
        resp->set_acknowledged(true);
        resp->set_cluster_version(0);
        return Status::OK;
    }

    Status Schedule(
        ServerContext* /*ctx*/,
        const titaninfer::controller::ScheduleRequest* req,
        titaninfer::controller::RoutingDecision* resp) override {
        auto d = ctrl_.schedule(req->model_name(), req->tenant_id());
        resp->set_target_node_id(d.target_node_id);
        resp->set_confidence_score(d.confidence_score);
        resp->set_trace_id(d.trace_id);
        if (d.error) resp->set_error(*d.error);
        return Status::OK;
    }

    Status GetScaleSignals(
        ServerContext* /*ctx*/,
        const google::protobuf::Empty* /*req*/,
        ServerWriter<titaninfer::controller::ScaleRecommendation>* writer)
        override {
        auto recs = ctrl_.compute_scale_signals();
        for (auto& r : recs) {
            titaninfer::controller::ScaleRecommendation sr;
            sr.set_model_name(r.model_name);
            sr.set_delta_replicas(r.delta_replicas);
            sr.set_justification(r.justification);
            sr.set_trigger_metric(r.trigger_metric);
            writer->Write(sr);
        }
        return Status::OK;
    }

    Status GetClusterState(
        ServerContext* /*ctx*/,
        const google::protobuf::Empty* /*req*/,
        titaninfer::controller::ClusterStateResponse* resp) override {
        auto nodes = ctrl_.list_nodes();
        for (auto& n : nodes) {
            auto* pn = resp->add_nodes();
            pn->set_node_id(n.node_id);
            pn->set_address(n.address);
        }
        resp->set_healthy(ctrl_.healthy_node_count());
        return Status::OK;
    }

    Status GetMetrics(
        ServerContext* /*ctx*/,
        const google::protobuf::Empty* /*req*/,
        titaninfer::controller::MetricsResponse* resp) override {
        resp->set_prometheus_text(ctrl_.metrics_text());
        return Status::OK;
    }

private:
    ClusterController& ctrl_;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running.store(false); }

int main(int argc, char* argv[]) {
    // Parse args / env
    std::string grpc_port  = "50051";
    int         metrics_port = 9090;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--port=", 7) == 0)
            grpc_port = argv[i] + 7;
        else if (std::strncmp(argv[i], "--metrics-port=", 15) == 0)
            metrics_port = std::atoi(argv[i] + 15);
    }
    if (const char* p = std::getenv("CONTROLLER_PORT")) grpc_port    = p;
    if (const char* p = std::getenv("METRICS_PORT"))
        metrics_port = std::atoi(p);

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto ctrl = ClusterController::Builder()
                    .setHeartbeatTimeoutMs(5000)
                    .setHeartbeatIntervalMs(2000)
                    .setSloP95TargetMs(100.0)
                    .setSloQueueDepthMax(8.0)
                    .build();

    ClusterControllerServiceImpl service(ctrl);
    grpc::EnableDefaultHealthCheckService(true);

    ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:" + grpc_port,
                              grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    std::cout << "[controller] gRPC listening on :" << grpc_port
              << "  metrics on :" << metrics_port << "\n";

    // Metrics HTTP server on background thread
    std::thread metrics_thread(run_metrics_server, metrics_port, &ctrl,
                                &g_running);

    // Block until signal
    while (g_running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

    server->Shutdown();
    metrics_thread.join();
    return 0;
}

#else  // TITANINFER_USE_GRPC not defined

#include <cstdio>
int main() {
    std::puts("Rebuild with -DUSE_GRPC=ON to enable the cluster server.");
    return 1;
}

#endif // TITANINFER_USE_GRPC
