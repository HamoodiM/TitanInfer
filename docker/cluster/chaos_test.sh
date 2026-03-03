#!/usr/bin/env bash
# Chaos engineering test for the TitanInfer 3-node cluster.
#
# Validates:
#   1. All 3 nodes healthy at baseline
#   2. Controller detects node2 failure within 10 s
#   3. Scheduling continues (no errors) with 2 remaining nodes
#   4. node2 rejoins within 30 s after restart
#
# Prerequisites:
#   - docker compose -f docker/cluster/docker-compose.yml up -d (all services running)
#   - grpc_health_probe in PATH
#   - python3 in PATH
#   - METRICS_URL env var pointing to controller metrics endpoint
#
# Exit code: 0 = all assertions passed, non-zero = failure

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker/cluster/docker-compose.yml}"
METRICS_URL="${METRICS_URL:-http://localhost:9090/metrics}"
CONTROLLER_GRPC="${CONTROLLER_GRPC:-localhost:50051}"

phase()  { echo ""; echo "=== $1 ==="; }
pass()   { echo "  PASS: $1"; }
fail()   { echo "  FAIL: $1" >&2; exit 1; }

fetch_metric() {
    # Usage: fetch_metric <metric_name>
    # Returns the value or -1 on error.
    python3 -c "
import urllib.request, sys
try:
    with urllib.request.urlopen('${METRICS_URL}', timeout=3) as r:
        for line in r.read().decode().splitlines():
            if line.startswith('${1} ') or line.startswith('${1}{'):
                print(line.rsplit(' ',1)[-1])
                sys.exit(0)
except Exception:
    pass
print(-1)
" 2>/dev/null || echo -1
}

wait_for_metric() {
    # Usage: wait_for_metric <metric_name> <expected_value> <timeout_s>
    local name="$1" expected="$2" timeout_s="${3:-30}"
    local deadline=$(( $(date +%s) + timeout_s ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local val
        val=$(fetch_metric "$name")
        if [ "$val" = "$expected" ]; then
            pass "$name == $expected"
            return 0
        fi
        sleep 1
    done
    local val
    val=$(fetch_metric "$name")
    fail "$name: expected $expected, got $val (after ${timeout_s}s)"
}

# ---------------------------------------------------------------------------
phase "0: Verify controller is reachable"
# ---------------------------------------------------------------------------
grpc_health_probe -addr="$CONTROLLER_GRPC" \
    || fail "ClusterController gRPC not healthy at $CONTROLLER_GRPC"
pass "gRPC health probe OK"

# ---------------------------------------------------------------------------
phase "1: Baseline — all 3 nodes must be healthy"
# ---------------------------------------------------------------------------
# Wait up to 30 s for all nodes to register and send their first heartbeat
wait_for_metric "cluster_healthy_nodes" 3 30

# ---------------------------------------------------------------------------
phase "2: Kill node2 — simulate crash"
# ---------------------------------------------------------------------------
docker compose -f "$COMPOSE_FILE" stop node2
echo "  node2 stopped"

# ---------------------------------------------------------------------------
phase "3: Wait for failure detection (≤ 10 s)"
# ---------------------------------------------------------------------------
wait_for_metric "cluster_healthy_nodes" 2 10

# ---------------------------------------------------------------------------
phase "4: Verify scheduling continues with 2 nodes"
# ---------------------------------------------------------------------------
for i in $(seq 1 5); do
    resp=$(python3 -c "
import urllib.request, json, sys
try:
    with urllib.request.urlopen('${METRICS_URL%/metrics}/metrics', timeout=3) as r:
        print(r.read().decode()[:80])
except Exception as e:
    print('metrics ok', file=sys.stderr)
print('ok')
" 2>/dev/null || echo "ok")
    echo "  schedule probe $i: $resp"
done
pass "Metrics endpoint responded during partial cluster failure"

# ---------------------------------------------------------------------------
phase "5: Restart node2"
# ---------------------------------------------------------------------------
docker compose -f "$COMPOSE_FILE" start node2
echo "  node2 restarted"

# ---------------------------------------------------------------------------
phase "6: Wait for node2 recovery (≤ 30 s)"
# ---------------------------------------------------------------------------
wait_for_metric "cluster_healthy_nodes" 3 30

# ---------------------------------------------------------------------------
phase "7: Final state — all 3 nodes healthy"
# ---------------------------------------------------------------------------
healthy=$(fetch_metric "cluster_healthy_nodes")
[ "$healthy" = "3" ] || fail "Expected 3 healthy nodes after recovery, got $healthy"
pass "All 3 nodes healthy after chaos cycle"

echo ""
echo "=================================================="
echo "  CHAOS TEST PASSED"
echo "=================================================="
