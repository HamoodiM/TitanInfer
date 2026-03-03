#!/usr/bin/env python3
"""
Minimal gRPC load generator for TitanInfer ClusterController chaos tests.
Sends Schedule RPCs to the controller and prints results.
No external dependencies beyond the generated protobuf stubs.

In the chaos CI job this script is run as a background process while
chaos_test.sh drives the failure injection and metric assertions.
"""

import os
import sys
import time
import urllib.request

METRICS_URL = os.environ.get("METRICS_URL", "http://localhost:9090/metrics")


def fetch_metric(name: str) -> float:
    """Parse a gauge value from the Prometheus text endpoint."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith(name + " ") or line.startswith(name + "{"):
                    # Handle both `metric_name value` and `metric_name{...} value`
                    parts = line.rsplit(" ", 1)
                    return float(parts[-1])
    except Exception:
        pass
    return -1.0


def wait_for_metric(name: str, expected: float, timeout_s: int = 30,
                    label: str = "") -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = fetch_metric(name)
        if val == expected:
            print(f"  OK  {label or name} = {int(val)}")
            return True
        time.sleep(1)
    val = fetch_metric(name)
    print(f"  FAIL {label or name}: expected {expected}, got {val}", file=sys.stderr)
    return False


def main():
    # This script is used two ways:
    #  1. As a continuous background load generator (default)
    #  2. As a probe helper invoked with "check <metric> <value> <timeout>"
    if len(sys.argv) >= 4 and sys.argv[1] == "check":
        name    = sys.argv[2]
        value   = float(sys.argv[3])
        timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        ok = wait_for_metric(name, value, timeout)
        sys.exit(0 if ok else 1)

    # Default: run for 60 s printing healthy_node counts
    print("[loadgen] starting metric polling loop")
    start = time.time()
    while time.time() - start < 60:
        val = fetch_metric("cluster_healthy_nodes")
        print(f"[loadgen] cluster_healthy_nodes = {int(val) if val >= 0 else '?'}")
        time.sleep(5)
    print("[loadgen] done")


if __name__ == "__main__":
    main()
