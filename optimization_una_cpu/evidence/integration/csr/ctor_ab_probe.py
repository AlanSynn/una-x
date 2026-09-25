import sys, time
sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/wt-integration/benchmarks/una_cpu")
from workload_gen import settings_for
import numpy as np
from urban_network_analysis import Settings
from urban_network_analysis.Topology import Topology
from urban_network_analysis.Engines.AccessibilityWElevation import AccessibilityWElevation

s = Settings()
for k, v in settings_for("W3_medium_proxy", "/Users/alansynn/orca/workspaces/una-x/bench_work/fixtures/W3_medium_proxy", "/tmp/ctor_out", "accessibility").items():
    if k == "knn_weights":
        v = np.asarray(v, dtype=np.float64)
    setattr(s, k, v)

topo = Topology(verbosity=0)
topo.AddNetwork(s); topo.AddOrigins(s); topo.AddDestinations(s); topo.AddObstacles(s)

sn = topo.network.start_nodes; en = topo.network.end_nodes
print("endpoint dtype:", sn.dtype, "E:", len(sn), "node_count:", topo.network.node_points.shape[0])

# direct builder timing on the actual topology, best of 5
walls = []
for _ in range(5):
    t0 = time.perf_counter_ns()
    AccessibilityWElevation._build_compact_graph_from_topology(topo, 4.0)
    walls.append((time.perf_counter_ns() - t0) / 1e6)
print("builder walls ms:", [round(w, 2) for w in walls])
