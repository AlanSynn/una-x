import contextlib, io, json, struct, sys
REPO = "/storage/scratch1/1/dsynn6/una-x"
sys.path.insert(0, f"{REPO}/tests/madina_api/flow")
import _flow_scenario as fs
fs.import_arm("facade")
mz = fs._ARM_MARKER["zonal"]; btd = fs._ARM_MARKER["betweenness"]
z = fs._make_zonal(mz)
_ = fs._seeded_origin_order(z, 20260930)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = btd.paralell_betweenness_exposure(
        z, search_radius=250, detour_ratio=1.0, decay=False,
        beta=0.003, num_cores=2, closest_destination=False)
print(json.dumps({int(k): struct.pack('>d', float(v)).hex()
                  for k, v in res["edge_gdf"]["betweenness"].items()}))
