#!/usr/bin/env python3
import json, os, platform
try:
    import psutil
except Exception:
    psutil=None
out={
"platform":platform.platform(),
"machine":platform.machine(),
"python":platform.python_version(),
"logical_cpus":os.cpu_count(),
"affinity":None,
"physical_cpus":None,
"memory_total_bytes":None,
"memory_available_bytes":None,
"swap_total_bytes":None,
}
if psutil:
    out["physical_cpus"]=psutil.cpu_count(logical=False)
    vm=psutil.virtual_memory(); out["memory_total_bytes"]=vm.total; out["memory_available_bytes"]=vm.available
    out["swap_total_bytes"]=psutil.swap_memory().total
    try: out["affinity"]=psutil.Process().cpu_affinity()
    except Exception: pass
print(json.dumps(out,indent=2,sort_keys=True))
