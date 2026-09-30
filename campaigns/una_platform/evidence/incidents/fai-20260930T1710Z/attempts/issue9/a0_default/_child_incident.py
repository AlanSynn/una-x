
import faulthandler, json, sys, threading, time, os, resource
from pathlib import Path
repo, attempt_dir, payload_path, params_path = sys.argv[1:5]
sys.path.insert(0, repo)
attempt_dir = Path(attempt_dir)
log = (attempt_dir / "child_log.txt").open("w")
faulthandler.enable(log)
def _dump(signum, frame):
    faulthandler.dump_traceback(file=log)
    log.flush()
import signal as _signal
_signal.signal(_signal.SIGUSR1, _dump)

state = {"phase": "boot", "units_done": 0, "last_marker": "start"}
hb_path = attempt_dir / "heartbeat.jsonl"
stop_hb = threading.Event()
def _hb():
    import resource
    while not stop_hb.is_set():
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            rec = {"t": time.time(), **state, "ru_maxrss_bytes": rss}
            with hb_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        stop_hb.wait(float(os.environ.get("INCIDENT_HEARTBEAT_S", "2.0")))
threading.Thread(target=_hb, daemon=True).start()

import runpy
payload_ns = runpy.run_path(payload_path)
params = json.loads(Path(params_path).read_text())
cancel_path = attempt_dir / "cancel.flag"
params["cancel_file"] = str(cancel_path)
state["phase"] = "payload"
try:
    result = payload_ns["payload"](params, state)
    state["phase"] = "done"
    (attempt_dir / "result.json").write_text(json.dumps(
        {"status": "completed", "result": result}, default=str))
except BaseException as exc:
    import traceback
    state["phase"] = "exception"
    (attempt_dir / "result.json").write_text(json.dumps(
        {"status": "failed",
         "error": f"{type(exc).__name__}: {exc}",
         "traceback": traceback.format_exc()}, default=str))
finally:
    stop_hb.set()
    log.flush()
