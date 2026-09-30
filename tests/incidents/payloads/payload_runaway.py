"""Watchdog-verification payload: a deliberately runaway child that
ignores the cooperative cancel flag (worst case), so the watchdog must
do the bounded-grace SIGTERM itself.  Used only by the watchdog/cleanup
verification test.
"""
import time


def payload(params, state):
    state["phase"] = "runaway"
    deadline = time.time() + float(params.get("run_for_s", 600))
    while time.time() < deadline:  # deliberately ignores cancel_file
        time.sleep(0.1)
    return {"never": True}
