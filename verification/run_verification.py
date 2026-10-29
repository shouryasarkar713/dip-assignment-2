"""Run the documented verification commands and record exit codes, durations and output tails in a JSON log.

    python verification/run_verification.py [--log verification_log.json] [--skip-regression] [--compare <dir>]

Commands: unit tests; regression_fast (recompute-and-compare); manifest check; report consistency check; optionally
compare_results.py between results/ and another results directory (e.g. from a full clean re-run).
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(name, cmd, log):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    out = (r.stdout or "").strip().splitlines()
    err = (r.stderr or "").strip().splitlines()
    log.append({"name": name, "command": " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c for c in cmd),
                "exit_code": r.returncode, "seconds": round(dt, 1), "stdout_tail": out[-6:], "stderr_tail": err[-4:]})
    print("%-28s exit=%d  %.0fs" % (name, r.returncode, dt), flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "verification_log.json"))
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    log = []
    py = sys.executable
    run("unit tests", [py, "-m", "unittest", "discover", "-s", os.path.join(ROOT, "tests"), "-t", os.path.join(ROOT, "tests")], log)
    if not a.skip_regression:
        run("regression_fast", [py, os.path.join(ROOT, "verification", "regression_fast.py")], log)
    run("manifest check", [py, os.path.join(ROOT, "verification", "make_manifest.py"), "--check"], log)
    run("report consistency", [py, os.path.join(ROOT, "verification", "check_report.py")], log)
    if a.compare:
        run("compare results", [py, os.path.join(ROOT, "verification", "compare_results.py"), os.path.join(ROOT, "results"), a.compare], log)
    with open(a.log, "w") as f:
        json.dump(log, f, indent=1)
    sys.exit(max(x["exit_code"] for x in log))


if __name__ == "__main__":
    main()
