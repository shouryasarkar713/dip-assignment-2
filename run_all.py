"""Run every experiment (Questions 1-6) from the command line and record exit codes and wall times.

    python run_all.py              # full run, writes results/*.json, figures/*.png, results/run_log.json
    python run_all.py --only q1 q2 # subset

Order matters only for Q6, which reads data/derived/reference_work_1536x1024.png created by q5_wiener.py.
Timings are recorded for information; they are hardware dependent and are never compared.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.dirname(__file__))
STEPS = [("q1", "q1_dft.py"), ("q2", "q2_filtering.py"), ("q3", "q3_homomorphic.py"), ("q4", "q4_edge_preserving.py"),
         ("q5", "q5_wiener.py"), ("q6", "q6_motion.py")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    log = []
    for name, script in STEPS:
        if a.only and name not in a.only:
            continue
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", script)], cwd=os.path.join(ROOT, "src"), capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.perf_counter() - t0
        warn = [l for l in (r.stderr or "").splitlines() if "warn" in l.lower()]
        log.append({"step": name, "script": script, "exit_code": r.returncode, "seconds": round(dt, 1), "warnings": warn[:10], "stdout_tail": r.stdout.strip().splitlines()[-3:], "stderr_tail": (r.stderr or "").strip().splitlines()[-5:]})
        print("%s exit=%d %.0fs" % (name, r.returncode, dt), flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            break
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "run_log.json" if not a.only else "run_log_partial.json"), "w") as f:
        json.dump(log, f, indent=1)
    sys.exit(max([x["exit_code"] for x in log] + [0]))


if __name__ == "__main__":
    main()
