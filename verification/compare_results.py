"""Compare two results directories (JSON files) number by number with explicit tolerances.

    python verification/compare_results.py <dirA> <dirB> [--rtol 1e-6] [--atol 1e-9]

Keys that contain timing information are skipped (hardware dependent, not a numerical result):
'seconds', 'time', 'timings', 't_', 'timing', 'mtime', 'experiment_run_time', 'cpu_note', 'env'.
Exit code 0 iff every compared number agrees within  |a-b| <= atol + rtol*max(|a|,|b|).
"""
import argparse
import json
import math
import os
import sys

SKIP_EXACT = {"bilateral_s", "kuwahara_s", "soft_kuwahara_s", "bilateral_s_with_radius_N//2", "env", "cpu_note", "timings_512x512", "seed_note"}


def skip(key):
    k = str(key)
    return k in SKIP_EXACT or k.startswith("t_") or "seconds" in k or "time" in k or k == "timing"


def walk(a, b, path, out, rtol, atol, stats):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if skip(k):
                continue
            if k not in b:
                out.append("%s/%s: missing in B" % (path, k))
                continue
            walk(a[k], b[k], path + "/" + str(k), out, rtol, atol, stats)
        for k in b:
            if k not in a and not skip(k):
                out.append("%s/%s: missing in A" % (path, k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s: length %d vs %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), out, rtol, atol, stats)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        stats["n"] += 1
        d = abs(a - b)
        tol = atol + rtol * max(abs(a), abs(b))
        if not (d <= tol or (math.isnan(a) and math.isnan(b))):
            out.append("%s: %r vs %r (diff %.3g > tol %.3g)" % (path, a, b, d, tol))
        stats["maxrel"] = max(stats["maxrel"], d / (max(abs(a), abs(b)) + 1e-300) if d > 1e-300 else 0.0)
    else:
        if a != b and not (isinstance(a, str) and isinstance(b, str) and ("20" in a and ":" in a)):
            if type(a) != type(b) or a != b:
                out.append("%s: %r vs %r" % (path, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ar = ap.parse_args()
    bad = []
    stats = {"n": 0, "maxrel": 0.0}
    files = sorted(f for f in os.listdir(ar.a) if f.startswith("q") and f.endswith(".json"))
    for f in files:
        pa, pb = os.path.join(ar.a, f), os.path.join(ar.b, f)
        if not os.path.exists(pb):
            bad.append("%s: missing in B" % f)
            continue
        walk(json.load(open(pa)), json.load(open(pb)), f, bad, ar.rtol, ar.atol, stats)
    print("compared %d numbers in %d files; mismatches: %d; largest relative difference %.3g" % (stats["n"], len(files), len(bad), stats["maxrel"]))
    for m in bad[:60]:
        print("  ", m)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
