"""Create (or verify with --check) SHA256SUMS.txt for every file of the project, excluding the manifest itself,
virtual environments, caches and the _work/ scratch directory.

    python verification/make_manifest.py            # write SHA256SUMS.txt
    python verification/make_manifest.py --check    # verify; exit code 0 iff every listed file matches and none is missing/extra
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {"__pycache__", "_work", ".git", "_eqcache", "_figcache", "venv", ".venv"}
SKIP_FILES = {"SHA256SUMS.txt", "report_text_dump.json"}
SKIP_SUFFIX = (".pyc", ".aux", ".log", ".out", ".toc")


def files():
    out = []
    for d, dirs, fs in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(fs):
            if f in SKIP_FILES or f.endswith(SKIP_SUFFIX):
                continue
            out.append(os.path.relpath(os.path.join(d, f), ROOT).replace("\\", "/"))
    return sorted(out)


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    man = os.path.join(ROOT, "SHA256SUMS.txt")
    if not a.check:
        with open(man, "w", encoding="utf-8", newline="\n") as f:
            for p in files():
                f.write("%s  %s\n" % (sha(p), p))
        print("wrote %d entries" % len(files()))
        return
    listed = {}
    for line in open(man, encoding="utf-8"):
        if line.strip():
            h, p = line.rstrip("\n").split("  ", 1)
            listed[p] = h
    bad = []
    for p, h in listed.items():
        if not os.path.exists(os.path.join(ROOT, p)):
            bad.append("MISSING " + p)
        elif sha(p) != h:
            bad.append("MISMATCH " + p)
    for p in files():
        if p not in listed:
            bad.append("NOT LISTED " + p)
    print("manifest entries: %d; problems: %d" % (len(listed), len(bad)))
    for b in bad[:40]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
