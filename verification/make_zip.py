"""Create the submission archive from the project tree (same file selection as the SHA-256 manifest).

    python verification/make_zip.py <output.zip>

The archive has the single top-level folder DIP_A2_GITHUB_SUBMISSION_FINAL/ and contains every manifest file plus SHA256SUMS.txt.
"""
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import make_manifest  # noqa: E402


def main():
    out = sys.argv[1]
    names = make_manifest.files() + ["SHA256SUMS.txt"]
    top = "DIP_A2_GITHUB_SUBMISSION_FINAL/"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in sorted(set(names)):
            z.write(os.path.join(ROOT, n), top + n)
    print("wrote %s with %d files, %.1f MB" % (out, len(set(names)), os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
