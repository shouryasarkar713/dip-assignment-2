"""Build the final report (PDF via LaTeX and DOCX via python-docx) from results/*.json and figures/*.png.

    python report/build_report.py

Both files are produced from ONE block list (content_*.py -> render.py); verification/check_report.py checks that the
two outputs contain the same text, tables and numbers, and that every Table/Figure reference resolves.
"""
import json
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

import content_a as A  # noqa: E402
import content_b as B  # noqa: E402
import content_c as Cc  # noqa: E402
import content_d as D  # noqa: E402
import content_e as E  # noqa: E402
import content_f as F  # noqa: E402
import content_g as G  # noqa: E402
import render  # noqa: E402


def blocks():
    R = {k: A.load("%s.json" % k) for k in ("q1", "q2", "q3", "q4", "q5", "q6")}
    out = [A.cover(R["q1"])]
    out.append(("h1", "Summary of findings"))
    out += G.summary(R)
    out += A.section0(R["q1"])
    out.append(A.photo_table())
    out += A.section1(R["q1"])
    out += A.section2(R["q2"])
    out += B.section3(R["q3"])
    out += B.section4(R["q4"])
    out += D.section4_all(R["q4"])
    out += Cc.section5(R["q5"])
    out += E.section6_a(R["q6"])
    out += F.section6_b(R["q6"])
    out += G.closing(R)
    return out


def main():
    bl = blocks()
    bad = render.check_references(bl)
    if bad:
        for b in bad:
            print("REFERENCE PROBLEM:", b)
        raise SystemExit("unresolved Table/Figure references (see above)")
    pdf = render.build_pdf(bl, HERE)
    docx = os.path.join(HERE, "DIP_A2_Report.docx")
    render.to_docx(bl, docx, os.path.join(HERE, "_eqcache"))
    with open(os.path.join(HERE, "report_text_dump.json"), "w", encoding="utf-8") as f:
        json.dump([b for b in bl if b[0] in ("p", "h1", "h2", "h3", "table", "bullets", "eq")], f, ensure_ascii=False)
    print("PDF:", pdf)
    print("DOCX:", docx)


if __name__ == "__main__":
    main()
