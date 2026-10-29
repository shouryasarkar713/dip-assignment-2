"""Consistency checks of the built report (PDF and DOCX) against the block model / results.

    python verification/check_report.py            # exit 0 iff every check passes

Checks: (1) both files exist and are non-trivial; (2) the PDF has no unresolved references ('??'), placeholders or NaN;
(3) every heading appears in the PDF text and in the DOCX; (4) table count and the text of every table cell agree between
the block model, the DOCX and (for numeric cells) the PDF; (5) the number of figures/equation images in the DOCX equals
the block model, the PDF contains at least that many embedded images; (6) every Table/Figure reference resolves;
(7) the cover shows the student name and roll number, and no e-mail address, development-history wording, machine-specific path or placeholder text appears in the PDF, the DOCX or the LaTeX source; the DOCX has no comments or tracked changes and its title and author are set; (8) key numbers of results/*.json occur in the PDF.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "report"))
res = []


def chk(name, ok, detail=""):
    res.append(bool(ok))
    print("%-4s %-80s %s" % ("PASS" if ok else "FAIL", name, detail))


def norm(s):
    s = s.replace("−", "-").replace("–", "-").replace(" ", " ")
    return re.sub(r"\s+", "", s)


def main():
    import pymupdf
    from docx import Document
    import build_report
    import render
    pdf_p = os.path.join(ROOT, "report", "DIP_A2_Report.pdf")
    docx_p = os.path.join(ROOT, "report", "DIP_A2_Report.docx")
    chk("PDF exists and > 1 MB", os.path.exists(pdf_p) and os.path.getsize(pdf_p) > 1e6, str(os.path.getsize(pdf_p)) if os.path.exists(pdf_p) else "")
    chk("DOCX exists and > 1 MB", os.path.exists(docx_p) and os.path.getsize(docx_p) > 1e6, str(os.path.getsize(docx_p)) if os.path.exists(docx_p) else "")
    blocks = build_report.blocks()
    bad_refs = render.check_references(blocks)
    chk("all Table/Figure references in the text resolve", not bad_refs, "; ".join(bad_refs[:3]))
    doc = pymupdf.open(pdf_p)
    pdf_text = "\n".join(p.get_text() for p in doc)
    pdf_norm = norm(pdf_text)
    chk("PDF page count", len(doc) >= 40, "%d pages" % len(doc))
    for bad in ("??", "NaN", "nan", "TODO", "INSERT", "XXX", "Traceback"):
        if bad == "nan":
            chk("PDF contains no stand-alone 'nan'", re.search(r"nan", pdf_text) is None)
        else:
            chk("PDF contains no '%s'" % bad, bad not in pdf_text)
    d = Document(docx_p)
    d_text = "\n".join(p.text for p in d.paragraphs)
    for bad in ("**", "\\frac", "\\sigma", "NaN", "TODO", "INSERT", "$"):
        chk("DOCX body contains no raw '%s'" % bad, bad not in d_text)
    # headings
    heads = [b[1] for b in blocks if b[0] in ("h1", "h2", "h3")]
    miss_pdf = [h for h in heads if norm(re.sub(r"[$`*]", "", h))[:40] not in pdf_norm]
    d_heads = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
    miss_docx = [h for h in heads if not any(norm(re.sub(r"[$`*]", "", h))[:30] in norm(x) for x in d_heads)]
    chk("all %d headings present in PDF" % len(heads), not miss_pdf, "; ".join(miss_pdf[:3]))
    chk("all %d headings present in DOCX" % len(heads), not miss_docx, "; ".join(miss_docx[:3]))
    # tables
    tabs = [b for b in blocks if b[0] == "table"]
    chk("table count: model %d == DOCX %d" % (len(tabs), len(d.tables) - sum(1 for b in blocks if b[0] == "eq")), len(tabs) == len(d.tables) - sum(1 for b in blocks if b[0] == "eq"))
    # DOCX equation layout tables (1x2) are interleaved with data tables: identify data tables by header text
    data_tables = [t for t in d.tables if len(t.rows) > 0 and len(t.columns) >= 2 and not (len(t.columns) == 2 and len(t.rows) == 1 and t.rows[0].cells[1].text.strip().startswith("("))]
    chk("DOCX data-table count equals model", len(data_tables) == len(tabs), "%d vs %d" % (len(data_tables), len(tabs)))
    cell_mismatch = 0
    num_missing_pdf = 0
    n_num = 0
    for b, t in zip(tabs, data_tables):
        for r_model, r_doc in zip(b[3], t.rows[1:]):
            for v, c in zip(r_model, r_doc.cells):
                s = str(v)
                if not re.search(r"\d", s):
                    continue
                d_cell = norm(c.text)
                plain = norm("".join(render.tex_to_text(st) if kind == "math" else st for kind, st, *_ in render.inline_tokens(s)))
                if plain != d_cell:
                    cell_mismatch += 1
                if re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", s.strip()):
                    n_num += 1
                    if norm(s) not in pdf_norm:
                        num_missing_pdf += 1
    chk("DOCX table cells equal the model's strings", cell_mismatch == 0, "%d mismatching cells" % cell_mismatch)
    chk("numeric table cells (%d) present in the PDF text" % n_num, num_missing_pdf <= 0.01 * max(n_num, 1), "%d missing" % num_missing_pdf)
    # figures / equations
    n_fig = sum(1 for b in blocks if b[0] == "fig")
    n_eq = sum(1 for b in blocks if b[0] == "eq")
    chk("DOCX inline images = figures + equations (%d + %d)" % (n_fig, n_eq), len(d.inline_shapes) == n_fig + n_eq, "%d" % len(d.inline_shapes))
    n_pdf_img = sum(len(p.get_images()) for p in doc)
    chk("PDF embedded images >= figures (%d)" % n_fig, n_pdf_img >= n_fig, "%d" % n_pdf_img)
    # cover / open item
    chk("cover: student name and roll number present in PDF", "Shourya Sarkar" in pdf_text and "2023112013" in pdf_text)
    chk("cover: same in DOCX", "Shourya Sarkar" in d_text and "2023112013" in d_text)
    for bad in ("@", "e-mail", "email", "inferred", "NOT SUPPLIED", "supplied by the student", "administrative field", "placeholder"):
        chk("no '%s' in the PDF or DOCX text" % bad, bad.lower() not in pdf_text.lower() and bad.lower() not in d_text.lower())
    docx_all = d_text + "\n" + "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    for bad in ("previous submission", "earlier submission", "previous report", "earlier version", "rebuild", "FINAL_AUDIT", "Discrepancy matrix", "iteration", "pre-registered", "agent", "REBUILD", "E:/", "E:\\"):
        chk("no process/history wording '%s' in the PDF or DOCX" % bad, bad.lower() not in pdf_text.lower() and bad.lower() not in docx_all.lower())
    tex_p = os.path.join(ROOT, "report", "DIP_A2_Report.tex")
    with open(tex_p, encoding="utf-8") as fh:
        tex = fh.read()
    chk("LaTeX source contains no machine-specific absolute paths", re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\]\w", tex) is None)
    import zipfile
    with zipfile.ZipFile(docx_p) as zf:
        parts = zf.namelist()
        body = zf.read("word/document.xml").decode("utf-8")
    chk("DOCX has no comments part and no tracked changes", "word/comments.xml" not in parts and "<w:ins " not in body and "<w:del " not in body)
    cp = d.core_properties
    chk("DOCX core properties: title and author set", bool(cp.title) and cp.author == "Shourya Sarkar", "title=%r author=%r" % (cp.title, cp.author))
    h1 = [b[1] for b in blocks if b[0] == "h1"]
    chk("top-level sections: 0, Questions 1-6, 7 (conclusions), 8 (code and reproducibility); no section 9", h1[1].startswith("0. ") and h1[-2].startswith("7. ") and h1[-1].startswith("8. ") and not any(re.match(r"9\. ", h) for h in h1), "; ".join(h[:14] for h in h1))
    chk("no stale 'Section 9' reference in the PDF text", re.search(r"Section\s+9\b", pdf_text) is None)
    url = "https://github.com/shouryasarkar713/dip-assignment-2"
    chk("repository URL appears in the PDF and in the DOCX", norm(url) in pdf_norm and norm(url) in norm(docx_all))
    chk("repository URL in the report equals the README repository URL", url in open(os.path.join(ROOT, "README.md"), encoding="utf-8").read())
    flat = pdf_text.replace("\n", " ")
    chk("scope statement (Questions 1-6; Question 7 not part of the submission) in PDF", "Questions 1-6" in flat and "Question 7" in flat and "not part of this submission" in flat)
    # key numbers from the JSON
    q4 = json.load(open(os.path.join(ROOT, "results", "q4.json")))
    q6 = json.load(open(os.path.join(ROOT, "results", "q6.json")))
    q1 = json.load(open(os.path.join(ROOT, "results", "q1.json")))
    chk("131,074 classes in PDF", "131074" in pdf_norm or "131,074" in pdf_text)
    chk("NCR* from q4.json in PDF", ("%.3f" % q4["q4_3_breakdown"]["results"]["sigma_s=2,rule=A"]["NCR*_retention_0.5"]) in pdf_norm)
    chk("estimated theta, L from q6.json in PDF", ("%.1f" % q6["q6_4_tile_estimate"]["theta_hat_deg"]) in pdf_norm and ("%.1f" % q6["q6_4_tile_estimate"]["L_hat_px"]) in pdf_norm)
    chk("K_opt (30 dB) from q6.json in PDF", ("%.3g" % q6["q6_3_levels"]["30"]["K_opt"]) in pdf_norm)
    print("\n%d / %d checks passed" % (sum(res), len(res)))
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
