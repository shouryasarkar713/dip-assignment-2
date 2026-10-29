# Digital Image Processing - Assignment 2 (Questions 1-6)

Student: Shourya Sarkar, roll number 2023112013.
Scope: Questions 1-6 of the assignment sheet. Question 7 is not part of this submission.
Repository (public): https://github.com/shouryasarkar713/dip-assignment-2

## Contents
| Path | Contents |
|---|---|
| `report/DIP_A2_Report.pdf`, `report/DIP_A2_Report.docx` | The report (PDF, and editable DOCX with equations as images). `report/DIP_A2_Report.tex` is the LaTeX source of the PDF. |
| `report/build_report.py`, `report/content_*.py`, `report/render.py` | Report sources: one content model rendered to both PDF and DOCX; numbers are read from `results/*.json`. |
| `src/q1_dft.py` ... `src/q6_motion.py` | One implementation per question; `src/common.py` holds shared metrics and I/O. |
| `predictions/` | The dated analytical predictions required by Sections 1.2 and 2.2. |
| `results/q1.json` ... `results/q6.json` | Numerical results written by the experiments (machine readable). |
| `figures/` | Figures used in the report. |
| `data/provided/` | Images supplied with the assignment. |
| `data/photos/` | The four original photographs (unmodified; SHA-256 in `data/photos_original_sha256.json`). |
| `data/derived/` | Resized or cropped copies produced by the experiments; originals are never overwritten. |
| `tests/` | Unit and property tests. |
| `verification/` | Test, regression and report-consistency utilities, and `VERIFICATION_RECORD.md`. |
| `ASSIGNMENT_REQUIREMENTS.md` | Requirement-by-requirement checklist of Questions 1-6 with the location of each answer. |
| `SHA256SUMS.txt` | SHA-256 checksums of all files in the archive. |

## Setup
Python 3.11 and the packages in `requirements.txt` (`pip install -r requirements.txt`). Building the PDF needs `pdflatex`. All paths are relative to the project root.

## Running
```bash
python run_all.py                                      # all six experiments, writes results/*.json and figures/ (about 30 min)
python src/q2_filtering.py                             # a single question (q1_dft, q2_filtering, q3_homomorphic, q4_edge_preserving, q5_wiener, q6_motion)
python -m unittest discover -s tests -t tests          # unit and property tests
python verification/regression_fast.py                 # recomputes selected quantities and compares them with results/*.json
python report/build_report.py                          # regenerate report/DIP_A2_Report.{tex,pdf,docx}
python verification/check_report.py                    # consistency checks of the built report
python verification/make_manifest.py --check           # verify SHA256SUMS.txt
```
`q6_motion.py` reads `data/derived/reference_work_1536x1024.png`, which `q5_wiener.py` creates, so run `q5` before `q6`. Run the experiments one after the other rather than in parallel (peak memory is a few GB). Approximate run times on the 16-thread i5-1340P used here: q1 43 s, q2 150 s, q3 70 s, q4 430 s, q5 630 s, q6 450 s.

## Notes
* All randomness uses `numpy.random.default_rng` with fixed seeds (base seed 2026). Results are reproducible to floating-point rounding on the same NumPy/SciPy versions.
* The photographs are not pixel-aligned clean/degraded pairs, so no PSNR/SSIM is computed on them; real-photo results are qualitative or spectral/noise measurements.
* Test and verification details are in `verification/VERIFICATION_RECORD.md`.
