# Verification record - DIP Assignment 2 (Questions 1-6)

This file records how the code, results and report in this archive were checked. It is separate from the report and is not part of the assignment answers.

## 1. Environment
13th Gen Intel Core i5-1340P (12 cores / 16 threads), 15.6 GB RAM, Windows 11, Python 3.11.0, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9; `pdflatex` (TinyTeX) for the PDF, `python-docx` for the DOCX.

## 2. Unit and property tests (`tests/`, 61 tests)
Command: `python -m unittest discover -s tests -t tests`. Independent references are used wherever possible (literal DFT sum, direct spatial convolution, brute-force single-pixel filters, Monte-Carlo weight ratios, scikit-image SSIM, exact Dirichlet zeros).

| Module | Tests | Checked |
|---|---|---|
| `test_common_metrics.py` | 6 | PSNR/SSIM definitions, input validation, SSIM against scikit-image |
| `test_q1.py` | 10 | DFT against a literal sum and `numpy.fft`, inverse, Parseval, conjugate symmetry, conjugacy-class counts, shift and convolution theorems, sinusoid recovery, padding at N+M-1 |
| `test_q2.py` | 10 | `make_filter` values and validation, matched Gaussian, inverse design, Gibbs overshoot on an x-only strip, kernel ratios, sigma_s against `scipy.ndimage` |
| `test_q3.py` | 6 | log-domain identity, filter limits, identity for unit gains, illumination flattening, histogram equalisation |
| `test_q4.py` | 11 | bilateral and Kuwahara against brute force, limiting cases, step preservation, noise estimator, closed-form weight ratio against Monte-Carlo |
| `test_q5.py` | 7 | Wiener MSE reduction and limits, mis-specified noise power, noise estimator |
| `test_q6.py` | 8 | PSF normalisation, OTF against the aliased sinc-triangle sum, Dirichlet zeros, Wiener limits, blur estimator on known angle and length |
| `test_data_integrity.py` | 3 | photograph sizes, EXIF ISO/exposure, SHA-256 against the recorded values, provided image sizes |

`python verification/regression_fast.py` recomputes selected quantities from scratch (DFT error, class count, shift theorem, phase statistics, ILPF overshoot, Butterworth kernel ratio, inverse design, alpha(r) radius, Q2.3a PSNR, Q3 shadow statistics, Q4 and Q5 PSNR/SSIM samples, Q6 PSNR at the stored K_opt) and compares them with `results/*.json`. It is a spot check, not a re-run of the experiments.

## 3. Report checks (`python verification/check_report.py`, 63 checks)
The PDF and DOCX are rendered from one content model (`report/content_*.py`) by `report/render.py`. The checker verifies that both files exist; that there are no unresolved references, NaN, raw markup or placeholders; that every heading, table (including every cell) and figure/equation image of the content model is present in the DOCX and, for numbers, in the PDF; that all Table/Figure references resolve; that the cover shows the student name and roll number; that no e-mail address, development-history wording or machine-specific path occurs in the PDF, DOCX or LaTeX source; that the DOCX has no comments or tracked changes and has its title and author set; that the top-level sections are 0, Questions 1-6, 7 and 8; that the repository URL in the report is a clickable link in the PDF and the DOCX and equals the one in `README.md`, and that the report describes the repository as public; and that key numbers from the JSON files occur in the PDF.

Layout was inspected by rendering all 49 PDF pages, and by exporting the DOCX through Microsoft Word (51 pages) and inspecting the pages; no clipped text or table, raw markup or blank page was found. The DOCX equations are pictures, and inline mathematics is Unicode text.

## 4. Experiments and result files
`python run_all.py` regenerates all results and figures (about 30 min). The files in `src/`, `data/`, `figures/`, `results/q1.json` ... `results/q6.json`, `tests/`, `predictions/` and `run_all.py` are byte-identical (SHA-256) to those with which a complete regeneration was run from an extracted copy of the project: all six scripts exited with code 0 and `verification/compare_results.py` found 5,362 compared numbers identical (largest relative difference 0); the 43 figures and 14 derived images were byte-identical on that machine. The experiments were not re-run for the final edit of the report, which changed only the report text, its renderer, the documentation and two utility scripts.

The four photographs in `data/photos/` are unmodified: their SHA-256 values equal those in `data/photos_original_sha256.json` and the supplied archive.

| ID | File | EXIF (Nothing/CMF Phone 2 Pro) | Use |
|---|---|---|---|
| A | `IMG_20261026_195540763.jpg` | ISO 734, 1/14 s, f/1.87 | qualitative comparison scene for the noise analysis |
| B | `IMG_20261026_195556198.jpg` | ISO 6400, 1/16.7 s, f/1.87 | real low-light photograph (Problem 5.2) |
| C | `IMG_20261026_195635983.jpg` | ISO 491, 1/14 s, f/1.87 | real motion-blurred photograph (Problem 6.4) |
| D | `IMG_20261026_195702582.jpg` | ISO 722, 1/50 s, f/1.87 | sharper spectral control for Problem 6.4 |

All are 4080 x 3072 pixels. No PSNR, SSIM or MSE is computed between A and B, between C and D, or between a photograph and its restoration.

## 5. Archive
`SHA256SUMS.txt` lists every file of the archive except itself; check it with `python verification/make_manifest.py --check`. The archive is created with `python verification/make_zip.py <output.zip>`.

## 6. Limitations
* The real photographs have no registered clean reference, so their results are qualitative or spectral/noise measurements. The blur of photograph C is not uniform; the estimated direction and length describe an effective global box PSF.
* Several comparisons use parameters tuned with the clean reference; they are labelled oracle-tuned in the report.
* Byte-identical regeneration was observed on one machine and software stack; on others, results can differ at floating-point rounding level. Timings depend on the hardware.
* The viva is assessed separately.
