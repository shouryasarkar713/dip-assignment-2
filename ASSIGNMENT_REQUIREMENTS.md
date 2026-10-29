# Assignment requirements matrix (DIP Assignment 2, Questions 1-6)

Source: the Digital Image Processing Assignment 2 sheet (released 8 Oct 2026, due 29 Oct 2026); the sheet itself is not redistributed in this repository.
Scope: **Questions 1-6**. Question 7 (lens distortion), listed in the marks table of the assignment sheet, is not part of this submission, as confirmed by the instructor.

Global rules extracted from the PDF header
* "From scratch" = NumPy array arithmetic; `scipy.ndimage`, `scipy.signal`, `cv2`, `skimage` may only be used to *check* answers. `numpy.fft` is allowed everywhere.
* Every number must come from the machine that ran the code; the report must state CPU, RAM and NumPy version (done in report section 0 and `results/*.json -> env`).

Legend for "Verified by": **T** = unit/property test in `tests/`, **E** = executed experiment (JSON in `results/`), **X** = external cross-check (numpy/scipy/skimage), **M** = Monte-Carlo/analytic check, **V** = visual inspection of the rendered figure.

## Question 1 - DFT and the machinery of frequency (15)
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| 1.1 | 1-D DFT matrix W_N from scratch; 2-D DFT F = W f W^T; max abs error vs `numpy.fft.fft2`; own synthetic image of >=3 sinusoids (distinct (u_i,v_i), A_i, phi_i), parameters recorded as ground truth, no external image for this subtask | `src/q1_dft.py::dft_matrix, dft2, idft2`; error table for N = 16..512 with exact vs naive twiddles; literal quadruple-sum check; `SINUSOIDS` constant is the ground truth | T, E, X |
| 1.1a | Derive F(u,v)=F*(-u,-v) for real f; state redundant region; explain why visualisers show the full symmetric spectrum | report s1.1a; counts of conjugacy classes (N^2+4)/2 = 131074 vs rfft2 bins vs real DOF; numeric Hermitian check, phase antisymmetry | T, E |
| 1.1b | Verify 2-D shift theorem and 2-D circular convolution theorem with the custom DFT on the test image; **explicit tables** measured vs theory | tables: 9 shifts, 4 kernels (5x5 Laplacian, 3x3 Sobel, 15x15 Gaussian, 63x63 random) with defined errors and denominators, plus dual theorem | T, E |
| 1.1c | Recover frequencies/amplitudes/phases from spectral peaks only; derive and explain grid resolution limit Delta f | recovery table (exact + noisy), off-grid leakage, 2-component resolution experiment, derivation Delta f = 1/N cycles/pixel (1 cycle/image) | T, E |
| 1.2 | **Dated, written analytical prediction before running** phase-vs-magnitude experiments | `predictions/q1_2_prediction.md` (timestamp 2026-10-27 11:17:12 +0530; sha256 recorded in `results/q1.json`) | E |
| 1.2a | Phase of image + flat magnitude, display; explain why flat beats magnitude-swap; compare with prediction, reconcile | figure `q1_phase_magnitude.png`; white-noise-magnitude variant; swap with a *different* natural scene; table predicted vs measured | E, V |
| 1.2b | Original magnitude with zero phase; explain via Fourier symmetry | evenness, realness, origin maximum, energy concentration checks | E |
| 1.3 | Frequency-domain filtering with small kernel (5x5 Laplacian/Sobel): unpadded circular vs zero-padded linear | `circ_conv_direct` (no FFT) vs FFT, `q1_wraparound.png` | T, E |
| 1.3a | Visual evidence of wraparound corruption; derive minimal padding (Nx,pad, Ny,pad) for N x N image and M x M kernel | P >= N+M-1 derivation; sweep P = 512..1024 showing aliasing for P < N+M-1; border-band error exactly (M-1)/2 px | T, E |
| 1.3b | Memory factor and runtime-complexity increase of padding; relate to `scipy.signal.fftconvolve` internal padding | memory (P/N)^2, cost-model ratio, measured timings, `next_fast_len` behaviour, agreement with `fftconvolve` | E, X |
| 1.D | Deliverables: error table vs fft2; shift & convolution tables; dated prediction + plots + reconciliation; recovered parameters + resolution derivation; wraparound images + padding derivations | all present in report s1 | - |

## Question 2 - Frequency-domain filtering (15)
Pipeline mandated: zero-pad N x N to P x P with P = 2N, centre spectrum, multiply by real zero-phase H defined on the P x P grid, invert, real part, crop. D = sqrt((u-P/2)^2+(v-P/2)^2).
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| data | 4 provided images + synthesize 512x512 step image (B=200 square of side 256 on A=50) | `step_image()` | E |
| 2.1 | single `make_filter(kind, P, D0, n=None, highpass=False)` for ILPF, BLPF, GLPF (exp(-D^2/(2 D0^2))), HP = 1 - LP | `src/q2_filtering.py::make_filter` | T |
| 2.1a | Evaluate each LPF analytically and numerically at D=D0; why same D0 is not like-for-like; derive matched Gaussian D0'; use pairing everywhere | H(D0) = 1, 1/2, e^{-1/2}; D0' = D0/sqrt(2 ln 2) = 0.8493 D0; used in 2.2c, 2.3 and Problem 5 | T, E |
| 2.1b | Power fraction alpha(r) for characters pattern; radii for 90/95/99/99.5 %; explain | table on the padded P x P DFT (radii in padded bins) + DC share + AC-only table | E |
| 2.1c | Inverse design: H=16/17 at D=10, H=1/17 at D=40 -> n, D0 analytically; smallest n with H(2 D0)<=1 %; verify with make_filter | n=2, D0=20; n=4 | T, E |
| 2.2 | **Dated prediction** of ILPF step profile (Si closed form, overshoot %, first-peak distance P/(2 D0)); Gaussian prediction | `predictions/q2_2_prediction.md` (2026-10-27 11:17:29 +0530) | E |
| 2.2a | ILPF D0 in {10,20,40,80}: measured vs predicted overshoot and first-peak distance; reconcile; corner profile | tables + controlled reconciliation (zero-pad vs background pad vs x-only strip) ; near-corner rows | E, T |
| 2.2b | Butterworth n in {1,2,3,5,10,20} at D0=40: overshoot, |min h|/max h of h = F^-1{H(u,0)}; explain n=1 vs higher; prove non-negative unit-sum kernel cannot overshoot | table + proof in report | E, T |
| 2.2c | Grid ILPF/BLPF(n=2)/GLPF(D0') at radii from 2.1b; arrows/insets at >= 2 ringing locations; sharpness-ringing comment | `q2_visual_grid.png`, `q2_ringing_insets.png` | V |
| 2.2d | sigma_s derivation for HGLPF on PxP grid; compare with `scipy.ndimage.gaussian_filter` for D0 in {10,40,160}; table of max error; explain nonzero error and growth | sigma_s = P/(2 pi D0); 3 error columns (truncate 4, truncate 8, periodic) | E, X |
| 2.3a | Cameraman + N(0,20^2) noise (seed reported); PSNR vs D0 (5..150) for 3 families on one plot; optimum table; explain | sweep 5..300 (spec range 5..150 reported separately; optimum lies beyond 150) | E |
| 2.3b | Gaussian LPF on text_gaps: choose D0, before/after, justify, ideal-filter counterpart | D0'=80 chosen after sweep inspection; ILPF at same half-amplitude radius shown | E, V |
| 2.3c | IHPF, BHPF(n=4), GHPF common half-amplitude radius; means; H_HP(P/2,P/2); HFE k1 + k2 H_HP; binary ridge map from BHPF; comment | table of means (cropped vs padded domain), HFE gain, binary-map statistics | E |
| 2.D | radial profiles of six transfer functions on one axis; cutoff derivation; power table; inverse design; dated prediction + tables + kernel ratio table + proof; sigma_s table; PSNR plot + optimum table; repaired text; HP + HFE | report s2 | - |

## Question 3 - Homomorphic filtering (10)
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| data | unevenly lit photograph | provided `tunnel_image.jpg` (3000x3644), reduced 3x by block average | E |
| 3.1 | I = L R model; why log makes it additive; why log **before** FFT and what goes wrong when filtering the log-magnitude spectrum of I | report s3.1; numeric identity test; spectrum figure | T, E |
| 3.2 | Implement H = (gH-gL)[1 - exp(-c D^2/D0^2)] + gL on ln I, exponentiate; sweep gL, gH, D0 as image grid; identify flattening/contrast and halo settings; compare with global histogram equalisation; local-contrast measurements in shadow region; which recovers shadow detail without blowing out bright region and why | 27-point grid (3 gL x 3 gH x 3 D0), halo metric, region metrics, HE comparison table | T, E, V |

## Question 4 - Bilateral vs Kuwahara (15)
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| 4.1 | bilateral (sigma_s, sigma_r, normalised) and classical Kuwahara (4 overlapping sub-windows, min-variance mean) from scratch | `src/q4_edge_preserving.py` | T |
| 4.1a | definition of matched effective smoothing strength + justification | flat-field noise-variance-reduction ratio S(F); sigma_s by bisection; compared with n_eff = m^2 = 4 pi sigma_s^2 | E, T |
| 4.1b | PSNR and SSIM vs clean ground truth (sigma reported); cropped zooms at boundaries | tables for N=5..15, sigma sweep, sigma_r sweep (oracle view), figures | E, V |
| 4.2a | explain why hard arg-min causes blockiness (not blur/ringing) | report; selection-map figure | V |
| 4.2b | de-blocking modification, before/after crops at identical N, explicit metric | Soft-Kuwahara (softmax variance weights); metrics: TV excess, ramp gradient irregularity, ramp RMS deviation, circle-contour deviation (noisy and noise-free input); tau sweep | T, E |
| 4.2c | per-pixel operation counts (add, mul, exp) for classic, modified, bilateral; most expensive and why | formulas + numbers for N=7,11,15 (direct and integral-image), measured timings | E |
| 4.3a-b | step-edge generator, NCR sweep, edge-preservation metric vs NCR, NCR* identified and annotated | gradient retention and edge-band PSNR, 30 NCR values x 40 noise realisations, 6 (sigma_s, rule) configurations | E |
| 4.3c | explain breakdown through w_r(dI) = exp(-dI^2/(2 sigma_r^2)) | closed-form E[w_cross]/E[w_same] with noise in the difference, Monte-Carlo verification, expected-weight profile model | M, T |

## Question 5 - Wiener denoising (15)
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| data | instructor RGB image >= 1024x1024 (same as Q6) + own noisy photo | `reference_image.jpg` (3072x2048) -> 2x2 block mean 1536x1024 working image; Photo B (and A) | E |
| 5.1 | derive W = Sff/(Sff+Snn); implement; [0,1] scaling; state Snn and Sff estimation; sigma in {5,10,20,40}; same noise realisation for all methods; no clipping before filtering; compare with Q2 Gaussian & Butterworth LPF and Q4 bilateral; PSNR/SSIM, data range, SSIM implementation, parameters; label oracle-tuned | table over sigma; Wiener (known sigma / Immerkaer estimate), LPF and bilateral oracle-tuned, blind LPF; shrinkage figure; discussion incl. cases where Wiener does not win | T, E |
| 5.2a | sigma = 20 with 0.5 Snn and 2 Snn beside correct; PSNR/SSIM; explain | two variants (Sff fixed from correct Snn; Sff refit) + flat-patch noise and edge retention | T, E |
| 5.2b | why correct Wiener still blurs edges | derivation, Wiener kernel measurements, edge gradient retention | E |
| 5.2c | white-noise Wiener on own genuinely noisy photo; state noise estimate; qualitative comparison (crops, residuals, residual spectra); discuss non-white/stationary/signal-independent departures; **no PSNR/SSIM without clean registered reference** | Photo B native-resolution crop; patch-based sigma per channel, NLF, NPS of flat patches; Photo A comparison | E |

## Question 6 - Motion blur (20)
| ID | Required item | Implementation / evidence | Verified by |
|---|---|---|---|
| 6.1 | derive PSF h(x,y); discretise + normalise; boundary convention (circular) consistent; blur clean image + noise at stated SNR (definition given, no clipping); derive OTF sinc(L(fx cos + fy sin)); zero lines at m/L; spacing 1/L; zero lines perpendicular to motion | `motion_psf`, `otf_analytic`; check |H - sinc*triangle| and zero positions | T, E |
| 6.2 | naive inverse G/H with stated floor; log-magnitude spectra of G and raw inverse; mark sampled frequencies nearest predicted zeros; explain amplification; one-sentence reason | `q6_spectra.png`, noise-gain statistics | E, V |
| 6.3 | scalar-K Wiener deconvolution; image scaling; K as constant NSR; K sweep over orders of magnitude; low-K failure and high-K under-deconvolution; K maximising PSNR at >= 3 noise levels, how optimum shifts | 4 SNR levels (20,30,40,50 dB), K from 1e-8 to 1 in 0.05 decades; optimum vs 10^(-SNR/10) | T, E |
| 6.4 | own motion-blurred photo: estimate direction and length from log-spectrum, display spectrum and mark stripe spacing, explain perpendicular stripes; build PSF, Wiener deconvolve; report theta, L, K; identify artefact + model limitation; qualitative evaluation only | Photo C (blurred) and Photo D (sharper control); single-crop cepstrum + spectral-comb estimator (validated on synthetic blur of D, reported as giving no stripe pattern on the real photo) and a tile-averaged control-normalised spectral estimator (validated on split tiles of D, adopted); automatic vs adopted values separated | T, E, V |

## Deliverable format
* Report as PDF and editable DOCX (`report/`); code (`src/`), results (`results/`), figures (`figures/`), tests (`tests/`) and a SHA-256 manifest are in the archive.
* The viva is assessed separately and is not part of this package.
