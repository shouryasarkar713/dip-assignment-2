"""Report content, part G: summary of findings and closing section (limitations and reproducibility)."""
import math

from content_a import load, g, e, f


def summary(R):
    q1, q2, q3, q4, q5, q6 = (R[k] for k in ("q1", "q2", "q3", "q4", "q5", "q6"))
    B = []
    B.append(("p", "This report covers Questions 1-6. Derivations, tables and figures are in the corresponding sections, and every number is read from `results/*.json` when the report is built. The main findings are:"))
    lv = q6["q6_3_levels"]
    ox = q2["q2_3a"]["optimum_extended_sweep"]
    tab = q5["q5_1_table"]
    te = q6["q6_4_tile_estimate"]
    prim = q4["q4_3_breakdown"]["results"]["sigma_s=2,rule=A"]
    B.append(("bullets", [
        "**Q1.** The matrix DFT agrees with `numpy.fft.fft2` to %s (relative %s) with exact twiddles; conjugacy classes of a 512x512 real image = 131,074 = (N^2+4)/2 (not the 131,584 bins stored by `rfft2`, not the 262,144 real degrees of freedom). Shift and convolution theorems hold at rounding level; the circular/linear difference is confined to a (M-1)/2-pixel border and padding to $N+M-1$ is the exact threshold. The heuristic bound |r|<0.1 from the pre-implementation prediction is not supported: Parseval's relation gives the correlation of the phase-only reconstruction with the image exactly, r = E|F|/sqrt(E|F|^2) = 0.184." % (e(q1["q1_1_dft_error"]["full_512_max_abs_err"]), e(q1["q1_1_dft_error"]["table"][-1]["max_rel_err"])),
        "**Q2.** Matched Gaussian $D_0'=D_0/\\sqrt{2\\ln2}$; inverse design gives $n=2$, $D_0=20$ and minimal order $n=4$; the Si-based Gibbs prediction (8.95%%, peak at $P/2D_0$) is confirmed on an x-only strip, while the departure from it at small $D_0$ in the 2-D pipeline is attributed to the finite height of the square (control experiments exclude zero padding as the cause). Best low-pass PSNR at sigma = 20: Butterworth %.2f dB, Gaussian %.2f, ideal %.2f (optima beyond the required range, near $D_0\\approx210$)." % (ox["BLPF n=2"]["peak_PSNR_dB"], ox["GLPF"]["peak_PSNR_dB"], ox["ILPF"]["peak_PSNR_dB"]),
        "**Q3.** The log-domain homomorphic filter can flatten illumination without clipping the aperture, but histogram equalisation gives higher raw shadow contrast with better noise selectivity; the homomorphic filter wins on saturation, tonal fidelity and controllability. Large $\\gamma_H$ with small $D_0$ produces halos.",
        "**Q4.** With the matching defined by measured flat-region noise-variance reduction, classical Kuwahara and the bilateral filter (sigma_r = 2 sigma) are close at small windows and Kuwahara wins at large windows, while oracle-tuned sigma_r lets the bilateral filter catch up; soft-Kuwahara removes blockiness. The analytic weight-ratio half-point is NCR = 0.347; the empirical edge-retention-0.5 threshold of the bilateral filter ($\\sigma_s=2$) is NCR$^*$ = %s; they are different quantities." % f(prim["NCR*_retention_0.5"], 3),
        "**Q5.** Wiener denoising (no clean reference needed) gains %.2f-%.2f dB over the noisy input at sigma = 5-40 and is within 0.3 dB of oracle-tuned low-pass filters, but an oracle-tuned bilateral filter beats it at sigma <= 20: Wiener does not win everywhere. A factor-two noise-power error costs under 0.4 dB when only the denominator is mis-specified. The real photo B has a flat-patch noise of 2-3 grey levels, 3.3-4x photo A at matched levels, and is strongly spatially correlated, so the white-noise model is violated." % (min(d["wiener_known_sigma"]["PSNR"] - d["noisy"]["PSNR"] for d in tab), max(d["wiener_known_sigma"]["PSNR"] - d["noisy"]["PSNR"] for d in tab)),
        "**Q6.** The motion OTF is sinc(L s); inverse filtering fails (PSNR ~4.5 dB); the Wiener optimum is K_opt = %s, %s, %s, %s for SNR 20, 30, 40, 50 dB (K_opt exceeds the mean-power NSR by 6-126x because the signal spectrum is not flat). For the real motion-blurred photo C the single-crop spectra show **no clean zero-line pattern**; a tile-averaged, control-normalised analysis gives direction %.1f deg and length %.0f px (native resolution), and the Wiener deconvolution with this PSF visibly recovers the window bars, panel edges and clock face, with documented ringing and non-uniform-blur failures." % (g(lv["20"]["K_opt"], 2), g(lv["30"]["K_opt"], 2), g(lv["40"]["K_opt"], 2), g(lv["50"]["K_opt"], 2), te["theta_hat_deg"], te["L_hat_px"]),
    ]))
    B.append(("p", "**Limitations (Section 7):** no pixel-aligned clean references exist for the real photographs, so they are evaluated qualitatively or by spectral and noise statistics only."))
    return B


def closing(R):
    B = []
    B.append(("h1", "7. Conclusions and limitations"))
    B.append(("p", "**Conclusions.** In each problem a closed-form prediction was compared with measurement, and where the two disagree the violated assumption is identified. The 8.95% Gibbs overshoot of the ideal low-pass filter describes an edge whose neighbourhood is one-dimensional on the scale of the ripple; it does not describe small cutoffs applied to a 256-pixel square (Section 2.2). The analytic bilateral weight-ratio half-point (NCR = 0.347) and the measured edge-retention threshold are different quantities and are not expected to coincide (Section 4.3). The optimality of the Wiener filter is a statement about linear estimators under stationary second-order statistics: a locally adaptive nonlinear filter can exceed it on PSNR (Section 5.1), and the correlated, signal-dependent noise of the real photograph violates its assumptions (Section 5.2b). Several results rely on parameters tuned with the clean reference (marked oracle-tuned); they show what is attainable, not what a blind user would obtain."))
    B.append(("p", "**Limitations.**"))
    B.append(("bullets", [
        "**Real photographs.** A/B and C/D are not aligned clean/degraded pairs, so no PSNR/SSIM/MSE is computed on them. The blur of C is non-uniform: the estimated direction and length are a global effective value, the box PSF reproduces only the first lobe of the transfer function, and K for the real deblurring was chosen by inspection.",
        "**Resolutions.** Synthetic Q5/Q6 experiments use a 2x2 block-mean reduction of the instructor image (1536x1024, both sides >= 1024); Q3 uses a 3x reduction of the tunnel image; the real-photo analyses use native-resolution crops (1024x1024) and, for noise statistics, the full 4080x3072 frames. Derived images are stored separately in `data/derived/`; the four original photographs are unmodified (SHA-256 recorded).",
        "**Search grids.** Q5's oracle low-pass cutoff grid ends at 0.5 cycles/pixel (the sigma = 5 optimum sits at the edge); Q4's bilateral window is capped at 25x25 so the strength match cannot be reached for N >= 11 at sigma_r = 2 sigma (reported in Table 4.1).",
        "**Timings.** Wall-clock timings depend on the hardware (Table 0.1) and are reported for orientation only.",
    ]))
    B.append(("h1", "8. Code and reproducibility"))
    B.append(("p", "Source code, experiment scripts, data-processing utilities and reproducibility instructions are available in the project repository:"))
    B.append(("p", "`https://github.com/shouryasarkar713/dip-assignment-2`"))
    B.append(("p", "The repository is private; access must be requested from the author. It contains the implementation of each question (`src/`), the input data and photographs (`data/`), the result files (`results/`), the figures (`figures/`), the tests (`tests/`) and the sources of this report (`report/`)."))
    B.append(("p", "All random numbers come from `numpy.random.default_rng` with fixed seeds. `python run_all.py` regenerates the results and figures from the supplied inputs, and `python report/build_report.py` regenerates the PDF and DOCX report from them. On a different software stack the numbers can differ at floating-point rounding level."))
    return B
