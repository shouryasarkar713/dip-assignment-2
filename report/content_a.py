"""Report content, part A: cover, section 0, Questions 1-2.  All numbers are read from results/*.json."""
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as f:
        return json.load(f)


def g(x, nd=4):
    """general-purpose numeric formatting"""
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 10 ** (-nd):
        return ("%." + str(max(nd - 2, 1)) + "e") % x
    return ("%." + str(nd) + "g") % x


def dc_unpadded_percent():
    """Fraction (percent) of the spectral power carried by the DC sample of the ORIGINAL (unpadded) 688x688 DFT of
    characters_test_pattern.tif: (sum f)^2 / (N^2 sum f^2).  Computed from the image at build time (not stored in the JSON)."""
    import numpy as np
    from PIL import Image
    with Image.open(os.path.join(ROOT, "data", "provided", "characters_test_pattern.tif")) as im:
        a = np.asarray(im.convert("L"), dtype=np.float64)
    return float(100.0 * a.sum() ** 2 / (a.size * np.sum(a ** 2)))


def e(x, d=2):
    return ("%." + str(d) + "e") % x


def f(x, d=2):
    return ("%." + str(d) + "f") % x


def cover(R):
    env = R["env"]
    return ("title", "Digital Image Processing - Assignment 2",
            "Questions 1-6: DFT, frequency-domain filtering, homomorphic filtering, edge-preserving smoothing, Wiener denoising and motion deblurring",
            ["Student: Shourya Sarkar",
             "Roll number / student ID: 2023112013",
             "Course: Digital Image Processing, Assignment 2 (released 8 October 2026, due 29 October 2026)"])


def section0(R1):
    env = R1["env"]
    B = []
    B.append(("h1", "0. Scope, environment and conventions"))
    B.append(("p", "**Scope.** This report covers Questions 1-6 of the assignment. Question 7 (lens distortion), which appears in the marks table of the assignment sheet, is not part of this submission, as confirmed by the instructor.".replace("**", "")))
    B.append(("p", "Every number in this report was produced by the code in `src/` on the machine described below. Tables and quoted values are read from `results/*.json`, and the report is generated from those files by `report/build_report.py`."))
    B.append(("table", "Machine and software used for every experiment.", ["Item", "Value"],
              [["CPU", "13th Gen Intel(R) Core(TM) i5-1340P, 12 cores / 16 logical processors"], ["RAM", "15.6 GB"],
               ["OS", env["platform"]], ["Python", env["python"]], ["NumPy", env["numpy"]], ["SciPy (checks only)", env["scipy"]], ["Matplotlib", env["matplotlib"]],
               ["Random seed", "2026 (all experiments; per-experiment offsets documented in the code)"]], "ll"))
    B.append(("p", "**Conventions.** Arrays are indexed $f[r,c]$ (row = $y$ downwards, column = $x$ to the right). DFT frequencies $(u,v)$ are the integer indices along axes 0 and 1; the forward transform is unnormalised, $F(u,v)=\\sum_{r,c} f[r,c]\\,e^{-2\\pi i (ur+vc)/N}$, and the inverse carries $1/N^2$ (NumPy's convention). Image intensities are on the 0-255 scale unless a section says that they are normalised to [0,1]. Metrics: PSNR with data range 255, SSIM implemented from scratch (Gaussian 11x11 window, $\\sigma=1.5$, $K_1=0.01$, $K_2=0.03$, 'valid' region, mean over channels), which agrees with `skimage.metrics.structural_similarity(gaussian_weights=True, use_sample_covariance=False)` to better than $10^{-13}$ (tested). Noisy inputs are never clipped before filtering; clipping is applied only for display and metric evaluation."))
    B.append(("p", "**'From scratch'.** All filters and transforms the problems ask for are written with NumPy array arithmetic. `numpy.fft` is used for the transforms as permitted; `scipy.ndimage`, `scipy.signal.fftconvolve`, `scipy.special.sici` and `skimage` are used only to check answers (explicitly marked as such in the text) and, for connected-component counting and Gaussian-equivalence checks, as measurement tools."))
    B.append(("p", "**Dated predictions.** Section 1.2 and Section 2.2 require analytical predictions written before the experiments. They are stored in `predictions/q1_2_prediction.md` and `predictions/q2_2_prediction.md`; they were written (timestamps 2026-10-27 11:17:12 and 11:17:29 +0530, recorded in the files) before the corresponding code existed. Their SHA-256 hashes and file modification times are stored in `results/q1.json` and `results/q2.json` next to the experiment run time. Each prediction is reproduced in its section and compared with the measurement."))
    B.append(("p", "**Real photographs.** Four original-resolution photographs were supplied (Table 0.2). Photographs A/B and C/D are *not* pixel-aligned clean/degraded pairs (different exposure, illumination, viewpoint and framing). Therefore no PSNR/SSIM/MSE is ever computed between A and B or between C and D, and no synthetic degradation is presented as a camera degradation. Quantitative restoration scores use synthetic degradations of the instructor-provided reference image only; the photographs are used for qualitative demonstrations and for noise/spectral measurements."))
    return B


PHOTO_TABLE = None


def photo_table():
    rows = [["A", "IMG_20261026_195540763.jpg", "4080x3072", "ISO 734, 1/14 s (0.07 s), f/1.87, 5.56 mm", "bright turquoise room, open window", "Q5 qualitative comparison (less noisy reference scene)"],
            ["B", "IMG_20261026_195556198.jpg", "4080x3072", "ISO 6400, 1/16.7 s (0.0600 s), f/1.87", "dark room, golden window lights", "Q5.2 real low-light noisy photo"],
            ["C", "IMG_20261026_195635983.jpg", "4080x3072", "ISO 491, 1/14 s (0.07 s), f/1.87", "blurred room: white door, barred window, wall clock", "Q6.4 real blurred photo"],
            ["D", "IMG_20261026_195702582.jpg", "4080x3072", "ISO 722, 1/50 s (0.02 s), f/1.87", "sharper view of the same room", "Q6.4 spectral control (no ground truth)"]]
    return ("table", "Table 0.2. The four supplied photographs (all Nothing/CMF Phone 2 Pro, 'MediaTek Camera Application', EXIF intact, files unmodified; SHA-256 recorded in data/photos_original_sha256.json).",
            ["ID", "File", "Size", "EXIF exposure", "Content", "Role"], rows, "lllllp{3.6cm}")


# =============================================================================== Question 1
def section1(R):
    B = []
    e11 = R["q1_1_dft_error"]
    B.append(("h1", "Question 1 - DFT and the machinery of frequency"))
    B.append(("h2", "1.1 Building and validating the DFT"))
    B.append(("p", "**Definition.** The 1-D DFT matrix is $W_N[k,n]=e^{-2\\pi i kn/N}$. Because the 2-D kernel factorises, $e^{-2\\pi i(ur+vc)/N}=e^{-2\\pi i ur/N}e^{-2\\pi i vc/N}$, the 2-D DFT is a pair of matrix products (rows then columns):".replace("**", "")))
    B.append(("eq", r"F = W_N\, f\, W_N^{T}, \qquad f = \frac{1}{N^{2}}\,\overline{W_N}\, F\, \overline{W_N}^{T}"))
    B.append(("p", "The twiddle exponent is computed as $(kn \\bmod N)$ before evaluating the complex exponential, which keeps every argument in $[0,2\\pi)$; the naive evaluation of $\\cos,\\sin$ at angles up to $2\\pi N$ loses accuracy (last column of Table 1.1). The same matrix code was checked against a literal quadruple sum (independent of $W_N$ and of NumPy): maximum difference %s on a 16x16 random image." % e(e11["direct_quadruple_sum_vs_custom_16x16"])))
    rows = []
    for r in e11["table"]:
        rows.append([r["N"], e(r["max_abs_err_exact_twiddles"]), e(r["max_abs_err_naive_twiddles"]), e(r["max_abs_F"]), e(r["max_rel_err"]), e(r["rel_L2_err"]), e(r["roundtrip_max_abs_err"])])
    B.append(("table", "Table 1.1. Custom DFT vs numpy.fft.fft2 on the top-left NxN crop of the provided test image: maximum absolute error max|F_custom - F_numpy|, the same divided by max|F_numpy| (relative), relative L2 error ||F_c - F_n||_2/||F_n||_2, and inverse round-trip error max|idft2(dft2(f)) - f|.",
              ["N", "max abs err (exact twiddles)", "max abs err (naive twiddles)", "max|F|", "max abs err / max|F|", "rel. L2 err", "round-trip err"], rows, "rrrrrrr"))
    B.append(("p", "At N=512 the absolute error is %s, which is $%s\\,$ of $\\max|F|=%s$: machine precision, as expected for a double-precision matrix product whose entries have magnitude up to $3.4\\times 10^{7}$ (the DC term). Timing (single run, N=512): custom matrix DFT %.3f s, numpy FFT %.3f s; the matrix method costs $O(N^3)$ against $O(N^2\\log N)$ and is used for validation only." % (e(e11["full_512_max_abs_err"]), e(e11["table"][-1]["max_rel_err"]), e(e11["table"][-1]["max_abs_F"]), e11["timing"]["custom_seconds"], e11["timing"]["numpy_seconds"])))

    # ---- 1.1a
    cs = R["q1_1a_conj_symmetry"]
    c512 = cs["counts_512"]
    B.append(("h2", "1.1a Conjugate symmetry"))
    B.append(("p", "**Derivation.** For a real image, $f=\\bar f$. Evaluating the transform at $(-u,-v)$ and conjugating:".replace("**", "")))
    B.append(("eq", r"F(-u,-v)=\sum_{r,c} f[r,c]\,e^{+2\pi i(ur+vc)/N}=\overline{\sum_{r,c}\overline{f[r,c]}\,e^{-2\pi i(ur+vc)/N}}=\overline{F(u,v)}"))
    B.append(("p", "(indices are taken modulo $N$). Hence $|F(u,v)|=|F(-u,-v)|$ (even magnitude) and $\\angle F(-u,-v)=-\\angle F(u,v)$ (odd phase). Numerically, on the test image the maximum of $|F(u,v)-\\overline{F(-u,-v)}|$ is %s (%s of $\\max|F|$), the phase antisymmetry error over bins with $|F|>10^{-3}\\max|F|$ is %s rad, whereas the same test applied to a complex-valued input fails with a relative error of %s (the property is specific to real images)." % (e(cs["max_abs_F_minus_conjF(-u,-v)_real_image"]), e(cs["relative_to_max|F|"]), e(cs["phase_antisymmetry_max_err_over_|F|>1e-3max"]), f(cs["same_test_with_complex_input_(must_FAIL)"], 3))))
    B.append(("p", "**Redundant region.** Half of the plane is redundant: knowing $F$ on any half-plane (e.g. $v>0$, plus $v=0,\\,u\\ge 0$, plus the Nyquist column $v=N/2$ handled in the same way) determines the rest by $F(-u,-v)=\\overline{F(u,v)}$. **Why viewers show the full spectrum:** the spectrum is an $N\\times N$ array that must be stored and indexed in full for filter design (a transfer function must itself be symmetric, $H(-u,-v)=\\overline{H(u,v)}$, for the output to be real) and the centred ($\\mathrm{fftshift}$) display simply plots the whole array; the symmetry is a consequence of $f$ being real, not a property of the display. Figure 1.1 shows the full log-magnitude, the non-redundant half and the odd phase.".replace("**", "")))
    B.append(("fig", "figures/q1_spectrum.png", "Test image, centred log(1+|F|), the non-redundant half plane (v >= 0) and the (odd-symmetric) phase.", 1.0))
    B.append(("p", "**Counting conventions (even N=512).** Under $(u,v)\\mapsto(-u,-v) \\bmod N$ the fixed points are exactly the four bins with $u,v\\in\\{0,N/2\\}$. The number of conjugacy-equivalence classes (orbits) is therefore $4+(N^2-4)/2=(N^2+4)/2$, which is %d for N=512 (enumerated numerically: %d). This is different from (i) the number of bins stored by `rfft2`, $N(N/2+1)=%d$, which still contains %d redundant entries (the conjugate-symmetric pairs inside the two columns $v=0$ and $v=N/2$), and (ii) the number of independent real degrees of freedom: each of the 4 self-conjugate bins is real (1 DOF) and each of the other $(N^2-4)/2$ classes is complex (2 DOF), giving $4+(N^2-4)=N^2=%d$, equal to the number of pixels, as it must for an invertible transform. For odd N only DC is self-conjugate ($(N^2+1)/2=%d$ classes)." % (c512["conjugacy_classes"], c512["conjugacy_classes"], c512["rfft2_stored_bins"], c512["rfft2_redundant_bins"], c512["real_dof"], cs["counts_odd_N=511_(no_Nyquist_bins)"]["conjugacy_classes"])))
    B.append(("table", "Table 1.2. Counting of spectral quantities for N=512 (enumerated by code, tests/test_q1.py).", ["Quantity", "Value"],
              [["self-conjugate bins (fixed points)", c512["fixed_points"]], ["conjugacy classes = (N^2+4)/2", c512["conjugacy_classes"]], ["real degrees of freedom (= N^2)", c512["real_dof"]],
               ["bins stored by rfft2 = N(N/2+1)", c512["rfft2_stored_bins"]], ["redundant entries inside rfft2 storage", c512["rfft2_redundant_bins"]], ["real numbers stored by rfft2", c512["rfft2_stored_real_numbers"]]], "lr"))

    # ---- 1.1b shift
    B.append(("h2", "1.1b Verification of the shift and circular-convolution theorems"))
    B.append(("p", "**Shift theorem.** If $g[r,c]=f[(r-a)\\bmod N,(c-b)\\bmod N]$ then, substituting $r'=r-a$, $c'=c-b$,"))
    B.append(("eq", r"G(u,v)=e^{-2\pi i (ua+vb)/N}\,F(u,v)"))
    B.append(("p", "so the magnitude is unchanged and the phase changes linearly by $-2\\pi(ua+vb)/N$. Table 1.3 compares $G$ computed by the custom DFT of the shifted test image with the prediction (right-hand side); phase errors are evaluated on the 5717 bins with $|F|>10^{-3}\\max|F|$ (below that the phase of a numerically tiny bin is meaningless)."))
    rows = []
    for r in R["q1_1b_shift_theorem"]:
        rows.append(["(%d, %d)" % tuple(r["shift_(r,c)"]), e(r["max_abs_|DFT(shifted)-predicted|"]), e(r["rel_to_max|F|"]), e(r["max_||Fs|-|F||"]), e(r["max_phase_error_rad_(|F|>1e-3max)"]), e(r["spatial_check_max|f_shift-roll|"])])
    B.append(("table", "Table 1.3. Shift theorem: measured vs theoretical prediction. Columns: max|DFT(shifted) - F e^{-2 pi i (ua+vb)/N}|; the same relative to max|F| = 3.38e7; max over bins of ||F_s|-|F|| (theory: 0); max phase deviation from the predicted linear phase (rad); max spatial error after inverting the predicted spectrum vs np.roll (theory: 0).",
              ["shift (a,b)", "max abs err", "rel. to max|F|", "max ||Fs|-|F||", "max phase err (rad)", "spatial err"], rows, "lrrrrr"))
    B.append(("p", "**Circular convolution theorem.** With $(f\\circledast g)[r,c]=\\sum_{a,b} f[a,b]\\,g[(r-a) \\bmod N,(c-b) \\bmod N]$,"))
    B.append(("eq", r"\mathrm{DFT}\{f\circledast g\}(u,v)=F(u,v)\,G(u,v)"))
    B.append(("p", "The *reference* convolution is computed directly in the spatial domain by summing rolled copies of the image, $\\sum_{m,n}K[m,n]\\,f[(r-m+c_m) \\bmod N,(c-n+c_n) \\bmod N]$ (no FFT involved); the *tested* value is $\\mathrm{IDFT}\\{F\\,G\\}$ with the custom DFT. Errors are defined as: absolute $\\max|a-b|$; relative-L2 $\\|a-b\\|_2/\\|a\\|_2$; and max error divided by $\\max|a|$ ($a$ = direct spatial result)."))
    rows = [[r["kernel"], r["taps"], e(r["max|spatial|"]), e(r["max_abs_err"]), e(r["max_abs_err_over_max|spatial|"]), e(r["rel_L2_err_(||a-b||2/||a||2)"])] for r in R["q1_1b_convolution_theorem"]]
    B.append(("table", "Table 1.4. Circular convolution theorem on the 512x512 test image (kernel origin at its centre, wrapped).", ["kernel", "taps", "max|direct|", "max abs err", "max abs err / max|direct|", "rel. L2 err"], rows, "lrrrrr"))
    d = R["q1_1b_dual_theorem_32x32"]
    B.append(("p", "The dual statement $\\mathrm{DFT}\\{f\\cdot g\\}=\\frac{1}{N^2}F\\circledast G$ was verified on a 32x32 crop windowed by a Hann function: max abs error %s (%s relative). All discrepancies are at rounding level; the identities hold exactly in exact arithmetic." % (e(d["max_abs_err"]), e(d["rel_to_max|lhs|"]))))

    # ---- 1.1c
    sc = R["q1_1c_sinusoids"]
    B.append(("h2", "1.1c Sinusoid parameter recovery and the resolution limit"))
    B.append(("p", "**Synthetic image (ground truth).** $f[r,c]=\\sum_{i=1}^{4} A_i\\cos\\!\\big(2\\pi(u_i r+v_i c)/N+\\phi_i\\big)$ with $N=512$ and the parameters listed as 'true' in Table 1.5. A real cosine with parameters $(u,v,A,\\phi)$ produces two spectral lines, $F(\\pm u,\\pm v)=\\frac{AN^2}{2}e^{\\pm i\\phi}$; the recovery below therefore uses only the half plane $u>0$ (or $u=0,v>0$): frequency = peak position, $A=2|F|/N^2$, $\\phi=\\angle F$. The peak picker (local maxima of $|F|$ above $10^{-3}\\max|F|$) uses no ground-truth information."))
    rows = []
    for r in sc["comparison"]:
        t, rc = r["true"], r["recovered"]
        rows.append(["(%d, %d)" % (t["u"], t["v"]), f(t["A"], 3), f(t["phi"], 4), "(%d, %d)" % (rc["u"], rc["v"]), f(rc["A"], 6), f(rc["phi"], 6), e(r["err_A"]), e(r["err_phi"])])
    B.append(("table", "Table 1.5. True vs recovered parameters (noise-free, half-plane convention: signs of (u,v) and phi flipped where the true frequency lies in the other half plane).",
              ["true (u,v)", "true A", "true phi", "rec. (u,v)", "rec. A", "rec. phi", "A error", "phi error (rad)"], rows, "lrrlrrrr"))
    rn = sc["recovered_noisy"]
    B.append(("p", "With additive Gaussian noise of standard deviation %.1f (not part of the ground truth) the same procedure returns amplitudes %s and phases %s, i.e. errors below 0.02 in amplitude and 0.001 rad in phase, because the DFT averages $N^2=262144$ noisy samples per bin (error std $\\approx\\sigma\\sqrt{2/N^2}$)." % (sc["noise_sigma"], ", ".join(f(p["A"], 3) for p in rn), ", ".join(f(p["phi"], 4) for p in rn))))
    B.append(("p", "**Resolution limit.** For a sinusoid of frequency $f_0$ (cycles/pixel) observed on $N$ samples, the DFT bin spacing is $\\Delta f=1/N$ cycles/pixel (1 cycle per image); the continuous spectrum of the rectangular observation window is the Dirichlet kernel"))
    B.append(("eq", r"W_N(f-f_0)=\sum_{n=0}^{N-1} e^{-2\pi i (f-f_0)n}=e^{-i\pi (N-1)(f-f_0)}\,\frac{\sin(\pi N (f-f_0))}{\sin(\pi (f-f_0))}"))
    B.append(("p", "whose main lobe has null-to-null width $2/N$. The origin of the limit is the finite window: a signal observed over $T=N$ samples cannot resolve frequencies closer than $\\sim 1/T$ (time-frequency uncertainty). Bins at integer multiples of $1/N$ see each other's kernels at nulls (orthogonality), so on-grid sinusoids are recovered exactly (Table 1.5); off-grid sinusoids leak. Test: a component at $u=40.5$ (half a bin) gives a grid peak of $A_\\mathrm{peak}=%s$ instead of 80, i.e. a ratio %s, which equals the predicted scalloping loss $|\\mathrm{sinc}(0.5)|=2/\\pi=0.6366$ for one axis half a bin off-grid; zero padding by 4 recovers $(u,v)=(%s,%s)$ and $A=%s$." % (f(R["q1_1c_off_grid"]["grid_peaks_(top3)"][0]["A"], 3), f(R["q1_1c_off_grid"]["grid_peaks_(top3)"][0]["A"] / 80, 4), g(R["q1_1c_off_grid"]["padded_peak"]["u"]), g(R["q1_1c_off_grid"]["padded_peak"]["v"]), f(R["q1_1c_off_grid"]["padded_peak"]["A_from_peak"], 3))))
    rows = []
    for r in R["q1_1c_resolution"]:
        rows.append([r["separation_bins(=cycles/image)"], r["phase"], r["main_peaks_in_padded_spectrum"], ", ".join("%.2f" % x for x in r["peak_locations"]), "n/a" if r["valley/lower_peak"] is None else g(r["valley/lower_peak"], 3), "yes" if r["resolved_(2_peaks_and_valley<0.9)"] else "no"])
    B.append(("table", "Table 1.6. Two equal-amplitude sinusoids at u = 50 and u = 50 + delta (v = 0) on N = 512 samples; the spectrum is evaluated with 16x zero padding and the number of main lobes (>= 50% of the maximum) and the valley depth between them are reported. 'Resolved' means two main lobes with valley < 0.9 of the lower peak.",
              ["delta (bins)", "relative phase", "main peaks", "peak locations", "valley/lower peak", "resolved"], rows, "lrrlrr"))
    B.append(("p", "Reading the table: separations below one bin merge into one lobe or give biased peak positions; between about 0.75 and 1.5 bins the outcome depends on the relative phase of the two components (interference of the two Dirichlet kernels shifts and distorts the lobes), and from 2 bins (the null-to-null width) the components are always separated with a valley at the numerical zero. Hence the *guaranteed*, phase-independent resolution is $2\\Delta f=2/N$ and the *nominal* grid resolution is $\\Delta f=1/N$ cycles/pixel (here 0.00195 cycles/pixel = 1 cycle/image)."))
    B.append(("fig", "figures/q1_resolution.png", "Resolution experiment: zero-padded spectrum (blue) and the N-point DFT bins (red) for separations of 0.5, 1 and 2 bins.", 1.0))
    B.append(("fig", "figures/q1_sinusoid.png", "Synthetic four-sinusoid image and the zoomed spectrum with the true (u,v) and (-u,-v) positions overlaid.", 0.85))

    # ---- 1.2
    q = R["q1_2"]
    pr = R["q1_2_prediction_file"]
    B.append(("h2", "1.2 Phase versus magnitude"))
    B.append(("p", "**Pre-implementation prediction** (full text in `predictions/q1_2_prediction.md`, written %s, sha256 %s...; the experiment was run at %s)." % ("2026-10-27 11:17:12 +0530", pr["sha256"][:16], pr["experiment_run_time"])))
    B.append(("bullets", [
        "Flat magnitude $c$: $\\hat f=c\\,\\mathrm{IDFT}[F/|F|]=c\\,(f\\circledast w)$ with the zero-phase whitening kernel $w=\\mathrm{IDFT}[1/|F|]$. By the shift theorem an impulse keeps its position (its phase carries the position); phase is not additive, so superposition fails and the result is the image filtered by $1/|F|\\sim|k|^{\\alpha}$, a strong high-pass: contours and texture at their original positions, low-frequency tone lost, sparse heavy-tailed edge-map-like appearance.",
        "Exact consequences: $\\hat f$ real; $\\sum\\hat f^2=c^2$ (Parseval); mean $=c/N^2$; flat radial power spectrum.",
        "Heuristics (not exact): smoothed $|\\hat f|$ correlates with the gradient magnitude (predicted $r>0.3$); correlation of $\\hat f$ with $f$ small (predicted $|r|<0.1$).",
        "White-noise magnitude $R$ (Rayleigh-like, independent of the phase): $\\mathrm{corr}(\\hat f_R,\\hat f_{flat})=E[R]/\\sqrt{E[R^2]}=\\sqrt\\pi/2=0.886$.",
        "Zero phase: $\\hat f=\\mathrm{IDFT}|F|$ is real and even, maximum at the origin equal to $\\sum|F|/N^2$, no localisation: a smooth centred blob."]))
    B.append(("fig", "figures/q1_phase_magnitude.png", "Phase-only reconstructions and the zero-phase reconstruction. Top: original; phase + flat magnitude; phase + white-noise magnitude. Bottom: phase of the test image + magnitude of a different photograph; magnitude + zero phase (fftshifted, square-root display); the log-magnitude spectrum used in (b).", 1.0))
    ex = q["corr(fhat_flat,f)_exact_Parseval_prediction_E|F|/sqrt(E|F|^2)_(non-DC_bins)"]
    rows = [
        ["f_hat (flat) is real: max|Im|/max|Re|", "0 (exact)", e(q["real_valued_max|Im|/max|Re|"]), "confirmed (rounding)"],
        ["Parseval: sum f_hat^2 / c^2", "1", g(q["parseval_sum_f2_over_c2_(pred_1)"], 8), "confirmed"],
        ["mean(f_hat) N^2 / c", "1", g(q["mean_fhat_times_N2_over_c_(pred_1)"], 8), "confirmed"],
        ["flatness of radial power std/mean", "0", e(q["radial_power_flatness_std/mean_(r<256)"]), "confirmed"],
        ["corr(f_hat, f)", "|r| < 0.1 (heuristic)", f(q["corr(fhat_flat,f)_(pred_|r|<0.1)"], 4), "NOT confirmed; see text"],
        ["corr(smoothed |f_hat|, smoothed |grad f|)", "> 0.3", f(q["corr(smoothed|fhat_flat|,smoothed_|grad_f|)_(pred_>0.3)"], 3), "confirmed"],
        ["kurtosis of f_hat (Gaussian = 3; f: %.2f)" % q["kurtosis_f"], "heavy-tailed (> 3)", f(q["kurtosis_fhat_flat"], 2), "confirmed"],
        ["corr(f_hat_noise-magnitude, f_hat_flat)", "0.886 +- 0.01", f(q["corr(fhat_noise,fhat_flat)_(pred_0.886)"], 4), "confirmed"],
        ["zero phase: f_hat real / even (rel. err)", "0 / 0", "%s / %s" % (e(q["zerophase_max|Im|/max|Re|"]), e(q["zerophase_evenness_max|f(x)-f(-x)|/max"])), "confirmed"],
        ["zero phase: argmax and max value", "(0,0), sum|F|/N^2 = %s" % f(q["zerophase_max_value_predicted_sum|F|/N^2"], 2), "%s, %s" % (tuple(q["zerophase_argmax_(r,c)"]), f(q["zerophase_max_value"], 2)), "confirmed"]]
    B.append(("table", "Table 1.7. Pre-implementation predictions vs measurements (Section 1.2).", ["quantity", "prediction", "measured", "status"], rows, "p{5.2cm}p{3.5cm}p{3.2cm}p{3.5cm}"))
    B.append(("p", "**Reconciliation.** All exact predictions are confirmed to rounding precision, and the qualitative description (outlines and texture at the right places, no tone, heavy tails; Figure 1.4, top middle) is correct. One heuristic bound was wrong: the correlation of $\\hat f$ with $f$ was predicted to be below 0.1 in magnitude but measured %s. The error was in the heuristic, not in the theory: by Parseval the correlation between $f-\\bar f$ and $\\hat f=\\mathrm{IDFT}[F/|F|]$ is exactly $\\sum_{k\\ne0}|F_k|/\\sqrt{K\\sum_{k\\ne 0}|F_k|^2}=E|F|/\\sqrt{E|F|^2}$ over the non-DC bins, because $F\\,\\overline{F/|F|}=|F|$ is positive in every bin. For this image that ratio is %s (the measured value is %s): positive and small, because the spread of $|F|$ over six decades makes $E|F|\\ll\\sqrt{E|F|^2}$. The white-noise-magnitude variant agrees with $\\sqrt\\pi/2$ to four digits (the empirical $E[R]/\\sqrt{E[R^2]}$ of the particular noise draw is %s)." % (f(q["corr(fhat_flat,f)_(pred_|r|<0.1)"], 3), f(ex, 4), f(q["corr(fhat_flat,f)_(pred_|r|<0.1)"], 4), f(q["empirical_E[R]/sqrt(E[R^2])_of_noise_magnitudes"], 4))))
    B.append(("p", "**Why a flat magnitude isolates structure more cleanly than a magnitude swap.** With a flat magnitude, $\\hat f=c(f\\circledast w)$ contains *only* information of $f$ (its measured correlation with the second photograph is %s, i.e. none). With another natural image's magnitude, $\\hat f=f\\circledast\\mathrm{IDFT}[|G|/|F|]$: since $|G|$ has the typical $1/k^\\alpha$ envelope, the result keeps most of the low-frequency energy and mixes in the donor's amplitude structure: its correlation with $f$ is %s (tone and layout of $f$ visible, Figure 1.4 bottom left) but it is also correlated with the donor image (%s) and its gradient-magnitude correlation drops to %s. The swap therefore reflects the *product* of two sets of statistics, while the flat magnitude removes every amplitude cue so that what remains is attributable to phase (position) alone." % (f(q["corr(fhat_flat,donor_image)"], 3), f(q["corr(fhat_swap_magdonor,f)"], 3), f(q["corr(fhat_swap_magdonor,donor_image)"], 3), f(q["corr(smoothed|swap|,smoothed_grad_f)"], 3))))
    B.append(("p", "**(b) Zero phase.** With $\\angle F\\equiv 0$, $\\hat f=\\mathrm{IDFT}|F|$. Because $|F|$ is real, non-negative and even, $\\hat f$ is real and even ($\\hat f(x,y)=\\hat f(-x,-y)$; measured asymmetry %s) and all spectral components add constructively at the origin, so the global maximum sits at $(0,0)$ and equals $\\sum|F|/N^2=%s$ (measured %s). No linear phase means no position information: structure is not localised; the image is a centrally symmetric blob whose radial profile is the square-root of the power spectrum (a smooth, sharply peaked core with streaks from the dominant edge orientations). %.1f%% of its energy lies within 16 px of the origin (the test image: %.2f%%)." % (e(q["zerophase_evenness_max|f(x)-f(-x)|/max"]), f(q["zerophase_max_value_predicted_sum|F|/N^2"], 2), f(q["zerophase_max_value"], 2), 100 * q["zerophase_fraction_of_energy_within_radius16_of_origin"], 100 * q["zerophase_fraction_of_|f|_energy_in_same_disc_for_test_image"])))

    # ---- 1.3
    B.append(("h2", "1.3 Padding, wraparound and computational overhead"))
    B.append(("p", "A $5\\times5$ Laplacian-of-Gaussian-type kernel $K$ and the $3\\times3$ Sobel kernel are applied to the 512x512 test image by multiplying DFTs. The *linear* reference is computed by direct spatial summation with zeros outside the image (no FFT). Figure 1.5 compares the unpadded circular result with the zero-padded linear result."))
    B.append(("fig", "figures/q1_wraparound.png", "Wraparound evidence. Top row: 5x5 Laplacian, circular vs linear and the error confined to the first 2 pixels from the border (log plot: exactly zero beyond). Bottom row: a 31x31 Gaussian low-pass makes the border difference plainly visible: the circular result mixes in content from the opposite edge of the image (bright sky at the top with the ground at the bottom, dark coat at the left with the right edge), whereas the zero-padded linear result fades towards black at the border; the two differ by up to 49.9% of full scale within the 15-pixel band (right plot: column 256, first 60 rows).", 1.0))
    wa = R["q1_3a_wraparound"]
    rows = []
    for k, v in wa.items():
        rows.append([k, v["border_width_(M-1)/2"], e(v["max_err_interior"]), g(v["max_err_border_band"], 5), g(v.get("fraction_of_pixels_corrupted", v.get("fraction_of_pixels_corrupted_(>1e-9)")), 4), g(v.get("expected_fraction_1-(N-2h)^2/N^2", v.get("expected_fraction")), 4)])
    B.append(("table", "Table 1.8. Where circular and linear convolution differ: error is zero (to rounding) in the interior and non-zero only in the border band of width (M-1)/2. (Sobel: some border pixels are zero by coincidence on flat areas, so the corrupted fraction is slightly below the band area.)", ["kernel", "band width (M-1)/2", "max err interior", "max err in band", "fraction of pixels corrupted", "band area fraction 1-((N-2h)/N)^2"], rows, "lrrrrr"))
    B.append(("p", "**Minimal padding.** The linear convolution of an $N\\times N$ image with an $M\\times M$ kernel has support $(N+M-1)\\times(N+M-1)$. The DFT product corresponds to circular convolution of period $P$; a sample at linear index $n\\ge P$ lands on $n-P$ (time-domain aliasing). Exact equivalence with the linear convolution for all outputs therefore requires"))
    B.append(("eq", r"N_{x,\mathrm{pad}}=N_{y,\mathrm{pad}}\ \ge\ N+M-1 ,\qquad \text{minimal: } P=N+M-1=516\ \ (N=512,\,M=5)"))
    B.append(("p", "Both image and kernel are zero-padded (kernel origin at index 0) to $P\\times P$. The sweep of Table 1.9 confirms it: for $P<N+M-1$ the output differs from the true linear result by up to 1819 (the aliased rows), while the aliasing-folded prediction of the full linear result matches the FFT output to $10^{-11}$; at $P=516$ and above the error is $10^{-12}$. (If only the centred 'same' crop is required, aliased samples fall outside the crop when $P\\ge N+(M-1)/2=514$; this is visible in the last column.)"))
    rows = []
    for r in R["q1_3_padding_table"]:
        if not r["kernel"].startswith("Lap"):
            continue
        e1 = r.get("max_abs_err_vs_full_linear")
        e2 = r.get("max_abs_err_vs_full_linear_on_P_x_P")
        e3 = r.get("max_abs_err_vs_folded_full_(aliasing_prediction)")
        rows.append([r["P"], "yes" if r["P>=N+M-1"] else "no", f(r["memory_factor_(P/N)^2"], 4), e(e1 if e1 is not None else e2), "-" if e3 is None else e(e3), e(r["max_abs_err_cropped_same_vs_linear"])])
    B.append(("table", "Table 1.9. Zero-padding size sweep for the 5x5 Laplacian (N=512, M=5): max error of the FFT result vs the direct linear convolution (column 4; for P < N+M-1 over the P x P block), vs the folded (aliased) linear result (column 5; theory says 0), and for the centred 'same' crop.", ["P", "P >= N+M-1", "memory factor (P/N)^2", "max err vs linear", "max err vs folded linear", "max err of 'same' crop"], rows, "rrrrrr"))
    B.append(("fig", "figures/q1_padding_sweep.png", "Maximum error of the padded-FFT convolution versus P: it collapses to rounding level exactly at P = N+M-1 = 516.", 0.5))
    cs_ = R["q1_3b_cost"]
    B.append(("p", "**Cost of padding.** Storage grows by $(P/N)^2$ (complex arrays): 1.016 for $P=516$ and 4 for the $P=2N$ pipeline of Problem 2. A 2-D FFT of $P\\times P$ points costs $\\sim 5P^2\\log_2(P^2)$ flops, so relative to the unpadded $N\\times N$ transform the asymptotic increase is"))
    B.append(("eq", r"\frac{P^2\log P^2}{N^2\log N^2}=\Big(\frac{P}{N}\Big)^{2}\Big(1+\frac{\log (P/N)}{\log N}\Big),\qquad P=2N:\ 4\Big(1+\frac{1}{9}\Big)=4.44"))
    B.append(("p", "i.e. still $O(N^2\\log N)$ with a constant that depends on $(P/N)^2$; three transforms are needed (image, kernel, inverse). Direct spatial convolution costs $O(N^2M^2)$, so the FFT route wins beyond a kernel size of roughly $M\\gtrsim 10$ on this machine (Table 1.10). `scipy.signal.fftconvolve` implements exactly this scheme: it computes the full linear shape $N+M-1$ and then pads further to `scipy.fft.next_fast_len(N+M-1, real=True)` because FFT length matters: $514=2\\cdot257$, $526=2\\cdot263$ and $516=2^2\\cdot3\\cdot43$ contain large prime factors and are *slower* than the 5-smooth length 540 (compare the 'minimal' and 'fast length' timing columns of Table 1.10). Our padded FFT with the same length reproduces `fftconvolve` to $\\le 4\\times10^{-10}$ (last column)."))
    rows = []
    for r in cs_["rows"]:
        rows.append([r["M"], r["N+M-1"], r["scipy_style_next_fast_len"], f(r["memory_factor_(P/N)^2_minimal"], 3), f(r["memory_factor_(P/N)^2_fast_len"], 3), f(r["ratio_cost_model_P^2logP^2/N^2logN^2"], 3), f(1e3 * r["t_unpadded_circular_s"], 1), f(1e3 * r["t_padded_minimal_s"], 1), f(1e3 * r["t_padded_fast_len_s"], 1), f(1e3 * r["t_scipy_fftconvolve_full_s"], 1), "-" if r["t_direct_spatial_s"] is None else f(1e3 * r["t_direct_spatial_s"], 0), e(r["max_abs_diff_vs_scipy_fftconvolve"])])
    B.append(("table", "Table 1.10. Padding overhead for N=512 and M x M kernels. Memory factors (P/N)^2 for the minimal and the fast FFT length, cost-model ratio, and measured median times in ms (3 runs; machine dependent - the ratios, not the absolute times, are the result): unpadded circular, padded to N+M-1, padded to next fast length, scipy.signal.fftconvolve(full), direct spatial (rolled sums), and max difference of our fast-length result vs fftconvolve.",
              ["M", "N+M-1", "fast len", "mem (min)", "mem (fast)", "cost ratio", "t circ", "t pad min", "t pad fast", "t scipy", "t direct", "max diff vs scipy"], rows, "rrrrrrrrrrrr"))
    return B


# =============================================================================== Question 2
def section2(R):
    B = []
    B.append(("h1", "Question 2 - Frequency-domain filtering: ideal, Butterworth, Gaussian"))
    B.append(("p", "**Pipeline (used throughout).** Each $M\\times N$ image is placed in the top-left corner of a $P\\times P$ zero array with $P=2\\max(M,N)$, transformed, multiplied by a real zero-phase $H(u,v)$ defined on the centred $P\\times P$ grid with $D=\\sqrt{(u-P/2)^2+(v-P/2)^2}$, inverted, the real part taken and the result cropped. Radii $D_0$ are therefore in bins of the padded grid (cycles per $P$ pixels); e.g. for the 512x512 images $P=1024$, so $D_0=40$ corresponds to $0.039$ cycles/pixel. `make_filter(kind, P, D0, n=None, highpass=False)` (src/q2_filtering.py) builds"))
    B.append(("eq", r"H_{ILPF}=\begin{cases}1&D\le D_0\\0&D>D_0\end{cases}\qquad H_{BLPF}=\frac{1}{1+[D/D_0]^{2n}}\qquad H_{GLPF}=e^{-D^2/(2D_0^2)}\qquad H_{HP}=1-H_{LP}".replace(r"\begin{cases}1&D\le D_0\\0&D>D_0\end{cases}", r"\{1\ (D\le D_0),\ 0\ (D>D_0)\}")))
    a = R["q2_1a"]
    B.append(("h2", "2.1a Cutoff semantics and the matched Gaussian"))
    rows = [[k, g(v["measured_H(D=D0)"], 6), g(v["analytic"], 6)] for k, v in a["H_at_D=D0"].items() if isinstance(v, dict) and "analytic" in v]
    B.append(("table", "Table 2.1. Low-pass value at D = D0 = 40 (P = 1024): analytic vs measured with make_filter. 'GLPF matched' uses D0' = D0/sqrt(2 ln 2) = 33.97 (the value at (u,v) offset (24,32), i.e. exactly D = 40 on a diagonal, is also exactly 0.5).", ["filter", "measured H(D0)", "analytic"], rows, "lrr"))
    B.append(("p", "At $D=D_0$ the ILPF equals 1 (it jumps from 1 to 0 just beyond), the Butterworth filter equals $1/2$ for every order, and the Gaussian equals $e^{-1/2}=0.607$. The same numerical $D_0$ is therefore not like-for-like: the three filters have different attenuation at $D_0$ (ILPF has no half-amplitude point at all; the Gaussian is already above the Butterworth half-power value). Requiring $H_{GLPF}(D_0)=1/2$:"))
    B.append(("eq", r"e^{-D_0^2/(2D_0'^2)}=\frac12\ \Rightarrow\ D_0'=\frac{D_0}{\sqrt{2\ln 2}}=0.8493\,D_0"))
    B.append(("p", "**This matched pairing is used in every later comparison**: ILPF and BLPF with $D_0$, GLPF with $D_0'=D_0/\\sqrt{2\\ln2}$ (Sections 2.2c, 2.3, and Problem 5). Its spatial standard deviation is derived in 2.2d. Radial profiles of the six transfer functions are in Figure 2.1."))
    B.append(("fig", "figures/q2_radial_profiles.png", "Radial profiles H(D) of the six transfer functions on a common axis (D0 = 40; GLPF with matched D0' = 33.97), and the Butterworth order family.", 1.0))
    # 2.1b
    b = R["q2_1b"]
    B.append(("h2", "2.1b Energy-based cutoff selection"))
    P = b["P"]
    rows = []
    for t in ("90", "95", "99", "99.5"):
        r = b["table"][t]
        ra = b["table_mean_removed_(AC_only)"][t]
        rows.append([t + " %", r["radius_bins_(padded_P=%d)" % P], g(r["alpha(r)"], 5), f(r["radius_in_cycles_per_pixel"], 4), f(r["radius_in_unpadded_N=688_grid_bins"], 1), ra["radius_bins_(padded_P=%d)" % P]])
    B.append(("table", "Table 2.2. Radii enclosing $\\alpha(r)=100\\sum_{D\\le r}|F|^2/\\sum|F|^2$ of the spectral power of characters_test_pattern.tif (688x688, zero-padded to P = 1376). Columns: smallest integer radius with alpha >= target (padded bins), the achieved alpha, the radius in cycles/pixel, the equivalent radius on the unpadded 688 grid, and (for comparison) the radius if the mean is removed first.", ["target alpha", "radius r (padded bins)", "achieved alpha (%)", "r (cycles/pixel)", "r on unpadded grid", "r with DC removed"], rows, "rrrrrr"))
    B.append(("p", "The single DC sample of the zero-padded %d x %d spectral grid carries %.4f%% of the total spectral power. This fraction is specific to the padded grid used in this table and changes with the zero-padding resolution: with the unnormalised DFT convention used here, the DC coefficient $F(0,0)=\\sum f$ is unchanged by zero-padding, whereas Parseval's relation makes the total squared-magnitude sum $\\sum|F|^2=P^2\\sum f^2$ scale with $P^2$. The padded DC fraction is therefore $(\\sum f)^2/(P^2\\sum f^2)$, which equals $(N/P)^2$ times the fraction $(\\sum f)^2/(N^2\\sum f^2)$ of the original unpadded 688 x 688 DFT. For the unpadded DFT the DC sample carries $\\overline{f}^{2}/\\overline{f^2}=%.4f\\%%$ of the power, and $%.4f\\%%\\times(688/1376)^2=%.4f\\%%$. The mean grey level is 207 of 255 (a bright page with dark glyphs). Natural and graphic images have power spectra falling like $1/D^{\\beta}$, so 90%% of the power lies within $r=%d$ bins (0.012 cycles/pixel) although the Nyquist radius is 688. But power is not visual importance: the energy is dominated by the large, smooth bright background; the *information* that makes the characters legible (edge positions, stroke widths, serif detail, the noise-like texture patches) is carried by the small residual power at high frequencies. A filter that keeps 99%% of the energy ($r=410$) is visually near-lossless, whereas one that keeps 95%% ($r=46$) already shows strong ringing and blur (Figure 2.4). Edges are discontinuities whose Fourier amplitude decays only as $1/D$, so many weak coefficients are needed to place them." % (P, P, b["DC_share_%"], dc_unpadded_percent(), dc_unpadded_percent(), dc_unpadded_percent() * (688 / 1376) ** 2, b["table"]["90"]["radius_bins_(padded_P=%d)" % P])))
    # 2.1c
    c = R["q2_1c"]
    B.append(("h2", "2.1c Inverse design problem"))
    B.append(("p", "For the Butterworth low-pass $H=1/(1+(D/D_0)^{2n})$ we have $1/H-1=(D/D_0)^{2n}$. From $H(10)=16/17$: $(10/D_0)^{2n}=1/16$. From $H(40)=1/17$: $(40/D_0)^{2n}=16$. Dividing the second by the first eliminates $D_0$:"))
    B.append(("eq", r"\Big(\frac{40}{10}\Big)^{2n}=256\ \Rightarrow\ 4^{2n}=4^{4}\ \Rightarrow\ n=2,\qquad (10/D_0)^{4}=\frac1{16}\Rightarrow D_0=20"))
    B.append(("p", "Smallest integer order with $H(2D_0)\\le 1\\%%$: $H(2D_0)=1/(1+2^{2n})=1/(1+4^n)\\le 0.01\\iff 4^n\\ge 99\\iff n\\ge 3.31$, so $n=4$ ($n=3$ gives $1/65=1.54\\%%$, $n=4$ gives $1/257=0.389\\%%$). Verification with `make_filter` (P=1024, $D_0=20$, $n=2$): $H(10)=%s$ (target $16/17=%s$), $H(40)=%s$ (target $1/17=%s$); $H(2D_0)$ for $n=1..6$ is %s, so the smallest order meeting the 1%% requirement is $n=%d$." % (g(c["H(10)_measured"], 8), g(c["target_16/17"], 8), g(c["H(40)_measured"], 8), g(c["target_1/17"], 8), ", ".join(g(c["H(2D0)_vs_n"][str(k)], 4) for k in range(1, 7)), c["smallest_n_for_H(2D0)<=1%"])))

    # 2.2
    pr = R["q2_2_prediction_file"]
    B.append(("h2", "2.2 Ringing: prediction, measurement and reconciliation"))
    B.append(("p", "**Pre-implementation prediction** (`predictions/q2_2_prediction.md`, written 2026-10-27 11:17:29 +0530, sha256 %s...; experiment run %s)." % (pr["sha256"][:16], pr["experiment_run_time"])))
    B.append(("p", "Far from the corners the step image varies only along $x$; its spectrum is confined to $v=0$, so only the slice $H(u,0)$ matters: for the ILPF a 1-D ideal low-pass with cutoff $f_c=D_0/P$. Its impulse response is $2f_c\\,\\mathrm{sinc}(2f_cx)$ and its unit-step response is"))
    B.append(("eq", r"F(t)=\frac12+\frac{1}{\pi}\mathrm{Si}(2\pi f_c t),\qquad \mathrm{Si}(z)=\int_0^z\frac{\sin w}{w}\,dw"))
    B.append(("p", "so $g(x)=A+(B-A)F(x-x_e)$ near one edge (and $A+(B-A)[F(x-x_1)-F(x-x_2)]$ for the square). $F'(t)=\\sin(2\\pi f_ct)/(\\pi t)$ first vanishes at $2\\pi f_c t^*=\\pi$:"))
    B.append(("eq", r"t^{*}=\frac{1}{2f_c}=\frac{P}{2D_0},\qquad \mathrm{OS}=100\Big[\frac12+\frac{\mathrm{Si}(\pi)}{\pi}-1\Big]=100\,[0.0895]=8.95\%\ \ \text{(independent of }D_0)"))
    B.append(("p", "Predicted first-peak distances for $P=1024$: 51.2, 25.6, 12.8, 6.4 px for $D_0=10,20,40,80$ (discrete-grid refinement $P/(2D_0+1)$: 48.8, 25.0, 12.6, 6.4). The Gaussian filter has the non-negative Gaussian kernel of $\\sigma_s=P/(2\\pi D_0')$ and its step response is a Gaussian CDF: predicted overshoot exactly 0%."))
    rows = []
    rec = {r["D0"]: r for r in R["q2_2a"]["reconciliation"]}
    for r in R["q2_2a"]["ideal"]:
        m = r["measured"]
        rr = rec[r["D0"]]
        rows.append([r["D0"], f(r["pred_overshoot_%_single_edge"], 2), f(r["two_edge_Si_formula_overshoot_%_(fc=(D0+.5)/P)"], 2), f(m["overshoot_%"], 2), f(rr["overshoot_%_x-only_strip_(1-D_exact)"], 2),
                     f(r["pred_peak_distance_px_continuous_P/(2D0)"], 1), f(r["pred_peak_distance_px_discrete_P/(2D0+1)"], 1), f(m["first_peak_distance_px"], 1), f(rr["peak_distance_px_strip"], 1)])
    B.append(("table", "Table 2.3. ILPF on the 512x512 step image (A=50, B=200, square side 256), profile along the middle row y=256. OS = 100 (max g - B)/(B - A). 'Single-edge' = 8.95% (prediction); 'two-edge Si' = closed form with both edges and f_c=(D0+1/2)/P; 'measured' = assignment pipeline (zero padding, P=1024); 'x-only strip' = control experiment with an image that really depends on x only (rows identical, background padding). Peak distance from the edge at x=127.5 to the first local maximum inside the square.",
              ["D0", "pred OS % (single edge)", "pred OS % (two-edge Si)", "measured OS % (pipeline)", "measured OS % (x-only strip)", "pred dist (P/2D0)", "pred dist (P/(2D0+1))", "measured dist (px)", "strip dist (px)"], rows, "rrrrrrrrr"))
    B.append(("fig", "figures/q2_ringing_profiles.png", "Measured middle-row profiles (blue) vs the two-edge sine-integral closed form (dashed) for D0 = 10, 20, 40, 80; the dotted red line is the single-edge prediction B + 0.0895(B-A).", 1.0))
    B.append(("p", "**Reconciliation.** (i) The first-peak distances agree with the prediction to within one pixel at all four cutoffs (50.5, 26.5, 12.5, 6.5 px measured; deviations from $P/(2D_0)$: -0.7, +0.9, -0.3, +0.1 px; the discrete-grid refinement $P/(2D_0+1)$ is closer only at $D_0=40$ and not systematically better). (ii) For $D_0=40$ and $80$ the measured overshoot (8.27%, 8.78%) is within 0.7 and 0.2 percentage points of the 8.95% prediction; the remaining gap is the sampling of the peak on a 1-px grid (peak width $\\sim t^*$) and the second edge. (iii) For small $D_0$ the prediction fails: 3.44% measured at $D_0=10$ and 11.45% at $D_0=20$. The pre-implementation prediction named the zero-padding border edge as a possible cause; control experiments **rule this out for $D_0=10$**: padding with the background value $A$ (no $50\\to0$ jump) still gives only 2.26% (and 12.53% at $D_0=20$), whereas the x-only strip reproduces the 1-D theory (8.74%, 8.84%, 8.97%, 9.02% for $D_0=10..80$, within 0.2 pp of 8.95%, and within 1.4, 0.9, 0.3, 0.1 pp of the two-edge Si value for $D_0=10,20,40,80$; the residual at small $D_0$ comes from discrete-grid effects the closed form ignores). The dominant cause is therefore that the square is only 256 px tall while the ripple scale of the filter ($P/D_0=102$ px at $D_0=10$) is comparable: the image is not one-dimensional on that scale and the 2-D radially symmetric kernel (a jinc, not a sinc) integrates the other edges. The 1-D argument is valid only when the extent of the object in $y$ is large compared with $t^*$ (and $x$-extent with $t^*$ for the second edge's $\\mathrm{Si}$ tail, which changes the overshoot by $\\pm1$-2 pp at $D_0=10,20$ as the two-edge formula shows). The overshoot itself does not shrink with $D_0$: the Gibbs phenomenon only moves closer to the edge (6.5 px at $D_0=80$)."))
    rows = [[r["D0"], f(r["D0_prime"], 2), f(r["sigma_s_px"], 2), g(r["measured_overshoot_%_(raw,negative=max_below_B)"], 3), f(r["max_g_whole_image"] - 200, 12) if False else e(r["max_g_whole_image"] - 200), f(r["min_g_row"], 2)] for r in R["q2_2a"]["gaussian"]]
    B.append(("table", "Table 2.4. Gaussian low-pass (matched D0') on the step image: spatial sigma_s = P/(2 pi D0'), measured overshoot on the profile row (0 within 1e-12 %), maximum of the output over the whole image minus B (<= 0 up to rounding), minimum along the row.", ["D0", "D0'", "sigma_s (px)", "overshoot % (row)", "max(g) - B (whole image)", "min g along row"], rows, "rrrrrr"))
    B.append(("p", "The Gaussian overshoot is zero to $10^{-13}$ at all cutoffs, as predicted (a non-negative kernel of unit sum cannot exceed $\\max s$; see the proof below)."))
    cr = R["q2_2a"]["near_corner"]
    rows = [[r["D0"], r["row_offset_from_top_edge_px"], f(r["overshoot_%"], 2), "none" if r["first_peak_distance_px"] is None else f(r["first_peak_distance_px"], 1), f(r["pred_1D_overshoot_%"], 2), f(r["pred_1D_distance"], 1)] for r in cr]
    B.append(("table", "Table 2.5. Profiles along rows close to the top edge of the square (corner at x=127.5): the 1-D prediction no longer applies. 'none' = the profile never rises above B (or has no interior local maximum above B).", ["D0", "row offset below top edge (px)", "OS % measured", "first peak dist (px)", "1-D pred OS %", "1-D pred dist"], rows, "rrrrrr"))
    B.append(("p", "**Near a corner** (rows 2-16 px from the top edge) the profile is a mixture of $x$- and $y$-ripples: the image is not a function of $x$ only, its spectrum has energy off the line $v=0$ where the radial filter cuts at $|u|<\\sqrt{D_0^2-v^2}$, so the effective 1-D cutoff in $u$ is smaller and depends on $v$, and the response is the 2-D jinc convolved with a quarter-plane. The overshoot is then strongly modulated: at $D_0=40$ it is $-25.6\\%$ (row 2: the profile never reaches $B$ because the blurred top edge is still rising), 1.3% (row 6), 16.6% (row 16: constructive superposition of the $x$-ripple and the $y$-ripple, 1.9 times the 1-D value) and returns towards 8.3% at 128 px (middle row)."))
    # 2.2b
    rb = R["q2_2b"]
    B.append(("h2", "2.2b Butterworth order sweep"))
    rows = [[r["n"], f(r["overshoot_%"], 3), "none" if r["first_peak_distance_px"] is None or r["overshoot_%"] < 1e-3 else f(r["first_peak_distance_px"], 1), g(r["max_h"], 4), g(r["min_h"], 4), g(r["|min h|/max h"], 4)] for r in rb["rows"]]
    rows.append(["ILPF (ref.)", f(rb["ILPF_measured_overshoot_D0=40_%"], 3), "12.5", "-", "-", g(rb["ILPF_kernel_ratio"], 4)])
    rows.append(["GLPF (ref.)", "0", "-", "-", "-", g(rb["GLPF_kernel_ratio"], 3)])
    B.append(("table", "Table 2.6. Butterworth low-pass at D0 = 40 (P=1024): measured overshoot on the middle row, 1-D kernel h(x) = IDFT{H(u,0)} extrema and the ratio |min h|/max h. Ideal-filter limit (prediction): 8.95%; ILPF measured with the same pipeline: 8.27%.", ["n", "overshoot %", "first peak dist (px)", "max h", "min h", "|min h|/max h"], rows, "rrrrrr"))
    B.append(("fig", "figures/q2_butterworth_kernels.png", "Left: 1-D spatial kernels h(x) = F^-1{H(u,0)} of the Butterworth filters at D0=40: n=1 is positive, higher orders develop negative side lobes. Right: step responses.", 1.0))
    B.append(("p", "**Why no ringing for n=1 but ringing for higher orders.** For $n=1$, $H(u,0)=1/(1+(u/D_0)^2)$ is a Lorentzian whose inverse transform is the two-sided exponential $\\propto e^{-2\\pi D_0|x|/P}$, strictly positive: $|\\min h|/\\max h=2.5\\times10^{-6}$ (the tiny negative value is the discrete Nyquist truncation, $H(P/2)=0.006$), and the measured overshoot is $9\\times10^{-6}$%. As $n$ grows the kernel acquires negative side lobes (ratio 0.043, 0.097, 0.158, 0.200, 0.212 for $n=2,3,5,10,20$, approaching the ILPF's 0.217) and the overshoot grows monotonically with this ratio: 3.36%, 5.60%, 7.50%, 8.49%, 8.43% towards the ideal-filter limit (8.95% predicted, 8.27% measured for the ILPF with the same pipeline; the last two values are within the sampling error of each other). The unit-step response is the running sum of the kernel, $s(x)=\\sum_{t\\le x}h(t)$, which overshoots exactly when the kernel has negative lobes."))
    B.append(("p", "**Proof that a non-negative, unit-sum kernel cannot overshoot.** Let $h\\ge0$, $\\sum h=1$ and $g=h\\ast s$, i.e. $g(x)=\\sum_t h(t)s(x-t)$. Because the weights are non-negative and sum to one, $g(x)$ is a convex combination of values of $s$, so $\\min s\\le g(x)\\le\\max s$ for all $x$; for the step image $A\\le g\\le B$, so $\\mathrm{OS}\\le0$. Moreover the step response $\\sum_{t\\le x}h(t)$ is non-decreasing because every increment $h(t)\\ge0$, so it is monotone. Conversely, overshoot requires $\\sum_{t\\le x}h(t)>1$ for some $x$, i.e. a negative part of $h$. For the 2-D filter acting on an x-only image, the relevant 1-D kernel is the projection (marginal) of the 2-D kernel, which is again non-negative when the 2-D kernel is: the Gaussian satisfies the hypothesis (its kernel is a positive Gaussian, ratio $6\\times10^{-17}$ numerically), the ideal filter and Butterworth $n\\ge2$ do not."))
    # 2.2c
    cc = R["q2_2c"]
    B.append(("h2", "2.2c Visual comparison on the characters test pattern"))
    B.append(("p", "ILPF, BLPF ($n=2$) and GLPF (matched $D_0'=D_0/\\sqrt{2\\ln2}$) were applied at the four radii of Table 2.2 ($D_0=%s$, i.e. alpha = 90/95/99/99.5%%). Figure 2.4 shows the grid; red arrows R1 and R2 mark two locations where ILPF ringing is visible at $D_0=46$ (concentric ripples around the circles at R1 and the squares at R2, also the periodic 'ghost' ripples across the whole flat background), and Figure 2.5 shows zoomed insets of the same locations for the three filter families." % ", ".join(str(x) for x in cc["radii"])))
    B.append(("fig", "figures/q2_visual_grid.png", "Filter type (rows: ILPF, BLPF n=2, GLPF with matched D0') versus cutoff (columns: the radii enclosing 90, 95, 99, 99.5% of the power). Red arrows R1, R2: ringing locations.", 1.0))
    B.append(("fig", "figures/q2_ringing_insets.png", "Zoomed insets at R1 and R2 (original, ILPF, BLPF n=2, GLPF) at D0 = 46: ripples in the ideal case only (faint overshoot for n=2, none for the Gaussian).", 0.9))
    mm = cc["min_max_of_outputs_(ringing_pushes_beyond_[0,255])"]
    rms = cc["metrics_rms_error_vs_original"]
    rows = []
    for nm in ("ILPF", "BLPF n=2", "GLPF D0'"):
        rows.append([nm] + ["%s [%s, %s]" % (f(rms["%s r=%d" % (nm, r)], 1), f(mm["%s r=%d" % (nm, r)][0], 0), f(mm["%s r=%d" % (nm, r)][1], 0)) for r in cc["radii"]])
    B.append(("table", "Table 2.7. RMS error of the filtered image relative to the original and [min, max] of the output (the original range is [0,255]; values outside it are pure ringing overshoot).", ["filter"] + ["radius %d" % r for r in cc["radii"]], rows, "lrrrr"))
    B.append(("p", "**Sharpness-ringing trade-off.** At any cutoff the ideal filter retains the sharpest transitions but produces the strongest ringing (output pushed to $[-134,416]$ at $D_0=559$ against $[-16,273]$ for the Gaussian); the Butterworth filter ($n=2$) is intermediate (about 4% overshoot, one weak ripple); the Gaussian is monotone but gives the widest transitions for the same half-amplitude radius. RMS errors against the original are lowest for the Gaussian/Butterworth at 410 and 559 because the ILPF's ripples add error even where the passband is nearly everything; at 16 and 46 all three are blurred and the ILPF is additionally disturbed by ripples that look like a texture of the glyphs themselves (e.g. 'ghost' squares in the flat area)."))
    # 2.2d
    d = R["q2_2d"]
    B.append(("h2", "2.2d Gaussian equivalence check"))
    B.append(("p", "A spatial Gaussian $g(x)=\\frac{1}{\\sqrt{2\\pi}\\sigma_s}e^{-x^2/(2\\sigma_s^2)}$ has transfer function $e^{-2\\pi^2\\sigma_s^2 f^2}$ with $f$ in cycles/pixel. On a $P$-point grid $f=k/P$ with $k=u-P/2$ the frequency-domain filter is $e^{-k^2/(2D_0^2)}$, so $2\\pi^2\\sigma_s^2/P^2=1/(2D_0^2)$:"))
    B.append(("eq", r"\sigma_s=\frac{P}{2\pi D_0}"))
    rows = [[r["D0"], f(r["sigma_s_px"], 3), e(r["H_at_Nyquist_exp(-(P/2)^2/(2D0^2))"]), e(r["max_abs_err_truncate4_zero_boundary"]), e(r["max_abs_err_truncate8_zero_boundary"]), e(r["max_abs_err_truncate8_periodic_padded"])] for r in d["rows"]]
    B.append(("table", "Table 2.8. cameraman.tif (P=1024): max |g_freq - g_spatial| (grey levels, 0-255 scale) against scipy.ndimage.gaussian_filter with sigma_s = P/(2 pi D0): default truncation at 4 sigma with zero boundary (mode='constant'), truncation at 8 sigma, and 8 sigma on the zero-padded periodic P x P array (isolating the kernel sampling difference).", ["D0", "sigma_s (px)", "H at Nyquist", "err (truncate 4, zero bdry)", "err (truncate 8, zero bdry)", "err (truncate 8, periodic)"], rows, "rrrrrr"))
    B.append(("p", "The error is not exactly zero for two reasons. (1) *Kernel truncation*: `gaussian_filter` cuts the kernel at 4$\\sigma_s$ (mass $\\mathrm{erfc}(4/\\sqrt2)=6.3\\times10^{-5}$ lost per axis, giving the $10^{-2}$ errors of column 4 for $D_0=10,40$); with truncation at 8$\\sigma_s$ the two agree to $10^{-13}$, which confirms the derivation of $\\sigma_s$ and the pipeline (zero padding $\\equiv$ `mode='constant'`). (2) *Grid sampling*: the frequency-domain filter is band-limited at the Nyquist frequency (it only exists for $|k|\\le P/2$), whereas the spatial Gaussian sampled on the pixel grid is aliased; the discrepancy is controlled by the Gaussian's value at Nyquist, $e^{-(P/2)^2/(2D_0^2)}$, which is $0$ for $D_0=10,40$ but $%s$ for $D_0=160$ ($\\sigma_s=1.02$ px). Hence the error **grows with $D_0$** and is not reduced by a longer kernel (0.0698 at $D_0=160$ with either truncation): a Gaussian narrower than about one pixel is not representable on the grid." % e(d["rows"][2]["H_at_Nyquist_exp(-(P/2)^2/(2D0^2))"])))

    # 2.3
    a3 = R["q2_3a"]
    B.append(("h2", "2.3a Noise-blur trade-off"))
    B.append(("p", "Zero-mean Gaussian noise with $\\sigma=20$ (seed 2026, drawn with `numpy.random.default_rng(2026)`, **not clipped**) was added to cameraman.tif (PSNR of the noisy input: %.2f dB, SSIM %.3f). PSNR (clipping the output to [0,255] for the metric) was evaluated against the clean image for $D_0=5..150$ (as required) and, because all three curves are still rising at $D_0=150$, for $D_0$ up to 300 so that the true optimum is found; for the Gaussian the sweep parameter is the half-amplitude radius $D_0$ with $D_0'=D_0/\\sqrt{2\\ln2}$." % (a3["PSNR_noisy_clipped_for_metric"], a3["SSIM_noisy"])))
    B.append(("fig", "figures/q2_psnr_sweep.png", "PSNR vs D0 for the three low-pass families on one plot (noise sigma=20, seed 2026); dots mark each optimum. Right: the three outputs at their optimum.", 1.0))
    ox, osr = a3["optimum_extended_sweep"], a3["optimum_within_spec_range_5_150"]
    rows = [[k, osr[k]["best_D0_within_5..150"], f(osr[k]["PSNR_dB"], 2), ox[k]["best_D0"], f(ox[k]["peak_PSNR_dB"], 2), f(ox[k]["SSIM"], 3)] for k in ("ILPF", "BLPF n=2", "GLPF")]
    B.append(("table", "Table 2.9. Optimal cutoff and peak PSNR. Columns 2-3: best D0 inside the required range 5..150 (at the edge of the range for every family); columns 4-6: optimum of the extended sweep (D0 <= 300, step 2 beyond 150) with PSNR and SSIM.", ["family", "best D0 in 5..150", "PSNR (dB)", "best D0 (extended)", "peak PSNR (dB)", "SSIM at optimum"], rows, "lrrrrr"))
    B.append(("p", "**Interpretation.** Noise is white (flat spectrum, power $\\sigma^2$) while the image power falls rapidly with frequency, so a low-pass trades removed noise power (proportional to the passband *area*, $\\propto D_0^2$) against removed signal power (the image energy beyond $D_0$). Small $D_0$ blurs the image (signal loss dominates), large $D_0$ passes almost all noise; the maximum is where marginal noise power equals marginal signal power: near $D_0\\approx 210$ bins ($0.2$ cycles/pixel, i.e. about 40%% of Nyquist), where the cameraman's spectrum has fallen to the noise floor. The Butterworth and Gaussian filters peak slightly higher than the ILPF because their smooth roll-off attenuates the transition band gradually; the ideal filter's abrupt cutoff gives a sinc kernel with slowly decaying side lobes, i.e. ringing (the added error) that remains even at the optimum, so it has the lowest peak (%.2f dB vs %.2f / %.2f dB; its SSIM %.3f is also clearly lower). The Butterworth ($n=2$) is the best of the three here (+%.2f dB over ILPF): it keeps a flatter passband than the Gaussian (whose attenuation already reaches $e^{-1/2}$ at $D_0'$ and 0.5 at $D_0$) while still rolling off smoothly, and the two optima are within 0.16 dB." % (ox["ILPF"]["peak_PSNR_dB"], ox["BLPF n=2"]["peak_PSNR_dB"], ox["GLPF"]["peak_PSNR_dB"], ox["ILPF"]["SSIM"], ox["BLPF n=2"]["peak_PSNR_dB"] - ox["ILPF"]["peak_PSNR_dB"])))
    # 2.3b
    t = R["q2_3b"]
    ts = R["q2_3b_sweep"]
    B.append(("h2", "2.3b Character repair"))
    rows = [[r["D0_gauss"], r["components_of_dark_pixels_(8-conn)"], f(r["dark_area_fraction_%"], 2), f(r["min_value"], 1)] for r in ts["rows"]]
    B.append(("p", "text_gaps.tif (444x508, padded to P=1016) contains 1-2 pixel gaps in the glyph strokes. A Gaussian low-pass was swept and a proxy was computed: the number of 8-connected components of dark pixels (< 127.5) - the original has %d, many being broken stroke fragments - together with the dark area fraction (Table 2.10). The proxy alone cannot choose $D_0$ (a very small $D_0$ yields one component, because everything merges into a grey smear); it was combined with visual inspection of Figure 2.7." % ts["original_components"]))
    B.append(("table", "Table 2.10. Gaussian LPF sweep on text_gaps.tif (parameter = Gaussian D0 on the P=1016 grid).", ["D0 (Gaussian)", "dark components", "dark area %", "min grey level"], rows, "rrrr"))
    B.append(("fig", "figures/q2_text_sweep.png", "Gaussian low-pass sweep on text_gaps.tif. Below D0~60 the words are smeared; D0~80 joins the broken strokes while the characters remain legible; at D0=160 the gaps are still visible.", 1.0))
    B.append(("fig", "figures/q2_text_repair.png", "Chosen repair: Gaussian D0' = 80 (half-amplitude radius 94.2) next to the ideal filter with the same half-amplitude radius.", 1.0))
    B.append(("fig", "figures/q2_text_repair_zoom.png", "Zoom of the top-left region: original, GLPF D0'=80, ILPF at radius 94.2.", 1.0))
    B.append(("p", "**Choice: Gaussian $D_0'=%d$** (half-amplitude radius $%.1f$ bins; spatial $\\sigma_s=P/(2\\pi D_0')=%.2f$ px, so the gap of 1-2 px is spanned by one standard deviation of the kernel). At $D_0'=60$ the components drop to 23 and neighbouring letters merge ('words' become blobs, minimum grey level 46 - the text is unreadable); at $D_0'\\ge100$ between 264 and 372 components remain and the gaps are visible. At $D_0'=80$ the number of components falls from %d to %d: the broken strokes are joined (the remaining components are separate letters and dots) while counters of letters such as 'a' and 'e' are still open. **Ideal filter at the same half-amplitude radius:** it reaches %d components too (it also bridges gaps), but its sinc kernel produces ringing: the output ranges over [%.0f, %.0f] instead of [%.0f, 255], the glyphs are surrounded by parallel ripples, and the ripples create faint false strokes in the white background (RMS difference from the Gaussian result %.1f grey levels). The Gaussian never overshoots (non-negative kernel)." % (t["chosen_D0_gauss"], t["half_amplitude_radius"], 1016 / (2 * math.pi * t["chosen_D0_gauss"]), t["components_original"], t["components_gaussian"], t["components_ideal_same_half_radius"], t["ideal_out_range"][0], t["ideal_out_range"][1], t["gaussian_out_range"][0], t["ideal_ringing_rms_vs_gaussian"])))
    # 2.3c
    h = R["q2_3c"]
    B.append(("h2", "2.3c High-pass filtering, the DC term and high-frequency emphasis"))
    B.append(("p", "thumb_print.tif (962x1026, $P=2052$) was filtered with IHPF, BHPF ($n=4$) and GHPF at a common half-amplitude radius $D_0=%d$ bins (GHPF: $D_0'=%.1f$)." % (h["D0_common_half_amplitude_radius"], h["D0_common_half_amplitude_radius"] * 0.8493218)))
    rows = [[r["filter"], g(r["H_HP(P/2,P/2)"], 3), f(r["mean_cropped_output"], 3), e(r["mean_over_full_padded_PxP_output"]), f(r["std_output"], 2), f(r["min"], 1), f(r["max"], 1)] for r in h["highpass"]]
    B.append(("table", "Table 2.11. High-pass outputs on thumb_print.tif (input mean %.2f). H_HP(P/2,P/2) is the DC gain; 'mean (cropped)' is the mean of the cropped 962x1026 output; 'mean (padded)' the mean over the whole PxP output." % h["highpass"][0]["input_mean"], ["filter", "H_HP(DC)", "mean (cropped output)", "mean (full padded PxP)", "std", "min", "max"], rows, "lrrrrrr"))
    B.append(("p", "All three high-pass filters have $H_{HP}(P/2,P/2)=1-H_{LP}(0)=0$: the DC term is removed. The mean of the output over the full padded array is $\\hat f_{pad}[\\,]=H(0)\\cdot\\frac1{P^2}F(0,0)=0$ exactly (measured $\\sim10^{-17}$, Table 2.11); the mean of the *cropped* output is not exactly zero (1.2-1.7 grey levels, against an input mean of 142.7) because the crop discards the filtered padded region, where the filter output ($\\approx$ minus a smoothed copy of the image edge) has the opposite sign. The tonal information (the overall brightness and slow shading) is lost: the high-pass image is zero-mean with ridges as positive/negative excursions."))
    hf = h["hfe"]
    B.append(("p", "**High-frequency emphasis** $H_{HFE}=k_1+k_2H_{HP}$ with $k_1=1$, $k_2=0.5$ (and the Gaussian high-pass above): the DC gain is $k_1=1$ (so tone is preserved: output mean %.1f vs input %.1f, ratio %.3f), the gain at high frequencies is $k_1+k_2=1.5$, and the local contrast of ridges (mean standard deviation in 15x15 windows) rises from %.1f to %.1f (x%.2f) while 1.9%% of pixels leave $[0,255]$ (clipped for display; $k_2=1.5$ would clip 12%%, so the milder setting was chosen). Figure 2.10 shows the original, the three high-pass outputs, the emphasised image and the Otsu-thresholded binary ridge maps; Figure 2.11 is a zoomed crop." % (hf["mean_out"], hf["mean_in"], hf["mean_ratio_out/in"], hf["local_std_15x15_in"], hf["local_std_15x15_out"], hf["ridge_contrast_gain"])))
    B.append(("fig", "figures/q2_highpass.png", "Top: original and the IHPF/BHPF/GHPF outputs (display stretched; means are ~1). Bottom: high-frequency emphasis (k1=1, k2=0.5) and Otsu-thresholded binary ridge maps.", 1.0))
    B.append(("fig", "figures/q2_highpass_crops.png", "200x200 crops of the three binary ridge maps and of the HFE image.", 1.0))
    rows = [[r["filter"], g(r["otsu_threshold"], 3), f(r["foreground_%"], 1), r["components"], r["tiny_components_<20px"], f(r["tiny_per_10k_px"], 2)] for r in h["binary_ridge_maps"]]
    B.append(("table", "Table 2.12. Binary ridge maps (Otsu threshold of each high-pass output).", ["filter", "Otsu threshold", "foreground %", "components", "components < 20 px", "tiny per 10^4 px"], rows, "lrrrrr"))
    B.append(("p", "**Which filter gives the cleanest ridges?** In this experiment the three maps are visually almost identical (Figure 2.11); the counts of isolated tiny components (a proxy for speckle) are IHPF 926 < BHPF 1025 < GHPF 1467. The difference is small and not what the textbook intuition suggests ('Butterworth is the cleanest'). The explanation consistent with the transfer functions: at a common half-amplitude radius the Gaussian high-pass rises most slowly, so it passes least of the low-frequency ridge band near the cutoff and relatively more of the pixel-scale noise, whereas the ideal high-pass passes the ridge band completely but its sinc side lobes ring along the ridge direction (ringing is partly hidden because ridges are themselves quasi-periodic). The Butterworth filter is the compromise (no abrupt cutoff, little ringing, steep enough transition): it is the recommended choice in practice, but the quantitative evidence here shows only a marginal advantage of the sharper filters in speckle count and no decisive winner; the binary maps are limited more by the global Otsu threshold than by the filter."))
    return B
