"""Question 1 - DFT and the machinery of frequency.

Convention used throughout: arrays are indexed f[r, c] (axis 0 = r, axis 1 = c), frequencies
(u, v) are the integer DFT indices along axes 0 and 1, F[u, v] = sum_{r,c} f[r,c] e^{-2 pi i (u r + v c)/N}
(unnormalised forward, 1/N^2 in the inverse), exactly numpy's convention.
Run:  python src/q1_dft.py      -> results/q1.json, figures/q1_*.png
"""
import hashlib
import os
import time

import numpy as np
import scipy.signal

import common as C

N_SEED = 2026


# ------------------------------------------------------------------ 1.1 DFT from scratch
def dft_matrix(N, exact_twiddles=True):
    """W[k, n] = exp(-2 pi i k n / N).  With exact_twiddles the integer product kn is reduced
    mod N before the exponential, which keeps every argument in [0, 2 pi) and avoids the loss of
    accuracy of evaluating cos/sin at large angles."""
    k = np.arange(N)
    kn = np.outer(k, k)
    if exact_twiddles:
        kn = kn % N
    return np.exp(-2j * np.pi * kn / N)


def dft2(f, W=None):
    """2-D DFT by row-column separability: F = W f W^T."""
    f = np.asarray(f)
    W = dft_matrix(f.shape[0]) if W is None else W
    if f.shape[0] != f.shape[1]:
        raise ValueError("square arrays only")
    return W @ f @ W.T


def idft2(F, W=None):
    """Inverse: f = (1/N^2) conj(W) F conj(W)^T."""
    F = np.asarray(F)
    N = F.shape[0]
    Wc = np.conj(dft_matrix(N) if W is None else W)
    return Wc @ F @ Wc.T / N ** 2


def dft2_direct(f):
    """Literal quadruple sum (independent of dft_matrix and of separability); small N only."""
    N = f.shape[0]
    r = np.arange(N)
    F = np.zeros((N, N), complex)
    for u in range(N):
        for v in range(N):
            ph = np.exp(-2j * np.pi * (u * r[:, None] + v * r[None, :]) / N)
            F[u, v] = np.sum(f * ph)
    return F


def conj_partner(F):
    """F evaluated at (-u, -v) mod N for every (u, v)."""
    return np.roll(np.flip(F, axis=(0, 1)), 1, axis=(0, 1))


def circ_shift_phase(F, a, b):
    """Shift-theorem prediction: DFT of f(r-a, c-b) = F[u,v] e^{-2 pi i (u a + v b)/N}."""
    N = F.shape[0]
    u = np.arange(N)[:, None]
    v = np.arange(N)[None, :]
    return F * np.exp(-2j * np.pi * (u * a + v * b) / N)


def circ_conv_direct(f, K, centre):
    """Circular convolution (f (*) g)[r,c] = sum_{m,n} K[m,n] f[(r-(m-cm)) % N, (c-(n-cn)) % N]
    evaluated directly with array rolls, no FFT.  K is the small kernel, `centre` its origin."""
    out = np.zeros_like(f, dtype=float)
    for m in range(K.shape[0]):
        for n in range(K.shape[1]):
            if K[m, n] != 0:
                out += K[m, n] * np.roll(f, (m - centre[0], n - centre[1]), axis=(0, 1))
    return out


def embed_kernel(K, N, centre):
    """Place the small kernel in an N x N array with its origin at index (0,0) (wrapping)."""
    g = np.zeros((N, N))
    for m in range(K.shape[0]):
        for n in range(K.shape[1]):
            g[(m - centre[0]) % N, (n - centre[1]) % N] += K[m, n]
    return g


def conjugacy_class_counts(N):
    """Enumerate the orbits of (u,v) -> (-u,-v) mod N on the N x N index grid."""
    u, v = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    iu, iv = (-u) % N, (-v) % N
    fixed = int(np.sum((iu == u) & (iv == v)))
    classes = fixed + (N * N - fixed) // 2
    # real degrees of freedom: a fixed bin is real (1 dof), each 2-element orbit is complex (2 dof)
    dof = fixed * 1 + ((N * N - fixed) // 2) * 2
    stored_rfft2 = N * (N // 2 + 1)
    return {"N": N, "fixed_points": fixed, "conjugacy_classes": classes,
            "formula_(N^2+4)/2": (N * N + 4) / 2 if N % 2 == 0 else None,
            "real_dof": dof, "N^2": N * N, "rfft2_stored_bins": stored_rfft2,
            "rfft2_redundant_bins": stored_rfft2 - classes,
            "rfft2_stored_real_numbers": 2 * stored_rfft2}


# ------------------------------------------------------------------ sinusoid synthesis / recovery
SINUSOIDS = [  # (u, v, A, phi)   cycles/image along axis 0 and axis 1
    (12, 20, 100.0, 0.50),
    (-45, 30, 60.0, -1.20),
    (70, -15, 40.0, 2.00),
    (5, 5, 25.0, -2.50),
]


def synth(N, comps):
    r = np.arange(N)[:, None]
    c = np.arange(N)[None, :]
    f = np.zeros((N, N))
    for (u, v, A, ph) in comps:
        f += A * np.cos(2 * np.pi * (u * r + v * c) / N + ph)
    return f


def signed(idx, N):
    return idx - N if idx >= N // 2 else idx


def pick_peaks(F, N, n_peaks, rel_thresh=1e-3):
    """Spectral peaks alone: local maxima of |F| (8-neighbourhood, circular) above a threshold,
    restricted to the half plane (u>0) or (u==0 and v>0) because of conjugate symmetry."""
    mag = np.abs(F)
    mx = np.ones_like(mag, bool)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            if du or dv:
                mx &= mag >= np.roll(mag, (du, dv), axis=(0, 1))
    mx &= mag > rel_thresh * mag.max()
    peaks = []
    for (u, v) in zip(*np.nonzero(mx)):
        su, sv = signed(u, N), signed(v, N)
        if su > 0 or (su == 0 and sv > 0):
            peaks.append((mag[u, v], su, sv, F[u, v]))
    peaks.sort(key=lambda t: -t[0])
    out = []
    for m, su, sv, val in peaks[:n_peaks]:
        out.append({"u": su, "v": sv, "A": 2 * m / N ** 2, "phi": float(np.angle(val))})
    return out


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ 1.2 helpers
def gauss_smooth_fft(img, sigma):
    """Gaussian smoothing through the frequency domain (periodic), for correlation statistics."""
    N = img.shape[0]
    k = np.fft.fftfreq(N)
    H = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))


def radial_average(P2d):
    N = P2d.shape[0]
    k = np.fft.fftshift(np.arange(N) - N // 2)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    rbin = np.round(kk).astype(int)
    s = np.bincount(rbin.ravel(), weights=P2d.ravel())
    n = np.bincount(rbin.ravel())
    return s / np.maximum(n, 1)


def kurtosis(x):
    x = np.asarray(x).ravel()
    x = x - x.mean()
    return float(np.mean(x ** 4) / np.mean(x ** 2) ** 2)


def pearson(a, b):
    a = np.asarray(a).ravel() - np.mean(a)
    b = np.asarray(b).ravel() - np.mean(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def stretch(x, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-30), 0, 1)


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "seed": N_SEED}
    rng = np.random.default_rng(N_SEED)
    img = C.load_gray(os.path.join(C.PROVIDED, "q1_testimage.png"))
    N = img.shape[0]
    assert img.shape == (512, 512)
    R["test_image"] = {"file": "data/provided/q1_testimage.png", "shape": list(img.shape),
                       "min": float(img.min()), "max": float(img.max()), "mean": float(img.mean())}

    # ---------------- 1.1 correctness vs numpy
    t = {}
    with C.Timer() as tm:
        W = dft_matrix(N)
        F = dft2(img, W)
    t["custom_seconds"] = tm.dt
    with C.Timer() as tm:
        Fn = np.fft.fft2(img)
    t["numpy_seconds"] = tm.dt
    err_table = []
    for NN in (16, 64, 128, 256, 512):
        x = img[:NN, :NN]
        Fx = dft2(x)
        Fref = np.fft.fft2(x)
        Wn = dft_matrix(NN, exact_twiddles=False)
        Fx_naive = Wn @ x @ Wn.T
        e = np.abs(Fx - Fref)
        err_table.append({"N": NN, "max_abs_err_exact_twiddles": float(e.max()),
                          "max_abs_err_naive_twiddles": float(np.abs(Fx_naive - Fref).max()),
                          "max_abs_F": float(np.abs(Fref).max()),
                          "max_rel_err": float(e.max() / np.abs(Fref).max()),
                          "rel_L2_err": float(np.linalg.norm(Fx - Fref) / np.linalg.norm(Fref)),
                          "roundtrip_max_abs_err": float(np.abs(idft2(Fx) - x).max())})
    # random complex-valued check and independent direct quadruple sum
    xs = rng.normal(size=(16, 16))
    direct_err = float(np.abs(dft2_direct(xs) - dft2(xs)).max())
    direct_err_vs_np = float(np.abs(dft2_direct(xs) - np.fft.fft2(xs)).max())
    R["q1_1_dft_error"] = {"table": err_table, "timing": t, "direct_quadruple_sum_vs_custom_16x16": direct_err,
                           "direct_quadruple_sum_vs_numpy_16x16": direct_err_vs_np,
                           "full_512_max_abs_err": float(np.abs(F - Fn).max())}

    # ---------------- 1.1a conjugate symmetry
    Fc = conj_partner(F)
    sym_err = float(np.abs(F - np.conj(Fc)).max())
    Frand = np.fft.fft2(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    sym_err_complex_input = float(np.abs(Frand - np.conj(conj_partner(Frand))).max())
    R["q1_1a_conj_symmetry"] = {
        "max_abs_F_minus_conjF(-u,-v)_real_image": sym_err,
        "relative_to_max|F|": sym_err / float(np.abs(F).max()),
        "same_test_with_complex_input_(must_FAIL)": sym_err_complex_input / float(np.abs(Frand).max()),
        "counts_512": conjugacy_class_counts(512),
        "counts_odd_N=511_(no_Nyquist_bins)": conjugacy_class_counts(511),
        "counts_N=8": conjugacy_class_counts(8),
        "phase_antisymmetry_max_err_over_|F|>1e-3max": None,
    }
    big = np.abs(F) > 1e-3 * np.abs(F).max()
    dphi = wrap_pi(np.angle(F) + np.angle(Fc))
    R["q1_1a_conj_symmetry"]["phase_antisymmetry_max_err_over_|F|>1e-3max"] = float(np.abs(dphi[big]).max())
    R["q1_1a_conj_symmetry"]["magnitude_evenness_max_rel_err"] = float(np.abs(np.abs(F) - np.abs(Fc)).max() / np.abs(F).max())

    # ---------------- 1.1b shift theorem
    shift_rows = []
    for (a, b) in [(0, 0), (1, 0), (0, 1), (5, 3), (-17, 40), (100, -37), (256, 256), (333, 101), (511, 511)]:
        fs = np.roll(img, (a, b), axis=(0, 1))
        Fs = dft2(fs, W)
        pred = circ_shift_phase(F, a, b)
        dmag = np.abs(np.abs(Fs) - np.abs(F))
        sig = np.abs(F) > 1e-3 * np.abs(F).max()
        ph_err = np.abs(wrap_pi(np.angle(Fs) - np.angle(pred)))[sig]
        # independent estimate of the shift from the phase difference at the strongest non-DC bins
        shift_rows.append({"shift_(r,c)": [a, b],
                           "max_abs_|DFT(shifted)-predicted|": float(np.abs(Fs - pred).max()),
                           "rel_to_max|F|": float(np.abs(Fs - pred).max() / np.abs(F).max()),
                           "max_||Fs|-|F||": float(dmag.max()),
                           "max_phase_error_rad_(|F|>1e-3max)": float(ph_err.max()),
                           "n_bins_compared": int(sig.sum()),
                           "spatial_check_max|f_shift-roll|": float(np.abs(np.real(idft2(pred, W)) - fs).max())})
    R["q1_1b_shift_theorem"] = shift_rows

    # ---------------- 1.1b circular convolution theorem
    lap5 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], float)
    gx = np.arange(-7, 8)
    g15 = np.exp(-(gx[:, None] ** 2 + gx[None, :] ** 2) / (2 * 3.0 ** 2))
    g15 /= g15.sum()
    rk = rng.random((63, 63))
    rk /= rk.sum()
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    conv_rows = []
    for name, K in [("Laplacian 5x5", lap5), ("Sobel 3x3", sobel), ("Gaussian 15x15 (sigma=3)", g15), ("random positive 63x63", rk)]:
        cen = (K.shape[0] // 2, K.shape[1] // 2)
        spatial = circ_conv_direct(img, K, cen)  # no FFT involved
        g = embed_kernel(K, N, cen)
        viaF = np.real(idft2(dft2(img, W) * dft2(g, W), W))
        e = np.abs(spatial - viaF)
        conv_rows.append({"kernel": name, "taps": int(K.size),
                          "max_abs_err": float(e.max()),
                          "rel_L2_err_(||a-b||2/||a||2)": float(np.linalg.norm(spatial - viaF) / np.linalg.norm(spatial)),
                          "max_abs_err_over_max|spatial|": float(e.max() / np.abs(spatial).max()),
                          "max|spatial|": float(np.abs(spatial).max())})
    R["q1_1b_convolution_theorem"] = conv_rows
    # dual theorem: DFT(f*g) = (1/N^2) F (*) G, small crop so the direct spectral convolution is cheap
    n2 = 32
    f2 = img[100:100 + n2, 200:200 + n2]
    g2 = np.hanning(n2)[:, None] * np.hanning(n2)[None, :] * 10
    W2 = dft_matrix(n2)
    lhs = dft2(f2 * g2, W2)
    F2, G2 = dft2(f2, W2), dft2(g2, W2)
    rhs = np.zeros_like(F2)
    for m in range(n2):
        for n in range(n2):
            rhs += F2[m, n] * np.roll(G2, (m, n), axis=(0, 1))
    rhs /= n2 ** 2
    R["q1_1b_dual_theorem_32x32"] = {"max_abs_err": float(np.abs(lhs - rhs).max()),
                                     "rel_to_max|lhs|": float(np.abs(lhs - rhs).max() / np.abs(lhs).max())}

    # ---------------- 1.1c sinusoid recovery
    NS = 512
    fs_img = synth(NS, SINUSOIDS)
    Fs = dft2(fs_img)
    peaks = pick_peaks(Fs, NS, 4)
    rec_rows = []
    for (u, v, A, ph) in sorted(SINUSOIDS, key=lambda t: -t[2]):
        # ground truth reported in the same half plane convention
        if u < 0 or (u == 0 and v < 0):
            u, v, ph = -u, -v, -ph
        match = [p for p in peaks if p["u"] == u and p["v"] == v]
        m = match[0] if match else None
        rec_rows.append({"true": {"u": u, "v": v, "A": A, "phi": wrap_pi(ph)},
                         "recovered": m,
                         "err_A": None if m is None else m["A"] - A,
                         "err_phi": None if m is None else float(wrap_pi(m["phi"] - ph))})
    # the same with noise and the same peak picking (not part of the ground truth)
    noisy = fs_img + rng.normal(0, 5.0, fs_img.shape)
    peaks_noisy = pick_peaks(dft2(noisy), NS, 4, rel_thresh=0.05)
    R["q1_1c_sinusoids"] = {"true_parameters_(u,v,A,phi)": SINUSOIDS, "N": NS, "recovered_noiseless": peaks,
                            "comparison": rec_rows, "noise_sigma": 5.0, "recovered_noisy": peaks_noisy,
                            "delta_f_cycles_per_pixel": 1.0 / NS, "delta_f_cycles_per_image": 1.0}
    # off-grid component: leakage, scalloping, refinement with zero padding (numpy FFT used for the padded transform)
    comp_off = [(40.5, 22.0, 80.0, 0.7)]
    foff = synth(NS, [(40.5, 22.0, 80.0, 0.7)])
    Foff = dft2(foff)
    pk = pick_peaks(Foff, NS, 3)
    pad = 4
    Fp = np.fft.fft2(foff, s=(pad * NS, pad * NS))
    mag = np.abs(Fp)
    iu, iv = np.unravel_index(np.argmax(mag[:pad * NS // 2, :]), mag[:pad * NS // 2, :].shape)
    R["q1_1c_off_grid"] = {"true": {"u": 40.5, "v": 22.0, "A": 80.0, "phi": 0.7},
                           "grid_peaks_(top3)": pk,
                           "zero_pad_factor": pad,
                           "padded_peak": {"u": (iu / pad), "v": signed(iv, pad * NS) / pad,
                                           "A_from_peak": 2 * mag[iu, iv] / NS ** 2},
                           "max_amp_loss_dB_at_half_bin_theory_(sinc(0.5)^2_in_2D_for_both_axes_off)": None}
    # 1-D resolution experiment: two equal, in-phase / quadrature sinusoids separated by delta bins (v = 0)
    res_rows = []
    L = 8192
    r = np.arange(NS)
    for delta in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        for phase2, label in ((0.0, "in phase"), (np.pi / 2, "quadrature")):
            col = np.cos(2 * np.pi * 50 * r / NS) + np.cos(2 * np.pi * (50 + delta) * r / NS + phase2)
            sp = np.abs(np.fft.fft(col, L))[:L // 2]          # zero padded x16: smooth interpolation of the DTFT
            fgrid = np.arange(L // 2) / L * NS
            win = (fgrid > 44) & (fgrid < 56 + delta)
            s_, f_ = sp[win], fgrid[win]
            lm = np.where((s_[1:-1] > s_[:-2]) & (s_[1:-1] > s_[2:]) & (s_[1:-1] >= 0.5 * s_.max()))[0] + 1   # main lobes only
            dip = float(s_[lm[0]:lm[-1] + 1].min() / min(s_[lm])) if len(lm) >= 2 else None
            res_rows.append({"separation_bins(=cycles/image)": delta, "phase": label,
                             "main_peaks_in_padded_spectrum": int(len(lm)),
                             "peak_locations": [float(f_[i]) for i in lm],
                             "valley/lower_peak": dip,
                             "resolved_(2_peaks_and_valley<0.9)": bool(len(lm) >= 2 and dip < 0.9)})
    R["q1_1c_resolution"] = res_rows

    # ---------------- 1.2 phase / magnitude  (prediction file hash recorded)
    pred_path = os.path.join(C.PREDICTIONS, "q1_2_prediction.md")
    R["q1_2_prediction_file"] = {"path": "predictions/q1_2_prediction.md",
                                 "sha256": hashlib.sha256(open(pred_path, "rb").read()).hexdigest(),
                                 "file_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(pred_path))),
                                 "experiment_run_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    mag, phase = np.abs(F), np.angle(F)
    c0 = 1.0
    fhat_flat = idft2(c0 * np.exp(1j * phase), W)
    imag_ratio = float(np.abs(fhat_flat.imag).max() / np.abs(fhat_flat.real).max())
    fhat_flat = fhat_flat.real
    noise = rng.normal(size=(N, N))
    Rn = np.abs(np.fft.fft2(noise))
    Rn = Rn / np.sqrt(np.mean(Rn ** 2)) * c0   # same energy scale as flat magnitude
    fhat_noise = np.real(idft2(Rn * np.exp(1j * phase), W))
    ref_l = C.luma(C.load_rgb(os.path.join(C.PROVIDED, "reference_image.jpg")))   # a different natural scene
    h0, w0 = ref_l.shape
    top, left = (h0 - 2048) // 2, (w0 - 2048) // 2
    cam = ref_l[top:top + 2048, left:left + 2048].reshape(512, 4, 512, 4).mean(axis=(1, 3))   # 4x4 block mean -> 512x512
    Fcam = dft2(cam, W)
    fhat_swap = np.real(idft2(np.abs(Fcam) / np.sqrt(np.mean(np.abs(Fcam) ** 2)) * np.exp(1j * phase), W))
    fhat_swap_rev = np.real(idft2(np.abs(F) * np.exp(1j * np.angle(Fcam)), W))
    fhat_zp_c = idft2(mag, W)
    fhat_zp = fhat_zp_c.real
    gy, gx_ = np.gradient(img)
    gmag = np.hypot(gx_, gy)
    rad = radial_average(np.abs(dft2(fhat_flat, W)) ** 2)
    k_edge = np.arange(len(rad)) < N // 2
    sm = 3.0
    q12 = {
        "real_valued_max|Im|/max|Re|": imag_ratio,
        "parseval_sum_f2_over_c2_(pred_1)": float(np.sum(fhat_flat ** 2) / c0 ** 2),
        "mean_fhat_times_N2_over_c_(pred_1)": float(fhat_flat.mean() * N ** 2 / c0),
        "radial_power_flatness_std/mean_(r<256)": float(rad[k_edge].std() / rad[k_edge].mean()),
        "corr(fhat_flat,f)_(pred_|r|<0.1)": pearson(fhat_flat, img),
        "corr(smoothed|fhat_flat|,smoothed_|grad_f|)_(pred_>0.3)": pearson(gauss_smooth_fft(np.abs(fhat_flat), sm), gauss_smooth_fft(gmag, sm)),
        "corr(fhat_flat,f)_exact_Parseval_prediction_E|F|/sqrt(E|F|^2)_(non-DC_bins)": float(np.mean(np.delete(mag.ravel(), 0)) / np.sqrt(np.mean(np.delete(mag.ravel(), 0) ** 2))),
        "kurtosis_fhat_flat": kurtosis(fhat_flat), "kurtosis_f": kurtosis(img),
        "kurtosis_gaussian_reference": 3.0,
        "corr(fhat_noise,fhat_flat)_(pred_0.886)": pearson(fhat_noise, fhat_flat),
        "pred_rho_sqrt(pi)/2": float(np.sqrt(np.pi) / 2),
        "empirical_E[R]/sqrt(E[R^2])_of_noise_magnitudes": float(Rn.mean() / np.sqrt(np.mean(Rn ** 2))),
        "corr(fhat_swap_magdonor,f)": pearson(fhat_swap, img),
        "corr(fhat_swap_magdonor,donor_image)": pearson(fhat_swap, cam),
        "corr(smoothed|swap|,smoothed_grad_f)": pearson(gauss_smooth_fft(np.abs(fhat_swap), sm), gauss_smooth_fft(gmag, sm)),
        "corr(fhat_flat,donor_image)": pearson(fhat_flat, cam),
        "kurtosis_swap": kurtosis(fhat_swap),
        "swap_reverse_corr_with_donor": pearson(fhat_swap_rev, cam), "swap_reverse_corr_with_testimage": pearson(fhat_swap_rev, img),
        # (b)
        "zerophase_max|Im|/max|Re|": float(np.abs(fhat_zp_c.imag).max() / np.abs(fhat_zp).max()),
        "zerophase_evenness_max|f(x)-f(-x)|/max": float(np.abs(fhat_zp - np.roll(np.flip(fhat_zp, (0, 1)), 1, (0, 1))).max() / np.abs(fhat_zp).max()),
        "zerophase_argmax_(r,c)": [int(i) for i in np.unravel_index(np.argmax(fhat_zp), fhat_zp.shape)],
        "zerophase_max_value": float(fhat_zp.max()), "zerophase_min_value": float(fhat_zp.min()),
        "zerophase_max_value_predicted_sum|F|/N^2": float(mag.sum() / N ** 2),
        "zerophase_corr_with_f": pearson(fhat_zp, img),
        "zerophase_fraction_of_energy_within_radius16_of_origin": None,
    }
    rr = np.minimum(np.arange(N), N - np.arange(N))
    d2 = rr[:, None] ** 2 + rr[None, :] ** 2
    q12["zerophase_fraction_of_energy_within_radius16_of_origin"] = float(np.sum(fhat_zp[d2 <= 256] ** 2) / np.sum(fhat_zp ** 2))
    q12["zerophase_fraction_of_|f|_energy_in_same_disc_for_test_image"] = float(np.sum(img[d2 <= 256] ** 2) / np.sum(img ** 2))
    R["q1_2"] = q12

    # ---------------- 1.3 padding / wraparound
    pad_rows = []
    for name, K in [("Laplacian 5x5", lap5), ("Sobel 3x3", sobel)]:
        M = K.shape[0]
        cen = (M // 2, M // 2)
        # full linear convolution (size N+M-1) by direct summation, independent of any FFT:
        # full[r,c] = sum_{m,n} K[m,n] img[r-m, c-n]
        full = np.zeros((N + M - 1, N + M - 1))
        for m in range(M):
            for n in range(M):
                full[m:m + N, n:n + N] += K[m, n] * img
        linear_same = full[cen[0]:cen[0] + N, cen[1]:cen[1] + N]
        for P in sorted({N, N + 1, N + M - 3, N + M - 2, N + M - 1, 540, 2 * N}):
            fp = np.zeros((P, P)); fp[:N, :N] = img
            kp = np.zeros((P, P)); kp[:M, :M] = K
            out = np.real(np.fft.ifft2(np.fft.fft2(fp) * np.fft.fft2(kp)))
            row = {"kernel": name, "P": P, "P>=N+M-1": bool(P >= N + M - 1), "memory_factor_(P/N)^2": (P / N) ** 2}
            if P >= N + M - 1:
                row["max_abs_err_vs_full_linear"] = float(np.abs(out[:N + M - 1, :N + M - 1] - full).max())
            else:
                # prediction: the full result folded modulo P (time-domain aliasing)
                idx = np.arange(N + M - 1) % P
                folded = np.zeros((P, P))
                np.add.at(folded, (idx[:, None], idx[None, :]), full)
                row["max_abs_err_vs_folded_full_(aliasing_prediction)"] = float(np.abs(out - folded).max())
                row["max_abs_err_vs_full_linear_on_P_x_P"] = float(np.abs(out - full[:P, :P]).max())
            ix = (cen[0] + np.arange(N)) % P   # wrap-around indexing: the crop of a P<N+cen array wraps
            row["max_abs_err_cropped_same_vs_linear"] = float(np.abs(out[np.ix_(ix, ix)] - linear_same).max())
            pad_rows.append(row)
    R["q1_3_padding_table"] = pad_rows
    # circular (unpadded, P=N, kernel centred) vs linear 'same' with zero boundary: where does it differ?
    border = {}
    for name, K in [("Laplacian 5x5", lap5), ("Sobel 3x3", sobel)]:
        M = K.shape[0]
        h = M // 2
        cen = (h, h)
        circ = circ_conv_direct(img, K, cen)
        padded = np.zeros((N + 2 * h, N + 2 * h))
        padded[h:-h, h:-h] = img
        lin = np.zeros((N, N))
        for m in range(M):
            for n in range(M):
                lin += K[m, n] * padded[h - (m - h):h - (m - h) + N, h - (n - h):h - (n - h) + N]
        e = np.abs(circ - lin)
        interior = e[h:-h, h:-h]
        band = e.copy()
        band[h:-h, h:-h] = 0
        border[name] = {"border_width_(M-1)/2": h, "max_err_interior": float(interior.max()), "max_err_border_band": float(band.max()),
                        "fraction_of_pixels_corrupted": float((e > 1e-9).mean()), "expected_fraction_1-(N-2h)^2/N^2": float(1 - ((N - 2 * h) / N) ** 2),
                        "max_|linear|": float(np.abs(lin).max())}
        if name.startswith("Lap"):
            circ_img, lin_img, err_img = circ, lin, e
    R["q1_3a_wraparound"] = border
    from scipy.fft import next_fast_len
    # timing and complexity
    time_rows = []
    for M in (3, 5, 9, 15, 31, 63):
        K = rng.random((M, M))
        reps = 3

        def med(fn):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        P1 = N + M - 1
        Pf = next_fast_len(P1, True)

        def fft_conv(P):
            fp = np.zeros((P, P))
            fp[:N, :N] = img
            kp = np.zeros((P, P))
            kp[:M, :M] = K
            return np.fft.irfft2(np.fft.rfft2(fp) * np.fft.rfft2(kp), s=(P, P))
        t_circ = med(lambda: np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(np.pad(K, ((0, N - M), (0, N - M))))))
        t_pad = med(lambda: fft_conv(P1))
        t_fast = med(lambda: fft_conv(Pf))
        t_sp = med(lambda: scipy.signal.fftconvolve(img, K, mode="full"))
        t_direct = med(lambda: circ_conv_direct(img, K, (0, 0))) if M <= 15 else None
        ref = scipy.signal.fftconvolve(img, K, mode="full")
        mine = fft_conv(Pf)[:N + M - 1, :N + M - 1]
        time_rows.append({"M": M, "N+M-1": P1, "scipy_style_next_fast_len": Pf,
                          "memory_factor_(P/N)^2_minimal": (P1 / N) ** 2, "memory_factor_(P/N)^2_fast_len": (Pf / N) ** 2,
                          "ratio_cost_model_P^2logP^2/N^2logN^2": float(P1 ** 2 * np.log(P1 ** 2) / (N ** 2 * np.log(N ** 2))),
                          "t_unpadded_circular_s": t_circ, "t_padded_minimal_s": t_pad, "t_padded_fast_len_s": t_fast,
                          "t_scipy_fftconvolve_full_s": t_sp, "t_direct_spatial_s": t_direct,
                          "max_abs_diff_vs_scipy_fftconvolve": float(np.abs(ref - mine).max())})
    R["q1_3b_cost"] = {"rows": time_rows, "P=2N_memory_factor": 4.0,
                       "P=2N_cost_ratio_model": float((2 * N) ** 2 * np.log((2 * N) ** 2) / (N ** 2 * np.log(N ** 2))),
                       "cpu_note": "timings are machine/BLAS dependent; the ratios are what is analysed"}

    # ---------------- figures
    # (1) test image, spectrum, phase
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("test image 512x512")
    ax[1].imshow(np.log1p(np.fft.fftshift(np.abs(F))), cmap="magma"); ax[1].set_title("log(1+|F|), centred (full, symmetric)")
    half = np.log1p(np.fft.fftshift(np.abs(F))).copy(); half[:, : N // 2] = np.nan
    ax[2].imshow(half, cmap="magma"); ax[2].set_title("non-redundant half plane (v>=0)")
    ax[3].imshow(np.fft.fftshift(np.angle(F)), cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[3].set_title("phase (odd symmetric)")
    for a in ax: a.axis("off")
    C.savefig(plt, fig, "q1_spectrum.png")
    # (2) sinusoid
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    ax[0].imshow(fs_img, cmap="gray"); ax[0].set_title("synthetic sum of 4 sinusoids"); ax[0].axis("off")
    sp = np.fft.fftshift(np.abs(Fs))
    zoom = slice(NS // 2 - 100, NS // 2 + 100)
    ax[1].imshow(np.log1p(sp[zoom, zoom]), cmap="magma", extent=[-100, 100, 100, -100])
    for (u, v, A, ph) in SINUSOIDS:
        for s in (1, -1):
            ax[1].plot(s * v, s * u, "c+", ms=9)
    ax[1].set_xlabel("v (cycles/image)"); ax[1].set_ylabel("u"); ax[1].set_title("spectrum (zoom); + = true (u,v) and (-u,-v)")
    C.savefig(plt, fig, "q1_sinusoid.png")
    # (3) phase / magnitude
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.2))
    show = [(img, "original"), (stretch(fhat_flat, 0.5, 99.5), "phase + flat magnitude"),
            (stretch(fhat_noise, 0.5, 99.5), "phase + white-noise magnitude"),
            (stretch(fhat_swap, 0.5, 99.5), "phase(test) + magnitude(other photo)"),
            (stretch(np.fft.fftshift(fhat_zp), 0, 99.7) ** 0.5, "magnitude + zero phase (fftshift, sqrt gamma)"),
            (stretch(np.fft.fftshift(np.log1p(mag)), 0, 100), "log(1+|F|) used in (b)")]
    for a, (im, t_) in zip(ax.ravel(), show):
        a.imshow(im, cmap="gray"); a.set_title(t_); a.axis("off")
    C.savefig(plt, fig, "q1_phase_magnitude.png")
    # (4) wrap-around
    g31x = np.arange(-15, 16)
    g31 = np.exp(-(g31x[:, None] ** 2 + g31x[None, :] ** 2) / (2 * 6.0 ** 2)); g31 /= g31.sum()
    circ_g = circ_conv_direct(img, g31, (15, 15))
    padg = np.zeros((N + 30, N + 30)); padg[15:-15, 15:-15] = img
    lin_g = np.zeros((N, N))
    for m in range(31):
        for n in range(31):
            lin_g += g31[m, n] * padg[15 - (m - 15):15 - (m - 15) + N, 15 - (n - 15):15 - (n - 15) + N]
    err_g = np.abs(circ_g - lin_g)
    R["q1_3a_wraparound"]["Gaussian 31x31 (sigma=6) low-pass"] = {
        "border_width_(M-1)/2": 15, "max_err_interior": float(err_g[15:-15, 15:-15].max()),
        "max_err_border_band": float(err_g.max()), "max_err_as_%_of_255": float(100 * err_g.max() / 255),
        "fraction_of_pixels_corrupted_(>1e-9)": float((err_g > 1e-9).mean()), "expected_fraction": float(1 - ((N - 30) / N) ** 2)}
    C.save_json(R, "q1.json")
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    ax[0, 0].imshow(circ_img, cmap="gray", vmin=-150, vmax=150); ax[0, 0].set_title("5x5 Laplacian: circular (unpadded)")
    ax[0, 1].imshow(lin_img, cmap="gray", vmin=-150, vmax=150); ax[0, 1].set_title("5x5 Laplacian: zero-padded linear")
    im_ = ax[0, 2].imshow(err_img[:24, :24], cmap="inferno", interpolation="nearest"); ax[0, 2].set_title("|circ - lin|, top-left 24x24 (px)")
    plt.colorbar(im_, ax=ax[0, 2], fraction=0.046)
    ax[0, 3].semilogy(np.arange(24), np.maximum(err_img[:24, 100], 1e-14), "o-", label="column 100 (down from top)"); ax[0, 3].semilogy(np.arange(24), np.maximum(err_img[100, :24], 1e-14), "s--", label="row 100 (from left)")
    ax[0, 3].set_title("error vs distance to border (log): zero beyond 2 px"); ax[0, 3].legend(fontsize=7); ax[0, 3].set_xlabel("pixels from border")
    ax[1, 0].imshow(circ_g, cmap="gray", vmin=0, vmax=255); ax[1, 0].set_title("31x31 Gaussian blur: circular")
    ax[1, 1].imshow(lin_g, cmap="gray", vmin=0, vmax=255); ax[1, 1].set_title("31x31 Gaussian blur: zero-padded linear")
    im2 = ax[1, 2].imshow(err_g, cmap="inferno"); ax[1, 2].set_title("|circ - lin| (15-px border band)"); plt.colorbar(im2, ax=ax[1, 2], fraction=0.046)
    ax[1, 3].plot(circ_g[:, 256][:60], label="circular"); ax[1, 3].plot(lin_g[:, 256][:60], "--", label="linear (zero outside)")
    ax[1, 3].set_title("column 256, first 60 rows"); ax[1, 3].legend(fontsize=7); ax[1, 3].set_xlabel("row")
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[1, 2]): a_.axis("off")
    C.savefig(plt, fig, "q1_wraparound.png")
    # (5) padding size sweep
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rows_l = [r for r in pad_rows if r["kernel"].startswith("Lap")]
    Ps = [r["P"] for r in rows_l]
    es = [r["max_abs_err_vs_full_linear"] if r["P>=N+M-1"] else r["max_abs_err_vs_full_linear_on_P_x_P"] for r in rows_l]
    ax.semilogy(Ps, np.maximum(es, 1e-16), "o-"); ax.axvline(N + 5 - 1, color="r", ls="--", label="N+M-1=516")
    ax.set_xlabel("FFT size P"); ax.set_ylabel("max |error| vs direct linear conv"); ax.legend()
    C.savefig(plt, fig, "q1_padding_sweep.png")
    # (6) resolution
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for a, d in zip(ax, (0.5, 1.0, 2.0)):
        r = np.arange(NS)
        col = np.cos(2 * np.pi * 50 * r / NS) + np.cos(2 * np.pi * (50 + d) * r / NS)
        spp = np.abs(np.fft.fft(col, L))[:L // 2]; fg = np.arange(L // 2) / L * NS
        w_ = (fg > 46) & (fg < 56)
        a.plot(fg[w_], spp[w_], label="zero-padded x16")
        spg = np.abs(np.fft.fft(col))[:NS // 2]
        a.stem(np.arange(NS // 2)[(np.arange(NS // 2) > 46) & (np.arange(NS // 2) < 56)], spg[(np.arange(NS // 2) > 46) & (np.arange(NS // 2) < 56)], linefmt="r-", markerfmt="ro", basefmt=" ", label="N-point DFT bins")
        a.set_title("separation %.1f bin(s), in phase" % d); a.set_xlabel("u (cycles/image)")
    ax[0].legend()
    C.savefig(plt, fig, "q1_resolution.png")

    C.save_json(R, "q1.json")
    print("q1 done")
    return R


if __name__ == "__main__":
    main()
