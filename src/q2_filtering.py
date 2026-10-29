"""Question 2 - frequency-domain filtering: ideal, Butterworth, Gaussian.

Pipeline (all of Q2): zero-pad the image into the top-left corner of a P x P array (P = 2*max(rows, cols)),
FFT, multiply by a real zero-phase H(u,v) given on the CENTRED P x P grid, inverse FFT, real part, crop.
D(u,v) = sqrt((u-P/2)^2 + (v-P/2)^2); D0 is always measured in padded-grid bins (cycles per P pixels).
scipy is used only for checks (gaussian_filter, sici, label) as the assignment allows.
Run: python src/q2_filtering.py -> results/q2.json, figures/q2_*.png
"""
import hashlib
import math
import os
import time

import numpy as np
import scipy.ndimage as ndi
from scipy.special import sici

import common as C

SEED = 2026
A_BG, B_FG = 50.0, 200.0
GAUSS_MATCH = 1.0 / np.sqrt(2.0 * np.log(2.0))      # D0' / D0 so that H_GLPF(D0) = 1/2


# ------------------------------------------------------------------ 2.1 filters
def radius_grid(P):
    u = np.arange(P) - P / 2.0
    return np.sqrt(u[:, None] ** 2 + u[None, :] ** 2)


def make_filter(kind, P, D0, n=None, highpass=False):
    """P x P transfer function.  kind in {'ideal','butterworth','gaussian'}.
    ideal: 1 for D<=D0; butterworth: 1/(1+(D/D0)^(2n)); gaussian: exp(-D^2/(2 D0^2)). High pass = 1 - low pass."""
    if D0 <= 0:
        raise ValueError("D0 must be positive")
    D = radius_grid(P)
    if kind == "ideal":
        H = (D <= D0).astype(float)
    elif kind == "butterworth":
        if n is None:
            raise ValueError("Butterworth needs the order n")
        H = 1.0 / (1.0 + (D / D0) ** (2 * n))
    elif kind == "gaussian":
        H = np.exp(-D ** 2 / (2.0 * D0 ** 2))
    else:
        raise ValueError(kind)
    return 1.0 - H if highpass else H


def pad_size(shape):
    return 2 * max(shape)


def filter_image(img, H, return_padded=False):
    """Zero-padded, centred-spectrum frequency-domain filtering; real part, cropped to img.shape."""
    P = H.shape[0]
    M, N = img.shape
    pad = np.zeros((P, P))
    pad[:M, :N] = img
    G = np.fft.fft2(pad) * np.fft.ifftshift(H)
    out = np.real(np.fft.ifft2(G))
    return (out[:M, :N], out) if return_padded else out[:M, :N]


def filter_spectrum(img, P):
    pad = np.zeros((P, P))
    pad[:img.shape[0], :img.shape[1]] = img
    return np.fft.fft2(pad)


def apply_H(F, H, shape):
    out = np.real(np.fft.ifft2(F * np.fft.ifftshift(H)))
    return out[:shape[0], :shape[1]]


def power_fraction_table(img, targets=(90, 95, 99, 99.5)):
    M, N = img.shape
    P = pad_size(img.shape)
    pw = np.abs(np.fft.fftshift(filter_spectrum(img, P))) ** 2
    D = radius_grid(P)
    total = pw.sum()
    out = {}
    order = np.argsort(D.ravel())
    cs = np.cumsum(pw.ravel()[order]) / total * 100
    Ds = D.ravel()[order]
    for t in targets:
        i = int(np.searchsorted(cs, t))
        # smallest INTEGER radius r with alpha(r) >= t
        r = int(np.ceil(Ds[i] - 1e-9))
        a_r = float(pw[D <= r].sum() / total * 100)
        out[str(t)] = {"radius_bins_(padded_P=%d)" % P: r, "alpha(r)": a_r,
                       "radius_in_cycles_per_pixel": r / P,
                       "radius_in_unpadded_N=%d_grid_bins" % M: r * M / P}
    dc = float(pw[P // 2, P // 2] / total * 100)
    return out, dc, P


def otsu(img):
    h, edges = np.histogram(img, bins=256)
    c = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(c[np.argmax(var)])


# ------------------------------------------------------------------ 2.2 ringing
def step_image(N=512, side=256):
    s = np.full((N, N), A_BG)
    lo = (N - side) // 2
    s[lo:lo + side, lo:lo + side] = B_FG
    return s, lo


def profile_stats(g, row, lo, side, label=""):
    """Overshoot per the assignment: OS = 100*(max g - B)/(B-A) on the profile row; first peak distance
    from the left edge (edge position lo-0.5) to the first local maximum inside the bright square."""
    p = g[row, :]
    os_ = 100 * (p.max() - B_FG) / (B_FG - A_BG)
    us_ = 100 * (A_BG - p[max(0, lo - 90):lo].min()) / (B_FG - A_BG)     # dark-side undershoot next to the left edge only
    x = np.arange(len(p))
    pk = None
    for i in range(lo, lo + side - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > B_FG + 0.05:
            pk = i
            break
    # peak position refined by parabolic interpolation (sub-pixel), reported separately
    sub = None
    if pk is not None:
        a_, b_, c_ = p[pk - 1], p[pk], p[pk + 1]
        den = a_ - 2 * b_ + c_
        sub = pk + 0.5 * (a_ - c_) / den if den != 0 else pk
    return {"overshoot_%": float(os_), "undershoot_%": float(us_),
            "first_peak_pixel": pk, "first_peak_distance_px": None if pk is None else float(pk - (lo - 0.5)),
            "first_peak_distance_subpixel_px": None if sub is None else float(sub - (lo - 0.5)),
            "first_peak_height_over_B_%": None if pk is None else float(100 * (p[pk] - B_FG) / (B_FG - A_BG))}


def sine_integral_prediction(P, D0_eff, lo, side, N=512):
    """Closed-form two-edge prediction g(x)=A+(B-A)[F(x-x1)-F(x-x2)], F(t)=1/2+Si(2 pi fc t)/pi."""
    fc = D0_eff / P
    x = np.arange(N)
    x1, x2 = lo - 0.5, lo + side - 0.5
    Fx = lambda t: 0.5 + sici(2 * np.pi * fc * t)[0] / np.pi
    g = A_BG + (B_FG - A_BG) * (Fx(x - x1) - Fx(x - x2))
    return g


def kernel_1d(H):
    """h(x) = F^-1{ H(u,0) } : the v=0 slice (row P/2 of the centred filter), inverse DFT."""
    P = H.shape[0]
    h1 = H[P // 2, :]                     # u runs along axis 1 here, v = 0
    return np.real(np.fft.ifft(np.fft.ifftshift(h1)))


def sigma_spatial(P, D0):
    """Spatial standard deviation (pixels) of the kernel of H=exp(-D^2/(2 D0^2)) on a P x P grid."""
    return P / (2 * np.pi * D0)


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "seed": SEED}
    imgs = {k: C.load_gray(os.path.join(C.PROVIDED, f)) for k, f in
            [("characters", "characters_test_pattern.tif"), ("text", "text_gaps.tif"),
             ("thumb", "thumb_print.tif"), ("cameraman", "cameraman.tif")]}
    R["images"] = {k: list(v.shape) for k, v in imgs.items()}

    # ================= 2.1a cutoff semantics
    P0 = 1024
    D0 = 40.0
    at_cutoff = {}
    for name, kw in [("ILPF", dict(kind="ideal")), ("BLPF n=1", dict(kind="butterworth", n=1)),
                     ("BLPF n=2", dict(kind="butterworth", n=2)), ("BLPF n=5", dict(kind="butterworth", n=5)),
                     ("GLPF (D0'=D0)", dict(kind="gaussian"))]:
        H = make_filter(P=P0, D0=D0, **kw)
        at_cutoff[name] = {"measured_H(D=D0)": float(H[P0 // 2, P0 // 2 + int(D0)]),
                           "analytic": {"ILPF": 1.0, "BLPF n=1": 0.5, "BLPF n=2": 0.5, "BLPF n=5": 0.5,
                                        "GLPF (D0'=D0)": float(np.exp(-0.5))}[name]}
    D0p = D0 * GAUSS_MATCH
    Hm = make_filter("gaussian", P0, D0p)
    # exact radius D=D0 sampled on the grid (integer offset 40 along an axis)
    at_cutoff["GLPF matched D0'=D0/sqrt(2 ln 2)"] = {"D0'": D0p, "measured_H(D=D0)": float(Hm[P0 // 2, P0 // 2 + int(D0)]), "analytic": 0.5}
    # a D that is exactly D0 along a diagonal (3-4-5 triangle): (24,32) has D=40
    at_cutoff["GLPF matched at (24,32) i.e. D=40 exactly (diagonal sample)"] = float(Hm[P0 // 2 + 24, P0 // 2 + 32])
    R["q2_1a"] = {"D0": D0, "P": P0, "H_at_D=D0": at_cutoff, "D0_prime_over_D0": GAUSS_MATCH,
                  "D0_prime_for_D0=40": D0p, "sigma_ratio_formula": "D0' = D0/sqrt(2 ln 2) = 0.8493 D0"}
    # radial profiles
    r = np.arange(0, 121)
    prof = {}
    plt_data = {}
    for nm, kw in [("ILPF", dict(kind="ideal")), ("BLPF n=2", dict(kind="butterworth", n=2)),
                   ("GLPF matched", dict(kind="gaussian"))]:
        d0 = D0p if nm.startswith("GLPF") else D0
        H = make_filter(P=P0, D0=d0, **kw)
        Hhp = make_filter(P=P0, D0=d0, highpass=True, **kw)
        plt_data[nm] = (H[P0 // 2, P0 // 2:P0 // 2 + 121], Hhp[P0 // 2, P0 // 2:P0 // 2 + 121])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for nm, (lp, hp) in plt_data.items():
        ax[0].plot(r, lp, label=nm + " low-pass")
        ax[0].plot(r, hp, "--", label=nm + " high-pass (1-LP)")
    ax[0].axvline(D0, color="k", lw=0.5); ax[0].axhline(0.5, color="k", lw=0.5)
    ax[0].set_xlabel("D (bins of the P=1024 grid)"); ax[0].set_ylabel("H(D)"); ax[0].set_title("six transfer functions, D0=40 (GLPF: D0'=33.97)"); ax[0].legend(fontsize=6, ncol=2)
    for n_ in (1, 2, 5, 10):
        ax[1].plot(r, make_filter("butterworth", P0, D0, n=n_)[P0 // 2, P0 // 2:P0 // 2 + 121], label="n=%d" % n_)
    ax[1].set_title("Butterworth order family, D0=40"); ax[1].legend(); ax[1].set_xlabel("D")
    C.savefig(plt, fig, "q2_radial_profiles.png")

    # ================= 2.1b energy cutoff
    pf, dc, Pc = power_fraction_table(imgs["characters"])
    # mean-removed variant (AC power only)
    ac = imgs["characters"] - imgs["characters"].mean()
    pf_ac, _, _ = power_fraction_table(ac)
    R["q2_1b"] = {"image": "characters_test_pattern.tif", "P": Pc, "table": pf, "DC_share_%": dc,
                  "table_mean_removed_(AC_only)": pf_ac, "note": "alpha(r) computed on the zero-padded P x P DFT used by the pipeline (radii in padded bins)"}
    radii = [pf[str(t)]["radius_bins_(padded_P=%d)" % Pc] for t in (90, 95, 99, 99.5)]

    # ================= 2.1c inverse design
    # H(10)=16/17 -> (10/D0)^(2n) = 1/16 ; H(40)=1/17 -> (40/D0)^(2n)=16 ; ratio -> 4^(2n)=256 -> n=2, D0=20
    n_sol = np.log(256) / (2 * np.log(4))
    D0_sol = 10 / (1 / 16) ** (1 / (2 * n_sol))
    Hs = make_filter("butterworth", 1024, D0_sol, n=int(round(n_sol)))
    v10 = float(Hs[512, 512 + 10]); v40 = float(Hs[512, 512 + 40])
    ns = {}
    for n_ in range(1, 7):
        h2 = make_filter("butterworth", 1024, D0_sol, n=n_)
        ns[n_] = float(h2[512, 512 + int(2 * D0_sol)])
    n_min = min(k for k, v in ns.items() if v <= 0.01)
    R["q2_1c"] = {"n_solution": float(n_sol), "D0_solution": float(D0_sol), "H(10)_measured": v10, "target_16/17": 16 / 17,
                  "H(40)_measured": v40, "target_1/17": 1 / 17, "H(2D0)_vs_n": ns,
                  "smallest_n_for_H(2D0)<=1%": n_min, "analytic_H(2D0)=1/(1+4^n)": {k: 1 / (1 + 4.0 ** k) for k in range(1, 7)}}

    # ================= 2.2 ringing
    pred_path = os.path.join(C.PREDICTIONS, "q2_2_prediction.md")
    R["q2_2_prediction_file"] = {"path": "predictions/q2_2_prediction.md",
                                 "sha256": hashlib.sha256(open(pred_path, "rb").read()).hexdigest(),
                                 "file_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(pred_path))),
                                 "experiment_run_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    s_img, lo = step_image()
    side = 256
    P = 1024
    row = 256
    Fs = filter_spectrum(s_img, P)
    rows_a = []
    prof_store = {}
    for D0_ in (10, 20, 40, 80):
        g = apply_H(Fs, make_filter("ideal", P, D0_), s_img.shape)
        st = profile_stats(g, row, lo, side)
        gp = sine_integral_prediction(P, D0_, lo, side)
        gpd = sine_integral_prediction(P, D0_ + 0.5, lo, side)
        pred_os = 100 * (0.5 + sici(np.pi)[0] / np.pi - 1)
        # one-edge closed form is independent of D0
        rows_a.append({"D0": D0_, "measured": st,
                       "pred_overshoot_%_single_edge": float(pred_os),
                       "pred_peak_distance_px_continuous_P/(2D0)": P / (2 * D0_),
                       "pred_peak_distance_px_discrete_P/(2D0+1)": P / (2 * D0_ + 1),
                       "two_edge_Si_formula_overshoot_%_(fc=D0/P)": float(100 * (gp.max() - B_FG) / (B_FG - A_BG)),
                       "two_edge_Si_formula_overshoot_%_(fc=(D0+.5)/P)": float(100 * (gpd.max() - B_FG) / (B_FG - A_BG)),
                       "max|measured-Si_formula(fc_eff)|_over_row_(intensity)": float(np.abs(g[row] - gpd).max()),
                       "meas_minus_pred_overshoot_pp": st["overshoot_%"] - pred_os,
                       "meas_minus_pred_distance_px_(vs_continuous)": None if st["first_peak_distance_px"] is None else st["first_peak_distance_px"] - P / (2 * D0_)})
        prof_store[D0_] = (g[row].copy(), gp, gpd)
    # controlled reconciliation experiments (same ILPF, same row): remove one suspected cause at a time
    def pad_with(img, P, val):
        a = np.full((P, P), float(val)); a[:img.shape[0], :img.shape[1]] = img; return a
    strip = np.tile(s_img[row:row + 1, :], (512, 1))        # every row equal -> exactly x-only image
    recon = []
    for D0_ in (10, 20, 40, 80):
        Hh = make_filter("ideal", P, D0_)
        g_zero = apply_H(Fs, Hh, s_img.shape)
        g_bg = np.real(np.fft.ifft2(np.fft.fft2(pad_with(s_img, P, A_BG)) * np.fft.ifftshift(Hh)))[:512, :512]
        g_strip = np.real(np.fft.ifft2(np.fft.fft2(pad_with(strip, P, A_BG)) * np.fft.ifftshift(Hh)))[:512, :512]
        gpd = sine_integral_prediction(P, D0_ + 0.5, lo, side)
        recon.append({"D0": D0_,
                      "overshoot_%_zero_padded_2D_(assignment_pipeline)": profile_stats(g_zero, row, lo, side)["overshoot_%"],
                      "overshoot_%_padded_with_background_A_(no_50->0_border_edge)": profile_stats(g_bg, row, lo, side)["overshoot_%"],
                      "overshoot_%_x-only_strip_(1-D_exact)": profile_stats(g_strip, row, lo, side)["overshoot_%"],
                      "two_edge_Si_formula_%_(fc=(D0+.5)/P)": float(100 * (gpd.max() - B_FG) / (B_FG - A_BG)),
                      "max|strip-Si_formula|_intensity": float(np.abs(g_strip[row] - gpd).max()),
                      "peak_distance_px_strip": profile_stats(g_strip, row, lo, side)["first_peak_distance_px"]})
    gau_rows = []
    for D0_ in (10, 20, 40, 80):
        d0p = D0_ * GAUSS_MATCH
        g = apply_H(Fs, make_filter("gaussian", P, d0p), s_img.shape)
        gau_rows.append({"D0": D0_, "D0_prime": d0p, "sigma_s_px": sigma_spatial(P, d0p),
                         "measured_overshoot_%_(raw,negative=max_below_B)": float(100 * (g[row].max() - B_FG) / (B_FG - A_BG)),
                         "max_g_row": float(g[row].max()), "min_g_row": float(g[row].min()),
                         "max_g_whole_image": float(g.max()), "min_g_whole_image": float(g.min()),
                         "predicted_overshoot_%": 0.0})
    # corner / near-corner profile
    corner_rows = []
    for D0_ in (20, 40):
        g = apply_H(Fs, make_filter("ideal", P, D0_), s_img.shape)
        for off in (2, 6, 16, 64, 128):
            rr = lo + off
            st = profile_stats(g, rr, lo, side)
            # reference: the 1-D prediction restricted to x (with the same profile)
            corner_rows.append({"D0": D0_, "row_offset_from_top_edge_px": off, "overshoot_%": st["overshoot_%"],
                                "first_peak_distance_px": st["first_peak_distance_px"],
                                "pred_1D_overshoot_%": float(100 * (0.5 + sici(np.pi)[0] / np.pi - 1)), "pred_1D_distance": P / (2 * D0_)})
    R["q2_2a"] = {"reconciliation": recon, "ideal": rows_a, "gaussian": gau_rows, "near_corner": corner_rows, "P": P, "row": row, "edge_position_x": lo - 0.5}
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    for a, D0_ in zip(ax.ravel(), (10, 20, 40, 80)):
        g, gp, gpd = prof_store[D0_]
        a.plot(g, label="measured (row 256)"); a.plot(gpd, "--", label="Si closed form (fc=(D0+1/2)/P)")
        a.axhline(B_FG * 1.0, color="gray", lw=0.5); a.axhline(B_FG + 0.0895 * 150, color="r", lw=0.5, ls=":")
        a.set_xlim(lo - 80, lo + 120); a.set_title("ILPF D0=%d" % D0_); a.legend(fontsize=7)
    C.savefig(plt, fig, "q2_ringing_profiles.png")
    # profile figure with corner rows
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    g40 = apply_H(Fs, make_filter("ideal", P, 40), s_img.shape)
    for off in (2, 6, 16, 128):
        ax[0].plot(g40[lo + off, lo - 40:lo + 60], label="row %d px below top edge" % off)
    ax[0].set_title("ILPF D0=40: profiles near the corner"); ax[0].legend(fontsize=7)
    ax[1].imshow(g40, cmap="gray"); ax[1].set_title("ILPF D0=40 output (profiles taken at x-range around left edge)")
    C.savefig(plt, fig, "q2_corner_profiles.png")

    # ---- 2.2b Butterworth order sweep
    rows_b = []
    for n_ in (1, 2, 3, 5, 10, 20):
        H = make_filter("butterworth", P, 40, n=n_)
        g = apply_H(Fs, H, s_img.shape)
        st = profile_stats(g, row, lo, side)
        h = kernel_1d(H)
        rows_b.append({"n": n_, "overshoot_%": st["overshoot_%"], "undershoot_%": st["undershoot_%"],
                       "first_peak_distance_px": st["first_peak_distance_px"],
                       "kernel_sum": float(h.sum()), "max_h": float(h.max()), "min_h": float(h.min()),
                       "|min h|/max h": float(abs(h.min()) / h.max()),
                       "kernel_nonneg_(min>=-1e-12)": bool(h.min() >= -1e-12)})
    hI = kernel_1d(make_filter("ideal", P, 40)); hG = kernel_1d(make_filter("gaussian", P, 40 * GAUSS_MATCH))
    R["q2_2b"] = {"D0": 40, "rows": rows_b, "ideal_limit_pred_%": float(100 * (0.5 + sici(np.pi)[0] / np.pi - 1)),
                  "ILPF_kernel_ratio": float(abs(hI.min()) / hI.max()), "GLPF_kernel_ratio": float(abs(hG.min()) / hG.max()),
                  "ILPF_measured_overshoot_D0=40_%": [x for x in rows_a if x["D0"] == 40][0]["measured"]["overshoot_%"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    xs = np.arange(-40, 41)
    for n_ in (1, 2, 5, 20):
        h = np.fft.fftshift(kernel_1d(make_filter("butterworth", P, 40, n=n_)))
        ax[0].plot(xs, h[P // 2 - 40:P // 2 + 41], label="n=%d" % n_)
    ax[0].set_title("1-D kernels h(x)=F^-1{H(u,0)}, D0=40"); ax[0].set_xlabel("x (px)"); ax[0].legend()
    for n_ in (1, 2, 5, 20):
        g = apply_H(Fs, make_filter("butterworth", P, 40, n=n_), s_img.shape)
        ax[1].plot(g[row, lo - 40:lo + 60], label="n=%d" % n_)
    ax[1].set_title("step response around the left edge"); ax[1].legend()
    C.savefig(plt, fig, "q2_butterworth_kernels.png")

    # ---- 2.2c visual grid
    ch = imgs["characters"]
    Pch = pad_size(ch.shape)
    Fch = filter_spectrum(ch, Pch)
    outs = {}
    kinds = [("ILPF", dict(kind="ideal"), 1.0), ("BLPF n=2", dict(kind="butterworth", n=2), 1.0),
             ("GLPF D0'", dict(kind="gaussian"), GAUSS_MATCH)]
    for nm, kw, sc in kinds:
        for rad in radii:
            outs[(nm, rad)] = apply_H(Fch, make_filter(P=Pch, D0=rad * sc, **kw), ch.shape)
    # automatic ringing locations: where ILPF(45) departs from the monotone (Butterworth-1 free) smooth image most
    ref_smooth = apply_H(Fch, make_filter("gaussian", Pch, radii[1] * GAUSS_MATCH), ch.shape)
    dev = np.abs(outs[("ILPF", radii[1])] - ref_smooth)
    dev[:40, :] = 0; dev[-40:, :] = 0; dev[:, :40] = 0; dev[:, -40:] = 0
    dev_s = ndi.uniform_filter(dev, 15)
    locs = []
    d2 = dev_s.copy()
    for _ in range(2):
        i = np.unravel_index(np.argmax(d2), d2.shape)
        locs.append((int(i[0]), int(i[1])))
        d2[max(0, i[0] - 120):i[0] + 120, max(0, i[1] - 120):i[1] + 120] = 0
    fig, ax = plt.subplots(3, 4, figsize=(14, 10.6))
    for i, (nm, kw, sc) in enumerate(kinds):
        for j, rad in enumerate(radii):
            a = ax[i, j]
            a.imshow(outs[(nm, rad)], cmap="gray", vmin=0, vmax=255)
            a.set_title("%s, radius %d (alpha=%s%%)" % (nm, rad, ("90", "95", "99", "99.5")[j]), fontsize=8)
            a.axis("off")
            if i == 0:
                for k, (ry, rx) in enumerate(locs):
                    a.annotate("", xy=(rx, ry), xytext=(rx + 70, ry + 70), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
                    a.text(rx + 72, ry + 78, "R%d" % (k + 1), color="red", fontsize=7)
    C.savefig(plt, fig, "q2_visual_grid.png")
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
    for k, (ry, rx) in enumerate(locs):
        ys, xs_ = slice(max(0, ry - 45), ry + 45), slice(max(0, rx - 45), rx + 45)
        ax[k, 0].imshow(ch[ys, xs_], cmap="gray", vmin=0, vmax=255); ax[k, 0].set_title("original, R%d" % (k + 1))
        for j, (nm, kw, sc) in enumerate(kinds):
            ax[k, j + 1].imshow(outs[(nm, radii[1])][ys, xs_], cmap="gray", vmin=0, vmax=255); ax[k, j + 1].set_title("%s r=%d" % (nm, radii[1]))
    for a in ax.ravel(): a.axis("off")
    C.savefig(plt, fig, "q2_ringing_insets.png")
    R["q2_2c"] = {"radii": radii, "ringing_locations_(row,col)": locs,
                  "ILPF_ringing_metric_rms(ILPF-GLPF_at_r45)": float(np.sqrt(np.mean((outs[("ILPF", radii[1])] - ref_smooth) ** 2))),
                  "metrics_rms_error_vs_original": {"%s r=%d" % (nm, rad): float(np.sqrt(np.mean((outs[(nm, rad)] - ch) ** 2))) for nm, _, _ in kinds for rad in radii},
                  "min_max_of_outputs_(ringing_pushes_beyond_[0,255])": {"%s r=%d" % (nm, rad): [float(outs[(nm, rad)].min()), float(outs[(nm, rad)].max())] for nm, _, _ in kinds for rad in radii}}

    # ---- 2.2d Gaussian equivalence
    cam = imgs["cameraman"]
    Pc2 = pad_size(cam.shape)
    Fcam = filter_spectrum(cam, Pc2)
    pad_cam = np.zeros((Pc2, Pc2)); pad_cam[:512, :512] = cam
    eq_rows = []
    for D0_ in (10, 40, 160):
        sg = sigma_spatial(Pc2, D0_)
        gf = apply_H(Fcam, make_filter("gaussian", Pc2, D0_), cam.shape)
        gs4 = ndi.gaussian_filter(cam, sg, mode="constant", cval=0.0)                   # truncate=4.0 (default)
        gs8 = ndi.gaussian_filter(cam, sg, mode="constant", cval=0.0, truncate=8.0)
        gwrap = ndi.gaussian_filter(pad_cam, sg, mode="wrap", truncate=8.0)[:512, :512]   # periodic on the padded grid
        nyq = float(np.exp(-(Pc2 / 2) ** 2 / (2 * D0_ ** 2)))
        eq_rows.append({"D0": D0_, "sigma_s_px": sg, "H_at_Nyquist_exp(-(P/2)^2/(2D0^2))": nyq,
                        "max_abs_err_truncate4_zero_boundary": float(np.abs(gf - gs4).max()),
                        "max_abs_err_truncate8_zero_boundary": float(np.abs(gf - gs8).max()),
                        "max_abs_err_truncate8_periodic_padded": float(np.abs(gf - gwrap).max()),
                        "kernel_truncation_tail_mass_4sigma_2*Q(4)_1D": float(math.erfc(4 / math.sqrt(2)))})
    R["q2_2d"] = {"P": Pc2, "rows": eq_rows, "formula": "sigma_s = P/(2 pi D0)"}

    # ================= 2.3a noise-blur tradeoff
    rng = np.random.default_rng(SEED)
    noisy = cam + rng.normal(0, 20.0, cam.shape)
    Fn = filter_spectrum(noisy, Pc2)
    D0s = np.concatenate([np.arange(5, 151), np.arange(152, 301, 2)])   # spec range 5..150, then extended to find the true optimum
    curves = {"ILPF": [], "BLPF n=2": [], "GLPF": []}
    for d in D0s:
        for nm, kw, sc in kinds:
            o = apply_H(Fn, make_filter(P=Pc2, D0=d * sc, **kw), cam.shape)
            key = "GLPF" if nm.startswith("G") else nm
            curves[key].append(C.psnr(cam, o))
    opt = {}
    opt_spec = {}
    for k, v in curves.items():
        i = int(np.argmax(v))
        opt[k] = {"best_D0": int(D0s[i]), "peak_PSNR_dB": float(v[i])}
        j = int(np.argmax(np.array(v)[D0s <= 150]))
        opt_spec[k] = {"best_D0_within_5..150": int(D0s[j]), "PSNR_dB": float(v[j]), "at_range_edge": bool(D0s[j] == 150)}
    best_imgs = {}
    for nm, kw, sc in kinds:
        key = "GLPF" if nm.startswith("G") else nm
        best_imgs[key] = apply_H(Fn, make_filter(P=Pc2, D0=opt[key]["best_D0"] * sc, **kw), cam.shape)
    for k in opt:
        opt[k]["SSIM"] = C.ssim(cam, best_imgs[k])
        opt[k]["sigma_s_px_equiv"] = None
    R["q2_3a"] = {"noise_sigma": 20.0, "seed": SEED, "noisy_not_clipped": True, "PSNR_noisy_clipped_for_metric": C.psnr(cam, noisy),
                  "SSIM_noisy": C.ssim(cam, np.clip(noisy, 0, 255)), "optimum_extended_sweep": opt, "optimum_within_spec_range_5_150": opt_spec, "D0_range": [5, 300],
                  "curves": {k: [float(x) for x in v] for k, v in curves.items()}, "D0_values": D0s,
                  "PSNR_at_D0": {k: {int(d): float(curves[k][int(d) - 5]) for d in (10, 20, 30, 40, 60, 80, 100, 150)} for k in curves}}
    opt = R["q2_3a"]["optimum_extended_sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, v in curves.items():
        ax[0].plot(D0s, v, label=k); ax[0].plot(opt[k]["best_D0"], opt[k]["peak_PSNR_dB"], "ko")
    ax[0].axhline(R["q2_3a"]["PSNR_noisy_clipped_for_metric"], color="gray", ls=":", label="noisy input")
    ax[0].set_xlabel("D0 (half-amplitude radius, P=1024 bins)"); ax[0].set_ylabel("PSNR (dB)"); ax[0].legend()
    ax[0].set_title("cameraman, noise sigma=20 (seed %d)" % SEED)
    ax[1].imshow(np.hstack([np.clip(noisy, 0, 255), best_imgs["ILPF"], best_imgs["BLPF n=2"], best_imgs["GLPF"]])[100:300, :], cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("noisy | ILPF | BLPF | GLPF (rows 100-300), each at its optimum D0"); ax[1].axis("off")
    C.savefig(plt, fig, "q2_psnr_sweep.png")

    # ================= 2.3b character repair
    tx = imgs["text"]
    Pt = pad_size(tx.shape)
    Ft = filter_spectrum(tx, Pt)
    cand = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160]
    comp_rows = []
    outs_t = {}
    for d in cand:
        o = apply_H(Ft, make_filter("gaussian", Pt, d), tx.shape)
        outs_t[d] = o
        thr = 127.5
        lab, nc = ndi.label(o < thr, structure=np.ones((3, 3)))
        comp_rows.append({"D0_gauss": d, "components_of_dark_pixels_(8-conn)": int(nc), "dark_area_fraction_%": float(100 * (o < thr).mean()),
                          "min_value": float(o.min()), "contrast_range": float(o.max() - o.min())})
    lab0, nc0 = ndi.label(tx < 127.5, structure=np.ones((3, 3)))
    R["q2_3b_sweep"] = {"original_components": int(nc0), "original_dark_area_%": float(100 * (tx < 127.5).mean()), "rows": comp_rows, "P": Pt}
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.6))
    ax[0, 0].imshow(tx, cmap="gray"); ax[0, 0].set_title("original")
    for a, d in zip(ax.ravel()[1:], (20, 40, 60, 80, 100, 120, 160)):
        a.imshow(outs_t[d], cmap="gray", vmin=0, vmax=255); a.set_title("GLPF D0=%d" % d)
    for a in ax.ravel(): a.axis("off")
    C.savefig(plt, fig, "q2_text_sweep.png")
    # candidate display of crops later in main (choice made after inspecting sweep)
    json_partial = R

    # ================= 2.3c high-pass
    th = imgs["thumb"]
    Pth = pad_size(th.shape)
    Fth = filter_spectrum(th, Pth)
    D0hp = 60
    hp_out = {}
    hp_rows = []
    for nm, kw, sc in [("IHPF", dict(kind="ideal"), 1.0), ("BHPF n=4", dict(kind="butterworth", n=4), 1.0), ("GHPF", dict(kind="gaussian"), GAUSS_MATCH)]:
        H = make_filter(P=Pth, D0=D0hp * sc, highpass=True, **kw)
        o_crop, o_full = filter_image(th, H, return_padded=True)
        hp_out[nm] = o_crop
        hp_rows.append({"filter": nm, "H_HP(P/2,P/2)": float(H[Pth // 2, Pth // 2]), "mean_cropped_output": float(o_crop.mean()),
                        "mean_over_full_padded_PxP_output": float(o_full.mean()), "input_mean": float(th.mean()),
                        "std_output": float(o_crop.std()), "min": float(o_crop.min()), "max": float(o_crop.max())})
    k1, k2 = 1.0, 0.5
    Hhfe = k1 + k2 * make_filter("gaussian", Pth, D0hp * GAUSS_MATCH, highpass=True)
    hfe = filter_image(th, Hhfe)
    def local_std(x, w=15):
        m = ndi.uniform_filter(x, w); return float(np.sqrt(np.maximum(ndi.uniform_filter(x * x, w) - m * m, 0)).mean())
    hfe_info = {"k1": k1, "k2": k2, "H_HFE(DC)": float(Hhfe[Pth // 2, Pth // 2]), "mean_out": float(hfe.mean()), "mean_in": float(th.mean()),
                "mean_ratio_out/in": float(hfe.mean() / th.mean()), "local_std_15x15_in": local_std(th), "local_std_15x15_out": local_std(hfe),
                "ridge_contrast_gain": local_std(hfe) / local_std(th), "min": float(hfe.min()), "max": float(hfe.max()),
                "fraction_clipped_0_255": float(((hfe < 0) | (hfe > 255)).mean())}
    bin_rows = []
    bins = {}
    for nm, o in hp_out.items():
        t = otsu(o)
        b = o > t
        lab, nc = ndi.label(b, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        bins[nm] = b
        bin_rows.append({"filter": nm, "otsu_threshold": t, "foreground_%": float(100 * b.mean()), "components": int(nc),
                         "tiny_components_<20px": int((sizes < 20).sum()), "tiny_per_10k_px": float((sizes < 20).sum() / b.size * 1e4),
                         "largest_component_%_of_foreground": float(100 * sizes.max() / sizes.sum())})
    R["q2_3c"] = {"D0_common_half_amplitude_radius": D0hp, "P": Pth, "highpass": hp_rows, "hfe": hfe_info, "binary_ridge_maps": bin_rows}
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    ax[0, 0].imshow(th, cmap="gray"); ax[0, 0].set_title("thumb_print original")
    for a, (nm, o) in zip(ax[0, 1:], hp_out.items()):
        a.imshow(stretch_(o), cmap="gray"); a.set_title("%s output (mean %.2f), display-stretched" % (nm, o.mean()), fontsize=8)
    ax[1, 0].imshow(np.clip(hfe, 0, 255), cmap="gray", vmin=0, vmax=255); ax[1, 0].set_title("HFE: k1=1, k2=0.5 (Gaussian HP)")
    for a, (nm, b) in zip(ax[1, 1:], bins.items()):
        a.imshow(b, cmap="gray"); a.set_title("%s -> Otsu binary ridge map" % nm, fontsize=8)
    for a in ax.ravel(): a.axis("off")
    C.savefig(plt, fig, "q2_highpass.png")
    # crops for visual quality
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    cy, cx = 400, 400
    for a, (nm, b) in zip(ax[:3], bins.items()):
        a.imshow(b[cy:cy + 200, cx:cx + 200], cmap="gray"); a.set_title(nm + " binary (200x200 crop)")
    ax[3].imshow(np.clip(hfe, 0, 255)[cy:cy + 200, cx:cx + 200], cmap="gray", vmin=0, vmax=255); ax[3].set_title("HFE crop")
    for a in ax: a.axis("off")
    C.savefig(plt, fig, "q2_highpass_crops.png")

    C.save_json(R, "q2.json")
    # return objects for the later text-repair choice
    return R, outs_t, tx, Pt


def stretch_(x, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-30), 0, 1)


def finish_text_repair(R, outs_t, tx, Pt, d_choice):
    plt = C.fig_style()
    Ft = filter_spectrum(tx, Pt)
    gl = outs_t[d_choice] if d_choice in outs_t else apply_H(Ft, make_filter("gaussian", Pt, d_choice), tx.shape)
    d_half = d_choice * np.sqrt(2 * np.log(2))     # half-amplitude radius of the Gaussian
    il = apply_H(Ft, make_filter("ideal", Pt, d_half), tx.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(tx, cmap="gray"); ax[0].set_title("original text_gaps")
    ax[1].imshow(gl, cmap="gray", vmin=0, vmax=255); ax[1].set_title("GLPF D0'=%g (half-amplitude radius %.1f)" % (d_choice, d_half))
    ax[2].imshow(il, cmap="gray", vmin=0, vmax=255); ax[2].set_title("ILPF at the same half-amplitude radius %.1f" % d_half)
    for a in ax: a.axis("off")
    C.savefig(plt, fig, "q2_text_repair.png")
    # zoomed crops (also for the report)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ys, xs = slice(0, 120), slice(0, 200)
    ax[0].imshow(tx[ys, xs], cmap="gray", vmin=0, vmax=255); ax[0].set_title("original (zoom)")
    ax[1].imshow(gl[ys, xs], cmap="gray", vmin=0, vmax=255); ax[1].set_title("GLPF (zoom)")
    ax[2].imshow(il[ys, xs], cmap="gray", vmin=0, vmax=255); ax[2].set_title("ILPF (zoom)")
    for a in ax: a.axis("off")
    C.savefig(plt, fig, "q2_text_repair_zoom.png")
    R["q2_3b"] = {"chosen_D0_gauss": d_choice, "half_amplitude_radius": float(d_half),
                  "components_original": R["q2_3b_sweep"]["original_components"],
                  "components_gaussian": int(ndi.label(gl < 127.5, structure=np.ones((3, 3)))[1]),
                  "components_ideal_same_half_radius": int(ndi.label(il < 127.5, structure=np.ones((3, 3)))[1]),
                  "ideal_out_range": [float(il.min()), float(il.max())], "gaussian_out_range": [float(gl.min()), float(gl.max())],
                  "ideal_ringing_rms_vs_gaussian": float(np.sqrt(np.mean((il - gl) ** 2)))}
    C.save_json(R, "q2.json")


TEXT_REPAIR_D0 = 80     # chosen after inspecting figures/q2_text_sweep.png and the component-count sweep (see report)

if __name__ == "__main__":
    R, outs_t, tx, Pt = main()
    finish_text_repair(R, outs_t, tx, Pt, TEXT_REPAIR_D0)
    print("q2 done")
