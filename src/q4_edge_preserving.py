"""Question 4 - edge-preserving smoothing: bilateral vs Kuwahara (classic and soft), complexity and breakdown.

All filters are written with NumPy array arithmetic only (shifted-slice sums, integral images).
Gray images, intensity range 0..255, noisy input is NOT clipped before filtering.
Run: python src/q4_edge_preserving.py -> results/q4.json, figures/q4_*.png
"""
import math
import os
import time

import numpy as np

import common as C

SEED = 2026
SIGMA_MAIN = 15.0
MAX_RADIUS = 12


# ------------------------------------------------------------------ filters
def bilateral(img, sigma_s, sigma_r, radius=None):
    """O_p = sum_q ws(|p-q|) wr(I_q-I_p) I_q / sum_q ws wr, ws=exp(-d^2/2ss^2), wr=exp(-dI^2/2sr^2).
    Window (2r+1)^2, r=min(ceil(3 sigma_s), MAX_RADIUS=12), reflect boundary (the cap only matters for sigma_s > 4)."""
    r = min(int(math.ceil(3 * sigma_s)), MAX_RADIUS) if radius is None else radius
    H, W = img.shape
    pad = np.pad(img, r, mode="reflect")
    num = np.zeros_like(img, dtype=float)
    den = np.zeros_like(img, dtype=float)
    inv = -1.0 / (2.0 * sigma_r ** 2)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s ** 2))
            q = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            w = ws * np.exp((q - img) ** 2 * inv)
            num += w * q
            den += w
    return num / den


def _box_stats(pad, m):
    """mean and population variance over every m x m window of `pad` ('valid' positions), by integral images."""
    ii = np.pad(pad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(pad * pad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[m:, m:] - ii[:-m, m:] - ii[m:, :-m] + ii[:-m, :-m]
    s2 = ii2[m:, m:] - ii2[:-m, m:] - ii2[m:, :-m] + ii2[:-m, :-m]
    mean = s / (m * m)
    return mean, np.maximum(s2 / (m * m) - mean * mean, 0.0)


def _quadrants(img, N):
    """Means and variances of the four overlapping (k+1)x(k+1) sub-windows (k=(N-1)/2) sharing the centre
    pixel, ordered TL, TR, BL, BR.  Returns arrays of shape (4,H,W)."""
    if N % 2 == 0 or N < 3:
        raise ValueError("N must be odd and >= 3")
    k = (N - 1) // 2
    m = k + 1
    H, W = img.shape
    pad = np.pad(img.astype(float), k, mode="reflect")
    M, V = _box_stats(pad, m)
    sl = [(0, 0), (0, k), (k, 0), (k, k)]
    means = np.stack([M[i:i + H, j:j + W] for i, j in sl])
    vars_ = np.stack([V[i:i + H, j:j + W] for i, j in sl])
    return means, vars_


def kuwahara(img, N):
    """Classical Kuwahara: mean of the sub-window with minimal variance (argmin; ties -> first)."""
    means, vars_ = _quadrants(img, N)
    idx = np.argmin(vars_, axis=0)
    return np.take_along_axis(means, idx[None], axis=0)[0]


def soft_kuwahara(img, N, tau):
    """Soft-Kuwahara: softmax weights over the four sub-windows, w_i ∝ exp(-(v_i - v_min)/tau).
    tau -> 0 recovers the classical argmin; tau -> inf averages the four means."""
    means, vars_ = _quadrants(img, N)
    w = np.exp(-(vars_ - vars_.min(axis=0, keepdims=True)) / tau)
    return (w * means).sum(0) / w.sum(0)


def noise_sigma_immerkaer(img):
    """Immerkaer (1996) fast noise-variance estimate: convolution with the 3x3 mask [1 -2 1; -2 4 -2; 1 -2 1]."""
    H, W = img.shape
    c = lambda dy, dx: img[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
    L = (c(-1, -1) + c(-1, 1) + c(1, -1) + c(1, 1)) - 2 * (c(-1, 0) + c(1, 0) + c(0, -1) + c(0, 1)) + 4 * c(0, 0)
    return float(np.sqrt(math.pi / 2) / (6.0 * (H - 2) * (W - 2)) * np.abs(L).sum())


def gaussian_smooth(img, sigma):
    r = int(math.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    p = np.pad(img, r, mode="reflect")
    t = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * t[i:i + img.shape[0], :] for i in range(2 * r + 1))


# ------------------------------------------------------------------ strength matching
def neff_gaussian(sigma_s, radius=None):
    r = min(int(math.ceil(3 * sigma_s)), MAX_RADIUS) if radius is None else radius
    x = np.arange(-r, r + 1)
    w = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma_s ** 2))
    return float(w.sum() ** 2 / (w ** 2).sum())


def flat_variance_ratio(fn, field):
    """S(F) = Var(F(noise)) / Var(noise) for a zero-mean white noise field (interior only)."""
    out = fn(field)
    b = 12
    return float(out[b:-b, b:-b].var() / field[b:-b, b:-b].var())


def match_sigma_s(target_ratio, sigma_noise, sigma_r_factor, field, lo=0.6, hi=6.0):
    """Bisection on sigma_s so that the bilateral flat-field variance ratio equals target_ratio."""
    f = lambda s: flat_variance_ratio(lambda x: bilateral(x + 128.0, s, sigma_r_factor * sigma_noise) - 128.0, field)
    for _ in range(11):
        mid = 0.5 * (lo + hi)
        if f(mid) > target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ metrics for blockiness
def tv_l1(x):
    return float((np.abs(np.diff(x, axis=0)).sum() + np.abs(np.diff(x, axis=1)).sum()) / x.size)


def circle_params(gt):
    mask = gt == gt.min()
    ys, xs = np.nonzero(mask)
    # the darkest level also forms the circle only (value 20); radius from area
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(mask.sum() / np.pi))


def boundary_radius(img, cy, cx, R, level, n_theta=720):
    """Sub-pixel radius r(theta) of the `level` crossing along rays from the circle centre (NaN where there is no
    crossing, e.g. where the disc is hidden behind the ramp rectangle); returns (r(theta), fraction of rays found)."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr = np.arange(R - 12, R + 12.01, 0.25)
    Y = cy + np.outer(np.sin(th), rr)
    X = cx + np.outer(np.cos(th), rr)
    inb = (Y >= 0) & (Y <= img.shape[0] - 1.001) & (X >= 0) & (X <= img.shape[1] - 1.001)
    y0, x0 = np.floor(Y).astype(int).clip(0, img.shape[0] - 2), np.floor(X).astype(int).clip(0, img.shape[1] - 2)
    fy, fx = Y - y0, X - x0
    v = (img[y0, x0] * (1 - fy) * (1 - fx) + img[y0 + 1, x0] * fy * (1 - fx) + img[y0, x0 + 1] * (1 - fy) * fx + img[y0 + 1, x0 + 1] * fy * fx)
    out = np.full(n_theta, np.nan)
    for t in range(n_theta):
        s = v[t] - level
        cross = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]            # dark inside -> brighter outside
        if len(cross):
            i = cross[np.argmin(np.abs(rr[cross] - R))]
            out[t] = rr[i] + 0.25 * (-s[i]) / (s[i + 1] - s[i])
    return out, float((~np.isnan(out)).mean())


def ramp_box(gt, margin=6):
    """Bounding box (r0,r1,c0,c1) of the ramp rectangle: the only region whose horizontal neighbours differ by small
    non-zero steps (true edges jump by >= 20 grey levels)."""
    gx = np.abs(np.diff(gt, axis=1))
    rws, cls = np.nonzero((gx > 0) & (gx < 6))
    return (int(np.percentile(rws, 1)) + margin, int(np.percentile(rws, 99)) - margin,
            int(np.percentile(cls, 1)) + margin, int(np.percentile(cls, 99)) - margin)


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "seed": SEED, "sigma_main": SIGMA_MAIN}
    gt = C.load_gray(os.path.join(C.PROVIDED, "q4_groundtruth.png"))
    R["ground_truth"] = {"shape": list(gt.shape), "n_distinct_levels": int(len(np.unique(gt))), "min": float(gt.min()), "max": float(gt.max())}
    cy, cx, Rc = circle_params(gt)
    bg = float(np.median(gt))
    level = 0.5 * (gt.min() + bg)
    R["circle"] = {"centre_row_col": [cy, cx], "radius_px": Rc, "inside_level": float(gt.min()), "background_level": bg}
    rng = np.random.default_rng(SEED)
    noise_field = rng.normal(0, SIGMA_MAIN, gt.shape)
    noisy = gt + noise_field
    R["noisy_input"] = {"PSNR_dB_(clip_for_metric)": C.psnr(gt, noisy), "SSIM": C.ssim(gt, np.clip(noisy, 0, 255)),
                        "immerkaer_sigma_estimate": noise_sigma_immerkaer(noisy)}
    sig_hat = noise_sigma_immerkaer(noisy)

    # ---------------- 4.1a strength matching (flat-field noise variance reduction)
    flat = np.random.default_rng(SEED + 1).normal(0, SIGMA_MAIN, (176, 176))
    Ns = [5, 7, 9, 11, 15]
    match_rows = []
    sigma_r = 2.0 * SIGMA_MAIN
    for N in Ns:
        m = (N + 1) // 2
        S_k = flat_variance_ratio(lambda x: kuwahara(x, N), flat)
        S_soft = flat_variance_ratio(lambda x: soft_kuwahara(x, N, SIGMA_MAIN ** 2), flat)       # tau = sigma^2 (true sigma on the flat field)
        sig_neff = m / (2 * math.sqrt(math.pi))                                  # n_eff(Gaussian) = 4 pi sigma_s^2 = m^2
        S_b_neff = flat_variance_ratio(lambda x: bilateral(x + 128.0, sig_neff, sigma_r) - 128.0, flat)
        sig_var = match_sigma_s(S_k, SIGMA_MAIN, 2.0, flat)
        S_b_var = flat_variance_ratio(lambda x: bilateral(x + 128.0, sig_var, sigma_r) - 128.0, flat)
        match_rows.append({"N": N, "sub_window_m": m, "kuwahara_flat_variance_ratio_measured": S_k, "soft_kuwahara_flat_variance_ratio_(tau=sigma^2)": S_soft, "analytic_1/m^2": 1.0 / m ** 2,
                           "sigma_s_from_neff_(m/(2 sqrt(pi)))": sig_neff, "neff_discrete_gaussian_of_that_sigma": neff_gaussian(sig_neff),
                           "bilateral_variance_ratio_at_that_sigma_s": S_b_neff,
                           "sigma_s_variance_matched_(bisection)": sig_var, "bilateral_variance_ratio_at_matched": S_b_var,
                           "match_reached_(within_5%)": bool(abs(S_b_var / S_k - 1) < 0.05), "bilateral_window_(2r+1)": 2 * min(int(math.ceil(3 * sig_var)), MAX_RADIUS) + 1})
    R["q4_1a_matching"] = {"rows": match_rows, "sigma_r_rule": "sigma_r = 2 sigma_noise = %.1f" % sigma_r,
                           "definition": "strength S(F)=Var(F(n))/Var(n) on a flat zero-mean white-noise field; bilateral sigma_s chosen so S matches Kuwahara"}

    # ---------------- 4.1b performance
    perf = []
    outs = {}
    for row_ in match_rows:
        N = row_["N"]
        ss = row_["sigma_s_variance_matched_(bisection)"]
        ob = bilateral(noisy, ss, sigma_r)
        ok_ = kuwahara(noisy, N)
        tau = 1.0 * sig_hat ** 2
        os_ = soft_kuwahara(noisy, N, tau)
        outs[N] = (ob, ok_, os_)
        perf.append({"N": N, "sigma_s_bilateral": ss,
                     "bilateral": {"PSNR": C.psnr(gt, ob), "SSIM": C.ssim(gt, ob)},
                     "kuwahara": {"PSNR": C.psnr(gt, ok_), "SSIM": C.ssim(gt, ok_)},
                     "soft_kuwahara": {"PSNR": C.psnr(gt, os_), "SSIM": C.ssim(gt, os_), "tau": tau}})
    R["q4_1b_performance"] = {"sigma": SIGMA_MAIN, "data_range": 255, "ssim_impl": "Gaussian 11x11 sigma 1.5, K1=.01 K2=.03, valid region (common.py; equals skimage within 1e-14)",
                              "rows": perf, "sigma_r": sigma_r}
    # oracle tuned sigma_r at N=7 and N=11 for the bilateral filter
    orc = []
    for N in (7, 11):
        ss = [r for r in match_rows if r["N"] == N][0]["sigma_s_variance_matched_(bisection)"]
        best = None
        for fac in (1.0, 2.0, 3.0, 4.0, 6.0):
            o = bilateral(noisy, ss, fac * SIGMA_MAIN)
            p = C.psnr(gt, o)
            orc.append({"N": N, "sigma_r/sigma": fac, "PSNR": p, "SSIM": C.ssim(gt, o)})
    R["q4_1b_bilateral_sigma_r_sweep_(oracle_view)"] = orc
    # sigma sweep at N=7 matched
    sig_rows = []
    for sg in (10.0, 15.0, 20.0, 30.0):
        nz = gt + np.random.default_rng(SEED + 7).normal(0, sg, gt.shape)
        N = 7
        fld = np.random.default_rng(SEED + 1).normal(0, sg, (176, 176))
        S_k = flat_variance_ratio(lambda x: kuwahara(x, N), fld)
        ss = match_sigma_s(S_k, sg, 2.0, fld)
        sh = noise_sigma_immerkaer(nz)
        r_ = {"sigma": sg, "noisy_PSNR": C.psnr(gt, nz), "sigma_s_matched": ss}
        for nm, o in (("bilateral", bilateral(nz, ss, 2 * sg)), ("kuwahara", kuwahara(nz, N)), ("soft_kuwahara", soft_kuwahara(nz, N, sh ** 2))):
            r_[nm] = {"PSNR": C.psnr(gt, o), "SSIM": C.ssim(gt, o)}
        sig_rows.append(r_)
    R["q4_1b_sigma_sweep_N7"] = sig_rows

    # figure: matched outputs and boundary crops (N=11 where blockiness shows, N=7 for comparison)
    N = 11
    ob, ok_, os_ = outs[N]
    cr = (int(cy - Rc - 25), int(cy - Rc + 75), int(cx + 40), int(cx + 140))        # patch of the circle edge upper right
    cr = (max(cr[0], 0), cr[1], min(cr[2], 512 - 100), cr[3])
    fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
    panels = [("clean", gt), ("noisy (sigma=%g)" % SIGMA_MAIN, np.clip(noisy, 0, 255)), ("bilateral", ob), ("Kuwahara N=%d" % N, ok_), ("soft-Kuwahara N=%d" % N, os_)]
    for j, (t, im) in enumerate(panels):
        ax[0, j].imshow(np.clip(im, 0, 255), cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(t); ax[0, j].axis("off")
        ax[1, j].imshow(np.clip(im, 0, 255)[cr[0]:cr[1], cr[2]:cr[3]], cmap="gray", vmin=0, vmax=255); ax[1, j].axis("off")
        if j == 0:
            ax[0, j].add_patch(plt.Rectangle((cr[2], cr[0]), cr[3] - cr[2], cr[1] - cr[0], fill=False, ec="red"))
    ax[1, 0].set_title("zoom of red box (circle boundary)")
    C.savefig(plt, fig, "q4_matched_outputs.png")
    # second zoom: ramp rectangle edge
    ys_, xs_ = np.nonzero(np.abs(np.diff(gt, axis=1)) > 0)  # not used; locate ramp by value range
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.4))
    rb0, rb1, cb0, cb1 = ramp_box(gt)
    cr2 = (rb0 - 8, rb1 + 8, cb0 - 8, cb1 + 8)
    for j, (t, im) in enumerate(panels):
        ax[j].imshow(np.clip(im, 0, 255)[cr2[0]:cr2[1], cr2[2]:cr2[3]], cmap="gray", vmin=40, vmax=130); ax[j].set_title(t); ax[j].axis("off")
    C.savefig(plt, fig, "q4_ramp_zoom.png")

    # ---------------- 4.2b blockiness
    clean = gt
    blk = []
    # ramp region = rows/cols where the clean image has a horizontal gradient (levels between 60 and 200 changing along x)
    ramp_mask = np.zeros_like(gt, bool)
    r0, r1, c0, c1 = ramp_box(gt)
    ramp_mask[r0:r1, c0:c1] = True
    R["ramp_box_(r0,r1,c0,c1)"] = [int(r0), int(r1), int(c0), int(c1)]
    dist = np.hypot(*(np.indices(gt.shape) - np.array([cy, cx])[:, None, None]))
    strong_edge = np.zeros_like(gt, bool)
    gy_, gx2 = np.gradient(gt)
    se = np.hypot(gy_, gx2) > 8
    from scipy.ndimage import binary_dilation
    smooth_mask = ~binary_dilation(se, iterations=8)
    tv_clean = tv_l1(gt[smooth_mask.any(1)][:, :]) if False else None
    r_clean, _ = boundary_radius(clean, cy, cx, Rc, level)

    def metrics(o, tag):
        tvx = (np.abs(np.diff(o, axis=0))[smooth_mask[:-1]].sum() + np.abs(np.diff(o, axis=1))[smooth_mask[:, :-1]].sum()) / (smooth_mask[:-1].sum() + smooth_mask[:, :-1].sum())
        tvc = (np.abs(np.diff(clean, axis=0))[smooth_mask[:-1]].sum() + np.abs(np.diff(clean, axis=1))[smooth_mask[:, :-1]].sum()) / (smooth_mask[:-1].sum() + smooth_mask[:, :-1].sum())
        r_o, frac = boundary_radius(o, cy, cx, Rc, level)
        ok_ = ~np.isnan(r_o) & ~np.isnan(r_clean)
        dev = float(np.sqrt(np.mean((r_o[ok_] - r_clean[ok_]) ** 2))) if ok_.any() else float("nan")
        ramp_rms = float(np.sqrt(np.mean((o[ramp_mask] - clean[ramp_mask]) ** 2)))
        gx_ramp = np.diff(o, axis=1)[:, :][ramp_mask[:, 1:]]
        return {"TV_smooth_regions_(mean|grad|)": float(tvx), "TV_clean_same_mask": float(tvc), "TV_excess": float(tvx - tvc),
                "circle_contour_deviation_from_clean_px": dev, "boundary_found_fraction": frac, "ramp_RMS_dev_from_clean": ramp_rms,
                "ramp_gradient_std_(clean_ramp_slope_is_constant)": float(gx_ramp.std()), "TV_whole_image": tv_l1(o)}
    seeds = range(4)
    for N in (7, 11, 15):
        m = (N + 1) // 2
        ss = [r for r in match_rows if r["N"] == N]
        ss = ss[0]["sigma_s_variance_matched_(bisection)"] if ss else m / (2 * math.sqrt(math.pi))
        acc = {"classic": [], "soft": [], "bilateral": []}
        for sd in seeds:
            nz = gt + np.random.default_rng(1000 + sd).normal(0, SIGMA_MAIN, gt.shape)
            sh = noise_sigma_immerkaer(nz)
            acc["classic"].append(metrics(kuwahara(nz, N), "c"))
            acc["soft"].append(metrics(soft_kuwahara(nz, N, sh ** 2), "s"))
            acc["bilateral"].append(metrics(bilateral(nz, ss, 2 * SIGMA_MAIN), "b"))
        row = {"N": N, "noisy_input": {k: float(np.mean([metrics(gt + np.random.default_rng(1000 + sd).normal(0, SIGMA_MAIN, gt.shape), "n")[k] for sd in seeds])) for k in ("TV_smooth_regions_(mean|grad|)", "circle_contour_deviation_from_clean_px", "ramp_RMS_dev_from_clean")}}
        for kname, lst in acc.items():
            row["noisy_" + kname] = {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}
            row["noisy_" + kname + "_std_over_seeds"] = {k: float(np.std([d[k] for d in lst])) for k in ("TV_excess", "circle_contour_deviation_from_clean_px", "ramp_RMS_dev_from_clean", "ramp_gradient_std_(clean_ramp_slope_is_constant)")}
        # noise-free input: pure filter artefact
        row["clean_classic"] = metrics(kuwahara(clean, N), "cc")
        row["clean_soft"] = metrics(soft_kuwahara(clean, N, max(sig_hat ** 2, 1e-6)), "cs")
        row["clean_bilateral"] = metrics(bilateral(clean, ss, 2 * SIGMA_MAIN), "cb")
        row["clean_input"] = metrics(clean, "ci")
        blk.append(row)
    R["q4_2b_blockiness"] = {"rows": blk, "seeds": len(list(seeds)), "tau_rule": "tau = (Immerkaer sigma_hat)^2 on noisy input; on the clean input tau uses the noisy estimate %.2f^2" % sig_hat}
    # tau sweep (soft Kuwahara) at N=11
    taus = []
    for lam in (0.0625, 0.25, 1.0, 4.0, 16.0):
        o = soft_kuwahara(noisy, 11, lam * sig_hat ** 2)
        mm = metrics(o, "t")
        taus.append({"lambda_tau/sigma2": lam, "PSNR": C.psnr(gt, o), "SSIM": C.ssim(gt, o), **{k: mm[k] for k in ("TV_excess", "circle_contour_deviation_from_clean_px", "ramp_RMS_dev_from_clean", "ramp_gradient_std_(clean_ramp_slope_is_constant)")}})
    R["q4_2b_tau_sweep_N11"] = taus
    # before/after crops at identical window size
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for i, N in enumerate((11, 15)):
        o_c, o_s = kuwahara(noisy, N), soft_kuwahara(noisy, N, sig_hat ** 2)
        for j, (t, im) in enumerate([("noisy", noisy), ("classical N=%d" % N, o_c), ("soft N=%d" % N, o_s)]):
            ax[i, j].imshow(np.clip(im, 0, 255)[cr[0]:cr[1], cr[2]:cr[3]], cmap="gray", vmin=0, vmax=255); ax[i, j].set_title(t); ax[i, j].axis("off")
        ax[i, 3].imshow(np.clip(np.vstack([kuwahara(gt, N)[rb0:rb1, cb0:cb0 + 160], soft_kuwahara(gt, N, sig_hat ** 2)[rb0:rb1, cb0:cb0 + 160]]), 66, 104), cmap="gray", vmin=66, vmax=104)
        ax[i, 3].set_title("noise-free ramp (display range 66-104): classical (top) / soft (bottom)"); ax[i, 3].axis("off")
    C.savefig(plt, fig, "q4_deblock.png")
    # window selection map (shows blockiness mechanism)
    means, vars_ = _quadrants(noisy, 11)
    sel = np.argmin(vars_, axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(sel[cr[0]:cr[1], cr[2]:cr[3]], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[0].set_title("argmin window index (TL,TR,BL,BR=0..3)")
    wts = np.exp(-(vars_ - vars_.min(0, keepdims=True)) / sig_hat ** 2); wts /= wts.sum(0)
    ax[1].imshow(wts.max(0)[cr[0]:cr[1], cr[2]:cr[3]], cmap="viridis", vmin=0.25, vmax=1); ax[1].set_title("soft: max weight (0.25=uniform, 1=hard)")
    ax[2].imshow(np.clip(kuwahara(noisy, 11), 0, 255)[cr[0]:cr[1], cr[2]:cr[3]], cmap="gray", vmin=0, vmax=255); ax[2].set_title("classical output N=11")
    for a in ax: a.axis("off")
    C.savefig(plt, fig, "q4_selection_map.png")

    # ---------------- 4.2c complexity
    comp = []
    for N in (7, 11, 15):
        m = (N + 1) // 2
        n = N * N
        bil = {"additions_subtractions": 3 * n, "multiplications": 4 * n, "exponentiations": n, "divisions": 1,
               "note": "per neighbour: sub, square, scale by -1/(2 sr^2), exp, ws*wr, w*I, 2 accumulations; ws precomputed"}
        kn = {"additions_subtractions": 4 * (2 * (m * m - 1) + 1) + 0, "multiplications": 4 * (m * m + 3), "exponentiations": 0, "comparisons": 3, "divisions": 0,
              "note": "direct window sums: m^2-1 adds for sum, m^2 mult + m^2-1 adds for sum of squares, then mean, mean^2, variance, x4 windows"}
        ki = {"additions_subtractions": 4 * (3 + 3 + 1), "multiplications": 4 * 3 + 2, "exponentiations": 0, "comparisons": 3, "divisions": 0,
              "note": "with integral images of I and I^2 (2 adds + 1 mult per pixel to build): 4 corner lookups = 3 adds per box sum, independent of N"}
        sk = {"additions_subtractions": kn["additions_subtractions"] + 4 + 3 + 3, "multiplications": kn["multiplications"] + 4 + 4, "exponentiations": 4, "divisions": 1,
              "note": "classical + 4 subs (v-vmin), 4 scalings by 1/tau, 4 exps, 4 w*mean, 6 adds, 1 division"}
        sk_i = {"additions_subtractions": ki["additions_subtractions"] + 4 + 3 + 3, "multiplications": ki["multiplications"] + 4 + 4, "exponentiations": 4, "divisions": 1,
                "note": "same with integral-image statistics"}
        comp.append({"N": N, "m": m, "bilateral_(2r+1)^2": bil, "classical_kuwahara_direct": kn, "classical_kuwahara_integral": ki, "soft_kuwahara_direct": sk, "soft_kuwahara_integral": sk_i})
    # measured runtimes
    tims = []
    for N in (7, 11, 15):
        ss = [r for r in match_rows if r["N"] == N][0]["sigma_s_variance_matched_(bisection)"]
        def tm(fn):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        tims.append({"N": N, "bilateral_s": tm(lambda: bilateral(noisy, ss, sigma_r)), "bilateral_s_with_radius_N//2": tm(lambda: bilateral(noisy, ss, sigma_r, radius=N // 2)),
                     "kuwahara_s": tm(lambda: kuwahara(noisy, N)), "soft_kuwahara_s": tm(lambda: soft_kuwahara(noisy, N, 1.0))})
    R["q4_2c_complexity"] = {"ops_per_pixel": comp, "timings_512x512": tims}

    # ---------------- 4.3 breakdown analysis
    ncrs = np.round(np.geomspace(0.05, 2.0, 30), 4)
    dI, I1 = 100.0, 60.0
    S = 64
    edge_col = S // 2                       # step between columns 31 and 32
    reps = 40

    def make_step(sigma, rng_):
        im = np.full((S, S), I1)
        im[:, edge_col:] = I1 + dI
        return im, im + rng_.normal(0, sigma, im.shape)

    def edge_metrics(sigma_s, rule):
        rows = []
        for ncr in ncrs:
            sigma = ncr * dI
            sr = 2.0 * sigma if rule == "A" else 0.5 * dI
            rng_ = np.random.default_rng(SEED + int(1e4 * ncr))
            prof = np.zeros(S)
            band_mse = 0.0
            for _ in range(reps):
                clean_s, noisy_s = make_step(sigma, rng_)
                o = bilateral(noisy_s, sigma_s, sr)
                prof += o[8:-8].mean(0)
                band_mse += np.mean((o[8:-8, edge_col - 3:edge_col + 3] - clean_s[8:-8, edge_col - 3:edge_col + 3]) ** 2)
            prof /= reps
            band_mse /= reps
            grad = np.abs(np.diff(prof))
            # retention: max slope of the mean profile near the edge, relative to the step height
            g = float(grad[edge_col - 4:edge_col + 3].max() / dI)
            rows.append({"NCR": float(ncr), "sigma": sigma, "sigma_r": sr, "gradient_retention": g,
                         "edge_band_PSNR_dB_(peak=dI)": float(10 * np.log10(dI ** 2 / band_mse)),
                         "edge_band_PSNR_noisy_dB": float(10 * np.log10(dI ** 2 / sigma ** 2))})
        return rows

    def crossing(rows, key, level_):
        x = np.log([r["NCR"] for r in rows]); y = np.array([r[key] for r in rows])
        for i in range(len(y) - 1):
            if (y[i] - level_) * (y[i + 1] - level_) <= 0 and y[i] != y[i + 1]:
                t = (level_ - y[i]) / (y[i + 1] - y[i])
                return float(np.exp(x[i] + t * (x[i + 1] - x[i])))
        return None

    def smooth_baseline(sigma_s):
        # plain Gaussian smoothing (the "completely smoothing" limit): gradient retention of a Gaussian step response
        return 1.0 / (sigma_s * math.sqrt(2 * math.pi))

    q43 = {}
    for sigma_s in (1.0, 2.0, 3.0):
        for rule in ("A", "B"):
            rows = edge_metrics(sigma_s, rule)
            gl = smooth_baseline(sigma_s)
            q43["sigma_s=%g,rule=%s" % (sigma_s, rule)] = {
                "rows": rows, "NCR*_retention_0.5": crossing(rows, "gradient_retention", 0.5),
                "NCR*_retention_halfway_to_gaussian_limit": crossing(rows, "gradient_retention", 0.5 * (1 + gl)),
                "gaussian_limit_retention_(continuous)": gl,
                "retention_at_NCR_0.05": rows[0]["gradient_retention"]}
    R["q4_3_breakdown"] = {"edge": {"I1": I1, "I2": I1 + dI, "dI": dI}, "reps": reps, "image": S, "primary": "sigma_s=2,rule=A (sigma_r = 2*sigma_noise)",
                           "results": q43, "NCR_grid": ncrs}
    # analytic quantities
    # (1) pairwise expected-weight ratio, with noise in the difference (D ~ N(dI, 2 sigma^2) cross, N(0, 2 sigma^2) same side)
    rho = lambda ncr, c=2.0: math.exp(-1.0 / (2 * (c ** 2 + 2) * ncr ** 2))
    ncr_half = 1.0 / math.sqrt(12 * math.log(2))
    # Monte-Carlo check of the closed form E[w_cross]/E[w_same]
    mc = []
    rng_ = np.random.default_rng(SEED + 3)
    for ncr in (0.2, 0.347, 0.5, 1.0):
        sg = ncr * dI; sr = 2 * sg
        n1, n2 = rng_.normal(0, sg, 2_000_000), rng_.normal(0, sg, 2_000_000)
        wc = np.exp(-(dI + n2 - n1) ** 2 / (2 * sr ** 2)).mean()
        ws_ = np.exp(-(n2 - n1) ** 2 / (2 * sr ** 2)).mean()
        mc.append({"NCR": ncr, "MC_ratio": float(wc / ws_), "closed_form_exp(-1/(12 NCR^2))": rho(ncr),
                   "naive_noise_free_exp(-dI^2/(2 sr^2))_=exp(-1/(8 NCR^2))": math.exp(-1 / (8 * ncr ** 2))})
    # (2) expected-weight (ratio-of-expectations) profile model for retention
    def model_retention(ncr, sigma_s, rule="A"):
        sg = ncr * dI; sr = 2 * sg if rule == "A" else 0.5 * dI
        r = int(math.ceil(3 * sigma_s))
        xs = np.arange(-r, r + 1)
        ws = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / (2 * sigma_s ** 2))
        prof = []
        for x in range(-6, 6):                          # pixel column x relative to edge (edge between -1 and 0)
            Ij = np.where(x + xs[None, :] >= 0, dI, 0.0) * np.ones_like(ws)
            Ic = dI if x >= 0 else 0.0
            mu = Ij - Ic
            Ew = np.sqrt(sr ** 2 / (sr ** 2 + 2 * sg ** 2)) * np.exp(-mu ** 2 / (2 * (sr ** 2 + 2 * sg ** 2)))
            Ew[r, r] = 1.0                              # centre pixel: weight exactly 1
            wgt = ws * Ew
            prof.append(Ic + (wgt * (Ij - Ic)).sum() / wgt.sum())
        return float(np.abs(np.diff(prof)).max() / dI)
    model = [{"NCR": float(n), "model_retention": model_retention(n, 2.0)} for n in ncrs]
    R["q4_3_analytic"] = {"rho_half_NCR_(closed_form)": ncr_half, "monte_carlo_check": mc, "expected_weight_model_sigma_s=2_ruleA": model,
                          "model_NCR*_retention_0.5": crossing(model, "model_retention", 0.5),
                          "rho_at_empirical_NCR*": rho(q43["sigma_s=2,rule=A"]["NCR*_retention_0.5"]) if q43["sigma_s=2,rule=A"]["NCR*_retention_0.5"] else None}
    # noise-free sharpness check: with NCR->0 retention -> 1
    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, st in (("sigma_s=1,rule=A", "--"), ("sigma_s=2,rule=A", "-"), ("sigma_s=3,rule=A", ":")):
        rows = q43[key]["rows"]
        ax[0].semilogx([r["NCR"] for r in rows], [r["gradient_retention"] for r in rows], st, label=key)
    prim = q43["sigma_s=2,rule=A"]
    nstar = prim["NCR*_retention_0.5"]
    ax[0].semilogx([m_["NCR"] for m_ in model], [m_["model_retention"] for m_ in model], "k-.", lw=1, label="expected-weight model (sigma_s=2)")
    ax[0].axhline(0.5, color="gray", lw=0.6)
    ax[0].axvline(nstar, color="r"); ax[0].annotate("NCR* = %.3f\n(retention 0.5)" % nstar, (nstar, 0.52), xytext=(nstar * 1.3, 0.75), arrowprops=dict(arrowstyle="->", color="r"), color="r")
    ax[0].axvline(ncr_half, color="b", ls=":"); ax[0].text(ncr_half * 0.97, 0.05, "weight-ratio 0.5\nNCR=%.3f" % ncr_half, color="b", ha="right", fontsize=8)
    ax[0].set_xlabel("NCR = sigma/|dI|"); ax[0].set_ylabel("edge gradient retention"); ax[0].set_title("rule A: sigma_r=2 sigma"); ax[0].legend(fontsize=7)
    for key, st in (("sigma_s=2,rule=A", "-"), ("sigma_s=2,rule=B", "--")):
        rows = q43[key]["rows"]
        ax[1].semilogx([r["NCR"] for r in rows], [r["edge_band_PSNR_dB_(peak=dI)"] for r in rows], st, label=key + " filtered")
    ax[1].semilogx([r["NCR"] for r in prim["rows"]], [r["edge_band_PSNR_noisy_dB"] for r in prim["rows"]], "k:", label="unfiltered noisy")
    ax[1].axvline(nstar, color="r"); ax[1].set_xlabel("NCR"); ax[1].set_ylabel("edge-band PSNR (dB, peak = |dI|)"); ax[1].legend(fontsize=7); ax[1].set_title("edge-region PSNR")
    ncg = np.geomspace(0.05, 2, 200)
    ax[2].semilogx(ncg, [rho(n) for n in ncg], label="rho=E[w_cross]/E[w_same], sigma_r=2 sigma")
    ax[2].semilogx(ncg, [math.exp(-1 / (8 * n ** 2)) for n in ncg], "--", label="noise-free w_r(dI)=exp(-dI^2/2sr^2)")
    ax[2].axhline(0.5, color="gray", lw=0.6); ax[2].axvline(ncr_half, color="b", ls=":"); ax[2].axvline(nstar, color="r")
    ax[2].set_xlabel("NCR"); ax[2].set_ylabel("weight ratio"); ax[2].legend(fontsize=7); ax[2].set_title("analytic range-weight ratio")
    C.savefig(plt, fig, "q4_breakdown.png")
    # example step images at three NCR
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for j, ncr in enumerate((0.1, 0.3, 0.6, 1.2)):
        sg = ncr * dI
        cl, nz = make_step(sg, np.random.default_rng(5))
        ax[0, j].imshow(nz, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title("noisy, NCR=%.1f" % ncr)
        ax[1, j].imshow(bilateral(nz, 2.0, 2 * sg), cmap="gray", vmin=0, vmax=255); ax[1, j].set_title("bilateral (sigma_s=2, sigma_r=2 sigma)")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    C.savefig(plt, fig, "q4_step_examples.png")

    C.save_json(R, "q4.json")
    print("q4 done")


if __name__ == "__main__":
    main()
