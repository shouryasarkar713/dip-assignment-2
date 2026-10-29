"""Question 5 - Wiener denoising (white additive noise, MMSE), comparison with Q2 low-pass and Q4 bilateral
filters on the instructor-provided reference photograph, noise-power mis-specification, and a real noisy photo.

Working image: reference_image.jpg (3072x2048) reduced by 2x2 block averaging to 1536x1024 (both sides >= 1024).
All metrics: 0..255 scale, data range 255, SSIM = common.ssim (Gaussian 11x11, sigma 1.5, per channel, mean).
Run: python src/q5_wiener.py -> results/q5.json, figures/q5_*.png
"""
import math
import os

import numpy as np
from scipy.fft import next_fast_len
from PIL import Image

import common as C
import q4_edge_preserving as Q4

SEED = 2026
SIGMAS = (5, 10, 20, 40)
PAD = 64


# ------------------------------------------------------------------ Wiener denoiser
def _pad_sym(g, pad=PAD):
    M, N = g.shape
    Pr, Pc = next_fast_len(M + 2 * pad), next_fast_len(N + 2 * pad)
    r0, c0 = (Pr - M) // 2, (Pc - N) // 2
    return np.pad(g, ((r0, Pr - M - r0), (c0, Pc - N - c0)), mode="symmetric"), (r0, c0)


def radial_psd(Pg, nbins=240):
    """Log-spaced radial average of a periodogram on a (Pr,Pc) grid; returns interpolator inputs."""
    Pr, Pc = Pg.shape
    fy, fx = np.fft.fftfreq(Pr), np.fft.fftfreq(Pc)
    rho = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.geomspace(1.0 / max(Pr, Pc), rho.max() * 1.0001, nbins + 1)
    idx = np.clip(np.searchsorted(edges, rho.ravel(), side="right") - 1, 0, nbins - 1)
    s = np.bincount(idx, weights=Pg.ravel(), minlength=nbins)
    n = np.bincount(idx, minlength=nbins)
    good = n > 0
    centre = np.sqrt(edges[:-1] * edges[1:])
    return centre[good], (s[good] / n[good]), rho


def wiener_denoise(g, sigma, noise_scale=1.0, floor=0.01, refit_signal_with_assumed=False, return_parts=False):
    """g: noisy channel in [0,1] units (NOT clipped).  sigma: noise std in the same units (Snn = sigma^2, white).
    Sff = radially averaged periodogram - Snn (floored at floor*Snn).  noise_scale multiplies the noise power used in the
    denominator (mis-specification test); if refit_signal_with_assumed the same assumed power is also subtracted in Sff."""
    M, N = g.shape
    gp, (r0, c0) = _pad_sym(g)
    Pr, Pc = gp.shape
    G = np.fft.fft2(gp)
    Pg = np.abs(G) ** 2 / (Pr * Pc)
    snn = sigma ** 2
    centre, prad, rho = radial_psd(Pg)
    sub = snn * (noise_scale if refit_signal_with_assumed else 1.0)
    sff_r = np.maximum(prad - sub, floor * snn)
    Sff = np.exp(np.interp(np.log(np.maximum(rho, centre[0])), np.log(centre), np.log(sff_r)))
    W = Sff / (Sff + noise_scale * snn)
    out = np.real(np.fft.ifft2(W * G))[r0:r0 + M, c0:c0 + N]
    if return_parts:
        return out, {"W": W, "Sff": Sff, "rho": rho, "radial_centre": centre, "radial_Pg": prad, "radial_Sff": sff_r, "snn": snn}
    return out


def lowpass_rect(g, fc, kind, n=2):
    """Zero-phase low-pass with half-amplitude radius fc (cycles/pixel); same formulas as Q2 (Butterworth n=2,
    Gaussian with D0' = D0/sqrt(2 ln 2)), symmetric padding instead of zero padding to avoid border ringing."""
    M, N = g.shape
    gp, (r0, c0) = _pad_sym(g)
    Pr, Pc = gp.shape
    fy, fx = np.fft.fftfreq(Pr), np.fft.fftfreq(Pc)
    D = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    if kind == "butterworth":
        H = 1.0 / (1.0 + (D / fc) ** (2 * n))
    else:
        H = np.exp(-D ** 2 / (2 * (fc / math.sqrt(2 * math.log(2))) ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(gp) * H))[r0:r0 + M, c0:c0 + N]


def per_channel(fn, img):
    return np.stack([fn(img[..., c]) for c in range(img.shape[2])], axis=-1)


def metrics(ref, out):
    return {"PSNR": C.psnr(ref, out), "SSIM": C.ssim(ref, np.clip(out, 0, 255))}


# ------------------------------------------------------------------ real-photo noise analysis
def plane_detrended_std(p):
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([np.ones(h * w), yy.ravel() - h / 2, xx.ravel() - w / 2], 1)
    coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
    res = p.ravel() - A @ coef
    return float(np.sqrt((res ** 2).sum() / (h * w - 3))), float(p.mean())


def patch_noise(chan, patch=24, keep_frac=0.1, max_patches=4000, mean_range=(8, 245)):
    """Noise-level function candidates: detrended std of non-overlapping patches; the `keep_frac` flattest
    (smallest 3x3-Laplacian energy) patches are the flat-region candidates."""
    H, W = chan.shape
    rows = []
    for i in range(0, H - patch + 1, patch):
        for j in range(0, W - patch + 1, patch):
            p = chan[i:i + patch, j:j + patch]
            if not (mean_range[0] < p.mean() < mean_range[1]):
                continue
            sd, m = plane_detrended_std(p)
            lap = np.abs(p[1:-1, 1:-1] * 4 - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]).mean()
            rows.append((sd, m, lap, i, j))
    rows = np.array(rows)
    return rows


def nps_radial(patch_stack):
    """Mean 2-D noise power spectrum of detrended square patches (unit-variance normalised) -> radial average."""
    n = patch_stack[0].shape[0]
    acc = np.zeros((n, n))
    for p in patch_stack:
        sd, _ = plane_detrended_std(p)
        yy, xx = np.mgrid[0:n, 0:n]
        A = np.stack([np.ones(n * n), yy.ravel() - n / 2, xx.ravel() - n / 2], 1)
        coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
        r = (p.ravel() - A @ coef).reshape(n, n)
        acc += np.abs(np.fft.fft2(r)) ** 2 / (n * n)
    acc /= len(patch_stack)
    fy = np.fft.fftfreq(n)
    rho = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
    bins = np.minimum((rho / 0.5 * 8).astype(int), 7)          # 8 radial bands up to 0.5
    prof = np.array([acc[bins == b].mean() for b in range(8)])
    return acc, prof


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "seed": SEED}
    src = C.load_rgb(os.path.join(C.PROVIDED, "reference_image.jpg"))
    R["reference"] = {"file": "data/provided/reference_image.jpg", "native_shape_HxWxC": list(src.shape)}
    ref = src[:src.shape[0] // 2 * 2, :src.shape[1] // 2 * 2].reshape(src.shape[0] // 2, 2, src.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    R["reference"]["working_shape_HxWxC"] = list(ref.shape)
    R["reference"]["working_note"] = "2x2 block average of the native image (area reduction); working image is float, 0..255; both sides >= 1024"
    C.save_png(ref, os.path.join(C.DERIVED, "reference_work_1536x1024.png"))
    H0, W0, _ = ref.shape
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(ref.shape)               # ONE unit-variance noise realisation, scaled for every sigma
    R["noise"] = {"unit_field_seed": SEED, "note": "noise_sigma = sigma * z with the same z for all methods; noisy input never clipped"}

    # crop used for oracle parameter tuning (centre 512x512)
    cy, cx = H0 // 2, W0 // 2
    crop = (slice(cy - 256, cy + 256), slice(cx - 256, cx + 256))
    R["tuning_crop_(rows,cols)"] = [[cy - 256, cy + 256], [cx - 256, cx + 256]]

    table = []
    outs = {}
    for s in SIGMAS:
        noisy = ref + s * z
        d = {"sigma": s, "noisy": {"PSNR": C.psnr(ref, noisy), "SSIM": C.ssim(ref, np.clip(noisy, 0, 255))}}
        # --- Wiener, known sigma and estimated sigma (white-noise assumption)
        sig01 = s / 255.0
        wk = per_channel(lambda c: wiener_denoise(c / 255.0, sig01) * 255.0, noisy)
        est = np.mean([Q4.noise_sigma_immerkaer(noisy[..., c]) for c in range(3)])
        we = per_channel(lambda c: wiener_denoise(c / 255.0, est / 255.0) * 255.0, noisy)
        d["wiener_known_sigma"] = metrics(ref, wk)
        d["wiener_estimated_sigma"] = dict(metrics(ref, we), sigma_hat=float(est))
        outs[(s, "wiener")] = wk
        # --- low-pass filters: oracle-tuned fc (grid) and blind fc from Sff=Snn crossing
        fcs = np.geomspace(0.01, 0.5, 28)
        for kind, nm in (("butterworth", "BLPF_n2"), ("gaussian", "GLPF")):
            # tune on the 512x512 crop (oracle: uses the clean reference), then apply to the full image
            sc = noisy[crop[0], crop[1], :]
            rc = ref[crop[0], crop[1], :]
            vals = []
            for fc in fcs:
                o = per_channel(lambda c: lowpass_rect(c, fc, kind), sc)
                vals.append(C.psnr(rc, o))
            fc_best = float(fcs[int(np.argmax(vals))])
            o_full = per_channel(lambda c: lowpass_rect(c, fc_best, kind), noisy)
            d[nm + "_oracle"] = dict(metrics(ref, o_full), fc_cycles_per_pixel=fc_best, tuned_on="512x512 centre crop, PSNR (oracle-tuned)")
            outs[(s, nm)] = o_full
        # blind cutoff: radius at which estimated Sff drops below Snn (from the Wiener parts of the green channel)
        _, parts = wiener_denoise(noisy[..., 1] / 255.0, sig01, return_parts=True)
        cen, sf = parts["radial_centre"], parts["radial_Sff"]
        below = np.nonzero(sf <= parts["snn"] * 1.0000001)[0]
        fc_blind = float(cen[below[0]]) if len(below) else 0.5
        o_blind = per_channel(lambda c: lowpass_rect(c, fc_blind, "gaussian"), noisy)
        d["GLPF_blind_Sff=Snn_crossing"] = dict(metrics(ref, o_blind), fc_cycles_per_pixel=fc_blind)
        # --- bilateral (Q4): grid tuned on crop (oracle), luminance-independent per channel
        sc = noisy[crop[0], crop[1], :]
        rc = ref[crop[0], crop[1], :]
        best, bp = -1, None
        for ss in (1.0, 2.0, 3.0):
            for fac in (1.0, 2.0, 3.0):
                o = per_channel(lambda c: Q4.bilateral(c, ss, fac * s), sc)
                p = C.psnr(rc, o)
                if p > best:
                    best, bp = p, (ss, fac)
        o_b = per_channel(lambda c: Q4.bilateral(c, bp[0], bp[1] * s), noisy)
        d["bilateral_oracle"] = dict(metrics(ref, o_b), sigma_s=bp[0], sigma_r_over_sigma=bp[1], tuned_on="512x512 centre crop, PSNR (oracle-tuned)")
        outs[(s, "bilateral")] = o_b
        table.append(d)
        print("sigma", s, "done", flush=True)
    R["q5_1_table"] = table
    R["q5_1_notes"] = {"data_range": 255, "ssim": "common.ssim: Gaussian window 11x11 sigma=1.5, K1=0.01, K2=0.03, valid region, mean over RGB channels; clip to [0,255] before metrics only",
                       "wiener": "Snn = sigma^2 (known) or Immerkaer estimate; Sff = log-radial mean periodogram - Snn (floor 0.01 Snn); symmetric padding 64 px", "oracle": "LPF and bilateral parameters tuned with the clean reference on a centre crop are oracle-tuned"}

    # --- shrinkage figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fcs_plot = {}
    for s in SIGMAS:
        noisy = ref[..., 1] + s * z[..., 1]
        _, parts = wiener_denoise(noisy / 255.0, s / 255.0, return_parts=True)
        Pr = parts["W"].shape[0]; Pc = parts["W"].shape[1]
        fxs = np.fft.fftfreq(Pc)[:Pc // 2]
        ax[0].semilogx(fxs[1:], parts["W"][0, 1:Pc // 2], label="sigma=%d" % s)
        ax[1].loglog(parts["radial_centre"], parts["radial_Pg"] / 255.0 ** 2 * 255.0 ** 2 / 255.0 ** 2, label="noisy periodogram sigma=%d" % s)
        ax[1].axhline((s / 255.0) ** 2, color="C%d" % SIGMAS.index(s), ls=":", lw=0.8)
        fcs_plot[s] = parts
    for kind, st in (("butterworth", "--"), ("gaussian", ":")):
        fx = np.fft.fftfreq(Pc)[1:Pc // 2]
        fc = 0.1
        H = 1 / (1 + (fx / fc) ** 4) if kind == "butterworth" else np.exp(-fx ** 2 / (2 * (fc / math.sqrt(2 * math.log(2))) ** 2))
        ax[0].semilogx(fx, H, "k" + st, lw=1, label="fixed LPF %s fc=0.1" % kind)
    ax[0].set_xlabel("horizontal frequency (cycles/pixel)"); ax[0].set_ylabel("W or H"); ax[0].set_title("Wiener gain W(f) along fy=0 (green channel)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("radial frequency"); ax[1].set_ylabel("power (per-pixel units, [0,1] images)"); ax[1].set_title("radial periodogram of noisy image, dotted = sigma^2"); ax[1].legend(fontsize=6)
    ax[2].imshow(np.fft.fftshift(fcs_plot[20]["W"]), cmap="viridis", extent=[-0.5, 0.5, 0.5, -0.5]); ax[2].set_title("2-D Wiener gain, sigma=20 (green)")
    C.savefig(plt, fig, "q5_wiener_gain.png")

    # --- visual comparison crops (sigma=20 and 40)
    r0, c0 = 420, 560
    cs = (slice(r0, r0 + 200), slice(c0, c0 + 300))
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.2))
    for i, s in enumerate((20, 40)):
        noisy = ref + s * z
        ims = [("clean", ref), ("noisy %d" % s, np.clip(noisy, 0, 255)), ("Wiener", outs[(s, "wiener")]), ("GLPF oracle", outs[(s, "GLPF")]), ("BLPF oracle", outs[(s, "BLPF_n2")]), ("bilateral oracle", outs[(s, "bilateral")])]
        for j, (t, im) in enumerate(ims):
            ax[i, j].imshow(np.clip(im[cs[0], cs[1]], 0, 255).astype(np.uint8)); ax[i, j].set_title(t, fontsize=9); ax[i, j].axis("off")
    C.savefig(plt, fig, "q5_comparison_crops.png")
    R["comparison_crop_(rows,cols)"] = [[r0, r0 + 200], [c0, c0 + 300]]

    # ================= 5.2 assumptions: mis-specified noise power at sigma = 20
    s = 20
    noisy = ref + s * z
    sig01 = s / 255.0
    # flat patch and edge pixels of the clean image
    gl = ref.mean(axis=2)
    bs = 48
    best_v, flat_ij = 1e18, None
    for i in range(0, H0 - bs, bs):
        for j in range(0, W0 - bs, bs):
            v = gl[i:i + bs, j:j + bs].var()
            if v < best_v:
                best_v, flat_ij = v, (i, j)
    gy_, gx_ = np.gradient(gl)
    gm = np.hypot(gy_, gx_)
    edge_mask = gm > np.percentile(gm, 95)
    def edge_acutance(o):
        oy, ox = np.gradient(o.mean(axis=2))
        return float(np.hypot(oy, ox)[edge_mask].mean() / gm[edge_mask].mean())
    mis = []
    mis_out = {}
    for variant in ("signal_PSD_from_correct_Snn", "signal_PSD_refit_with_assumed_Snn"):
        for k in (0.5, 1.0, 2.0):
            o = per_channel(lambda c: wiener_denoise(c / 255.0, sig01, noise_scale=k, refit_signal_with_assumed=(variant != "signal_PSD_from_correct_Snn")) * 255.0, noisy)
            fi, fj = flat_ij
            resid = o[fi:fi + bs, fj:fj + bs] - ref[fi:fi + bs, fj:fj + bs]
            mis.append({"variant": variant, "assumed_noise_power_factor": k, **metrics(ref, o),
                        "flat_patch_residual_std_(vs_clean)": float(resid.std()), "edge_gradient_retention": edge_acutance(o)})
            mis_out[(variant, k)] = o
    R["q5_2_misestimation"] = {"sigma": s, "rows": mis, "flat_patch_(i,j,size)": [flat_ij[0], flat_ij[1], bs],
                               "noisy_flat_patch_std": float(((noisy - ref)[flat_ij[0]:flat_ij[0] + bs, flat_ij[1]:flat_ij[1] + bs]).std()),
                               "edge_mask": "top 5% clean-gradient pixels of luminance"}
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.9))
    for a, (t, im) in zip(ax, [("noisy 20", np.clip(noisy, 0, 255)), ("assumed 0.5 Snn", mis_out[("signal_PSD_from_correct_Snn", 0.5)]), ("correct Snn", mis_out[("signal_PSD_from_correct_Snn", 1.0)]),
                               ("assumed 2 Snn", mis_out[("signal_PSD_from_correct_Snn", 2.0)]), ("clean", ref)]):
        a.imshow(np.clip(im[cs[0], cs[1]], 0, 255).astype(np.uint8)); a.set_title(t); a.axis("off")
    C.savefig(plt, fig, "q5_misestimation.png")
    # Wiener effective spatial kernel (centre row) for sigma 20
    _, parts = wiener_denoise(noisy[..., 1] / 255.0, sig01, return_parts=True)
    k = np.real(np.fft.ifft2(parts["W"]))
    kc = np.fft.fftshift(k)
    cr, cc_ = kc.shape[0] // 2, kc.shape[1] // 2
    R["q5_2_wiener_kernel_sigma20"] = {"sum": float(k.sum()), "peak": float(k.max()), "min": float(k.min()), "|min|/max": float(abs(k.min()) / k.max()),
                                       "half_max_width_px": float(np.sum(kc[cr, :] >= 0.5 * k.max()))}

    # ================= 5.2 real photos
    photos = {}
    for key in ("A", "B"):
        photos[key] = C.load_rgb(os.path.join(C.PHOTOS, C.PHOTO_FILES[key]))
    real = {}
    for key, img in photos.items():
        info = {"shape": list(img.shape), "channel_means": [float(img[..., c].mean()) for c in range(3)]}
        ch_rows = []
        # patch noise per channel across the whole frame (native resolution)
        for c, nm in enumerate("RGB"):
            rows = patch_noise(img[..., c], patch=24)
            # flat candidates: lowest 10% Laplacian energy
            thr = np.percentile(rows[:, 2], 10)
            flat = rows[rows[:, 2] <= thr]
            ch_rows.append({"channel": nm, "n_patches": int(len(rows)), "n_flat": int(len(flat)),
                            "median_sigma_flat_patches": float(np.median(flat[:, 0])), "mean_level_of_flat_patches": float(np.median(flat[:, 1])),
                            "sigma_flat_p25_p75": [float(np.percentile(flat[:, 0], 25)), float(np.percentile(flat[:, 0], 75))],
                            "immerkaer_whole_frame": Q4.noise_sigma_immerkaer(img[..., c])})
            if nm == "G":
                nlf = rows
        info["channels"] = ch_rows
        # noise-level function of luminance-ish green channel: median sigma in level bins (flat candidates)
        bins = [(8, 30), (30, 60), (60, 100), (100, 150), (150, 245)]
        g_rows = nlf[nlf[:, 2] <= np.percentile(nlf[:, 2], 25)]
        info["NLF_green_(flattest_25%_patches)"] = [{"level_range": b, "n": int(((g_rows[:, 1] >= b[0]) & (g_rows[:, 1] < b[1])).sum()),
                                                      "median_sigma": float(np.median(g_rows[(g_rows[:, 1] >= b[0]) & (g_rows[:, 1] < b[1]), 0])) if ((g_rows[:, 1] >= b[0]) & (g_rows[:, 1] < b[1])).sum() else None}
                                                     for b in bins]
        real[key] = info
        photos_nlf = g_rows
        real[key]["_nlf"] = g_rows[:, :2]
    ratios = []
    for ba, bb in zip(real["A"]["NLF_green_(flattest_25%_patches)"], real["B"]["NLF_green_(flattest_25%_patches)"]):
        if ba["median_sigma"] and bb["median_sigma"]:
            ratios.append({"level_range": ba["level_range"], "sigma_A": ba["median_sigma"], "sigma_B": bb["median_sigma"], "ratio_B/A": bb["median_sigma"] / ba["median_sigma"],
                           "nA": ba["n"], "nB": bb["n"]})
    real["matched_level_ratio_B_over_A"] = ratios
    R["q5_2_real_noise"] = {k: ({kk: vv for kk, vv in v.items() if kk != "_nlf"} if isinstance(v, dict) else v) for k, v in real.items()}

    # whiteness of real noise in flat patches (NPS) and Wiener on a native-resolution crop of photo B
    wiener_real = {}
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.6))
    crop_rc = {"A": (700, 1724, 1500, 2524), "B": (700, 1724, 1500, 2524)}
    for row, key in enumerate(("A", "B")):
        img = photos[key]
        r0_, r1_, c0_, c1_ = crop_rc[key]
        cp = img[r0_:r1_, c0_:c1_]
        sig_c = [Q4.noise_sigma_immerkaer(cp[..., c]) for c in range(3)]
        # patch-based sigma on the crop (flat 10 %)
        sig_p = []
        for c in range(3):
            rows = patch_noise(cp[..., c], patch=24)
            thr = np.percentile(rows[:, 2], 10)
            sig_p.append(float(np.median(rows[rows[:, 2] <= thr][:, 0])))
        # use the patch-based flat-region estimate (primary) per channel
        den = np.stack([wiener_denoise(cp[..., c] / 255.0, sig_p[c] / 255.0) * 255.0 for c in range(3)], -1)
        resid = cp - den
        # NPS of flat patches (green)
        rows = patch_noise(cp[..., 1], patch=32)
        thr = np.percentile(rows[:, 2], 5)
        sel = rows[rows[:, 2] <= thr]
        stack = [cp[int(i):int(i) + 32, int(j):int(j) + 32, 1] for (_, _, _, i, j) in sel[:60]]
        acc, prof = nps_radial(stack)
        wiener_real[key] = {"crop_(r0,r1,c0,c1)": [r0_, r1_, c0_, c1_], "sigma_immerkaer_RGB": sig_c, "sigma_flat_patch_RGB": sig_p,
                            "NPS_radial_8bands_(green,flat_patches)_normalised_to_mean": [float(x) for x in prof / prof.mean()],
                            "NPS_low_over_high_band_ratio": float(prof[0] / prof[-1]),
                            "n_flat_patches_used": len(stack),
                            "residual_std_RGB_(noisy-denoised)": [float(resid[..., c].std()) for c in range(3)],
                            "residual_std_over_sigma_est": [float(resid[..., c].std() / sig_p[c]) for c in range(3)],
                            "mean_abs_change": float(np.abs(resid).mean())}
        ax[row, 0].imshow(np.clip(cp, 0, 255).astype(np.uint8)); ax[row, 0].set_title("photo %s crop (native resolution, 1024x1024)" % key)
        z_ = (slice(250, 450), slice(300, 600))
        ax[row, 1].imshow(np.clip(cp[z_], 0, 255).astype(np.uint8)); ax[row, 1].set_title("noisy zoom")
        ax[row, 2].imshow(np.clip(den[z_], 0, 255).astype(np.uint8)); ax[row, 2].set_title("white-noise Wiener (sigma_hat per channel %s)" % [round(x, 1) for x in sig_p])
        ax[row, 3].imshow(np.clip(128 + 4 * resid[z_], 0, 255).astype(np.uint8)); ax[row, 3].set_title("residual noisy - denoised, x4 gain")
        for a in ax[row]: a.axis("off")
        Image.fromarray(np.clip(den, 0, 255).astype(np.uint8)).save(os.path.join(C.DERIVED, "photo%s_crop_wiener.png" % key))
        Image.fromarray(np.clip(cp, 0, 255).astype(np.uint8)).save(os.path.join(C.DERIVED, "photo%s_crop_native.png" % key))
    C.savefig(plt, fig, "q5_real_photos.png")
    R["q5_2_real_wiener"] = wiener_real
    # NLF + NPS figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, col in (("A", "C0"), ("B", "C3")):
        nl = real[key]["_nlf"]
        ax[0].scatter(nl[:, 1], nl[:, 0], s=4, alpha=0.4, c=col, label="photo %s (ISO %s)" % (key, "734" if key == "A" else "6400"))
    ax[0].set_xlabel("patch mean level (green, 0-255)"); ax[0].set_ylabel("detrended patch std"); ax[0].set_title("noise vs signal level (flattest 25% of 24x24 patches)"); ax[0].legend()
    for key, col in (("A", "C0"), ("B", "C3")):
        p = np.array(wiener_real[key]["NPS_radial_8bands_(green,flat_patches)_normalised_to_mean"])
        ax[1].plot((np.arange(8) + 0.5) / 8 * 0.5, p, "o-", c=col, label="photo %s" % key)
    ax[1].axhline(1.0, color="k", ls=":"); ax[1].set_xlabel("radial frequency (cycles/pixel)"); ax[1].set_ylabel("NPS / mean (white = 1)"); ax[1].set_title("noise power spectrum of flat patches"); ax[1].legend()
    C.savefig(plt, fig, "q5_real_noise_analysis.png")

    C.save_json(R, "q5.json")
    print("q5 done")


if __name__ == "__main__":
    main()
