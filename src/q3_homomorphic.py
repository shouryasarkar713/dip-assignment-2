"""Question 3 - homomorphic filtering of an unevenly lit photograph (provided tunnel image).

Model I = L * R  ->  ln I = ln L + ln R; filter ln(I+eps) with a Gaussian high-frequency-emphasis
H = (gH-gL)[1 - exp(-c D^2/D0^2)] + gL (D in cycles/pixel on the padded grid), exponentiate back.
Processing is applied to the luminance (BT.601 Y); colour is restored by the per-pixel ratio Y_out/Y_in.
Run: python src/q3_homomorphic.py -> results/q3.json, figures/q3_*.png
"""
import os

import numpy as np
from scipy.fft import next_fast_len

import common as C

EPS = 1.0 / 255.0          # offset so that ln(0) is never taken (documented in the report)
FACTOR = 3                 # 3000x3644 -> 1000x1214 block average


def block_mean(a, f):
    h, w = a.shape[:2]
    h2, w2 = h // f * f, w // f * f
    a = a[:h2, :w2]
    return a.reshape(h2 // f, f, w2 // f, f, *a.shape[2:]).mean(axis=(1, 3))


def hom_filter_grid(shape, gL, gH, D0, c=1.0):
    """H(u,v) on the (unshifted) FFT grid of the given padded shape; D in cycles/pixel."""
    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.fftfreq(shape[1])
    D2 = fy[:, None] ** 2 + fx[None, :] ** 2
    return (gH - gL) * (1.0 - np.exp(-c * D2 / D0 ** 2)) + gL


def homomorphic(Y01, gL, gH, D0, c=1.0, eps=EPS):
    """Y01 in [0,1].  Returns exp(filtered log image) - eps (un-normalised)."""
    M, N = Y01.shape
    Pr, Pc = next_fast_len(2 * M), next_fast_len(2 * N)
    pr0, pc0 = (Pr - M) // 2, (Pc - N) // 2
    z = np.log(Y01 + eps)
    zp = np.pad(z, ((pr0, Pr - M - pr0), (pc0, Pc - N - pc0)), mode="symmetric")   # mirror padding: no artificial edge
    H = hom_filter_grid(zp.shape, gL, gH, D0, c)
    zf = np.real(np.fft.ifft2(np.fft.fft2(zp) * H))[pr0:pr0 + M, pc0:pc0 + N]
    return np.exp(zf) - eps


def normalise(out, pct=99.9):
    return np.clip(out / np.percentile(out, pct), 0, 1)


def hist_equalise(Y01):
    """Global histogram equalisation of an 8-bit quantised luminance (classic cdf mapping)."""
    q = np.clip(np.round(Y01 * 255), 0, 255).astype(int)
    hist = np.bincount(q.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    cdf_min = cdf[hist > 0][0]
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255) / 255.0
    return lut[q]


def box_mean(x, w):
    """Mean over a w x w window ('valid' region) via an integral image."""
    ii = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]) / (w * w)


def local_std_map(x, w=9):
    m = box_mean(x, w)
    return np.sqrt(np.maximum(box_mean(x * x, w) - m * m, 0))


def region_stats(Y, box, b_disc, eps=EPS):
    r0, r1, c0, c1 = box
    sub = Y[r0:r1, c0:c1]
    ls = local_std_map(sub, 9)
    lm = box_mean(sub, 9)
    lls = local_std_map(np.log(sub + eps), 9)
    return {"mean_level_(0-255)": float(255 * sub.mean()), "local_std_9x9_(0-255)": float(255 * ls.mean()),
            "local_CV_std/mean": float((ls / (lm + 1e-3)).mean()), "local_std_of_ln(Y+eps)": float(lls.mean())}


def radial_profile_stats(Z, cy, cx, Re, bin_w=2.0):
    """Annular mean of z=ln(Y+eps) about the aperture centre; overshoot inside / undershoot outside the edge,
    both relative to the plateau-to-surround step S."""
    yy, xx = np.indices(Z.shape)
    rr = np.hypot(yy - cy, xx - cx)
    nb = int(rr.max() / bin_w) + 1
    idx = (rr / bin_w).astype(int)
    prof = np.bincount(idx.ravel(), weights=Z.ravel(), minlength=nb) / np.maximum(np.bincount(idx.ravel(), minlength=nb), 1)
    rs = (np.arange(nb) + 0.5) * bin_w
    sel = lambda a, b: (rs >= a * Re) & (rs < b * Re)
    plateau = prof[sel(0, 0.4)].mean()
    surround = prof[sel(1.6, 2.0)].mean()
    S = plateau - surround
    inner = (prof[sel(0.6, 1.0)].max() - plateau) / S
    outer = (surround - prof[sel(1.0, 1.4)].min()) / S
    return {"step_S_(ln_units)": float(S), "inner_overshoot/S": float(inner), "outer_undershoot/S": float(outer)}, rs, prof


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "eps": EPS, "downsample_factor": FACTOR}
    rgb = C.load_rgb(os.path.join(C.PROVIDED, "tunnel_image.jpg")) / 255.0
    R["source"] = {"file": "data/provided/tunnel_image.jpg", "shape_HxWxC": list(rgb.shape)}
    rgb = block_mean(rgb, FACTOR)
    H0, W0 = rgb.shape[:2]
    R["working_shape_HxW"] = [H0, W0]
    C.save_png(rgb * 255, os.path.join(C.DERIVED, "tunnel_work_1000x1214.png"))
    Y = C.luma(rgb)
    # --- regions (fractions of the working size), aperture from saturated pixels
    sat = Y >= 0.98
    ys, xs = np.nonzero(sat)
    cy, cx = float(ys.mean()), float(xs.mean())
    Re = float(np.sqrt(sat.sum() / np.pi))
    shadow = (int(0.615 * H0), int(0.80 * H0), int(0.18 * W0), int(0.45 * W0))      # faint tunnel wall, lower left
    corner = (int(0.03 * H0), int(0.25 * H0), int(0.03 * W0), int(0.30 * W0))        # deep black upper-left corner
    yy, xx = np.indices(Y.shape)
    bright_mask = np.hypot(yy - cy, xx - cx) <= 0.8 * Re
    R["regions"] = {"shadow_box_(r0,r1,c0,c1)": shadow, "corner_box": corner, "aperture_centre_(row,col)": [cy, cx],
                    "aperture_equiv_radius_px": Re, "saturated_fraction_of_input_%": float(100 * sat.mean()),
                    "input_luminance_percentiles_(0-255)": [float(v) for v in 255 * np.percentile(Y, [1, 10, 50, 90, 99, 99.9])]}

    def eval_img(Yo, tag):
        st, rs, prof = radial_profile_stats(np.log(Yo + EPS), cy, cx, Re)
        d = {"shadow": region_stats(Yo, shadow, None), "corner": region_stats(Yo, corner, None), "halo": st,
             "bright_region_mean_level": float(255 * Yo[bright_mask].mean()),
             "bright_region_fraction>=0.98_%": float(100 * (Yo[bright_mask] >= 0.98).mean()),
             "whole_image_fraction>=0.98_%": float(100 * (Yo >= 0.98).mean()),
             "whole_image_fraction<=0.02_%": float(100 * (Yo <= 0.02).mean()),
             "shadow_mean/bright_mean": float(Yo[shadow[0]:shadow[1], shadow[2]:shadow[3]].mean() / Yo[bright_mask].mean())}
        hist = np.bincount(np.clip(np.round(Yo * 255), 0, 255).astype(int).ravel(), minlength=256) / Yo.size
        d["entropy_bits"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
        return d, (rs, prof)

    base, prof_in = eval_img(Y, "input")
    R["input"] = base
    Yhe = hist_equalise(Y)
    he, prof_he = eval_img(Yhe, "HE")
    R["hist_equalisation"] = he

    # --- parameter grid
    gLs, gHs, D0s = (0.25, 0.5, 0.75), (1.5, 2.5, 4.0), (0.005, 0.02, 0.08)
    grid = []
    outs = {}
    for D0 in D0s:
        for gL in gLs:
            for gH in gHs:
                o = normalise(homomorphic(Y, gL, gH, D0))
                outs[(gL, gH, D0)] = o
                d, _ = eval_img(o, "hom")
                d.update({"gL": gL, "gH": gH, "D0_cycles_per_pixel": D0,
                          "D0_pixels_period": 1 / D0, "dc_gain": gL, "hf_gain": gH})
                grid.append(d)
    R["grid"] = grid
    # selection rules (stated in the report)
    thr_halo = 0.10
    for d in grid:
        d["halo_flag_(inner or outer > %.2f)" % thr_halo] = bool(max(d["halo"]["inner_overshoot/S"], d["halo"]["outer_undershoot/S"]) > thr_halo)
    ok = [d for d in grid if not d["halo_flag_(inner or outer > %.2f)" % thr_halo] and d["whole_image_fraction>=0.98_%"] < 5.0
          and d["shadow_mean/bright_mean"] >= 0.25]            # must flatten: input ratio is 0.073
    best = max(ok, key=lambda d: d["shadow"]["local_std_of_ln(Y+eps)"])
    halo_cases = sorted(grid, key=lambda d: -max(d["halo"]["inner_overshoot/S"], d["halo"]["outer_undershoot/S"]))[:3]
    R["selected_best_(max_shadow_log-contrast_subject_to_no_halo_flag,_<5%_saturation,_shadow/bright_ratio>=0.25)"] = {k: best[k] for k in ("gL", "gH", "D0_cycles_per_pixel")}
    R["halo_cases_top3"] = [{k: d[k] for k in ("gL", "gH", "D0_cycles_per_pixel")} | {"halo": d["halo"]} for d in halo_cases]
    gb = (best["gL"], best["gH"], best["D0_cycles_per_pixel"])
    Yb = outs[gb]
    # a flattening-oriented reference setting that also keeps local contrast
    R["best_vs_he_vs_input"] = {"input": base, "he": he, "homomorphic_best": eval_img(Yb, "best")[0], "setting": gb}

    # --- colour outputs
    def recolour(Yo):
        return np.clip(rgb * (Yo / (Y + EPS))[..., None], 0, 1)
    C.save_png(recolour(Yb) * 255, os.path.join(C.DERIVED, "tunnel_homomorphic_best.png"))
    C.save_png(recolour(Yhe) * 255, os.path.join(C.DERIVED, "tunnel_histeq.png"))

    # --- figures: three grids (one per D0)
    for D0 in D0s:
        fig, ax = plt.subplots(3, 3, figsize=(8.4, 10))
        for i, gL in enumerate(gLs):
            for j, gH in enumerate(gHs):
                d = [g for g in grid if g["gL"] == gL and g["gH"] == gH and g["D0_cycles_per_pixel"] == D0][0]
                ax[i, j].imshow(outs[(gL, gH, D0)], cmap="gray", vmin=0, vmax=1)
                flag = "HALO" if d["halo_flag_(inner or outer > %.2f)" % thr_halo] else ""
                ax[i, j].set_title("gL=%.2f gH=%.1f  %s\nshadow ln-contrast %.3f" % (gL, gH, flag, d["shadow"]["local_std_of_ln(Y+eps)"]), fontsize=7.5,
                                   color="red" if flag else "black")
                ax[i, j].axis("off")
                r0, r1, c0, c1 = shadow
                ax[i, j].add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="cyan", lw=0.8))
        fig.suptitle("Homomorphic grid, D0 = %g cycles/pixel (period %.0f px), c=1" % (D0, 1 / D0), fontsize=9)
        C.savefig(plt, fig, "q3_grid_D0_%s.png" % str(D0).replace(".", "p"))
    # comparison figure
    fig, ax = plt.subplots(2, 3, figsize=(12, 9.4))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("input")
    ax[0, 1].imshow(recolour(Yhe)); ax[0, 1].set_title("global histogram equalisation (luminance)")
    ax[0, 2].imshow(recolour(Yb)); ax[0, 2].set_title("homomorphic best: gL=%.2f gH=%.1f D0=%g" % gb)
    hal = halo_cases[0]
    ax[1, 0].imshow(recolour(outs[(hal["gL"], hal["gH"], hal["D0_cycles_per_pixel"])])); ax[1, 0].set_title("strongest halo: gL=%.2f gH=%.1f D0=%g" % (hal["gL"], hal["gH"], hal["D0_cycles_per_pixel"]))
    r0, r1, c0, c1 = shadow
    for a_, im in zip(ax.ravel()[:4], (rgb, recolour(Yhe), recolour(Yb), recolour(outs[(hal["gL"], hal["gH"], hal["D0_cycles_per_pixel"])]))):
        a_.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="cyan", lw=1))
    sh = lambda im: im[r0:r1, c0:c1]
    g_ = lambda a: np.clip(a * (0.6 / max(a.mean(), 1e-3)), 0, 1)       # mean-normalised view of shadow crop so contrast, not level, is compared
    ax[1, 1].imshow(np.hstack([g_(sh(Y)), g_(sh(Yhe)), g_(sh(Yb))]), cmap="gray", vmin=0, vmax=1); ax[1, 1].set_title("shadow box (mean-normalised): input | HE | homomorphic")
    ax[1, 2].plot(prof_in[0] / Re, prof_in[1], label="input"); ax[1, 2].plot(prof_he[0] / Re, prof_he[1], label="HE")
    st_b, rs_b, pf_b = radial_profile_stats(np.log(Yb + EPS), cy, cx, Re)
    ax[1, 2].plot(rs_b / Re, pf_b, label="homomorphic best")
    st_h, rs_h, pf_h = radial_profile_stats(np.log(outs[(hal["gL"], hal["gH"], hal["D0_cycles_per_pixel"])] + EPS), cy, cx, Re)
    ax[1, 2].plot(rs_h / Re, pf_h, label="halo case"); ax[1, 2].set_xlim(0, 3); ax[1, 2].set_xlabel("r / aperture radius"); ax[1, 2].set_ylabel("annular mean of ln(Y+eps)"); ax[1, 2].legend(fontsize=7)
    for a_ in ax.ravel()[:4]: a_.axis("off")
    ax[1, 1].axis("off")
    C.savefig(plt, fig, "q3_comparison.png")
    # spectrum of log image (for the log-before-FFT discussion)
    z = np.log(Y + EPS)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(z, cmap="gray"); ax[0].set_title("ln(Y+eps)")
    Fz = np.fft.fftshift(np.abs(np.fft.fft2(z - z.mean())))
    Fi = np.fft.fftshift(np.abs(np.fft.fft2(Y - Y.mean())))
    ax[1].imshow(np.log1p(Fi), cmap="magma"); ax[1].set_title("log|F| of Y (linear domain)")
    ax[2].imshow(np.log1p(Fz), cmap="magma"); ax[2].set_title("log|F| of ln Y")
    for a_ in ax: a_.axis("off")
    C.savefig(plt, fig, "q3_log_spectra.png")
    # filter shape
    Hc = hom_filter_grid((64, 64), 0.5, 2.5, 0.02)
    fx = np.linspace(0, 0.2, 400)
    fig, ax = plt.subplots(figsize=(5, 3))
    for D0 in (0.005, 0.02, 0.08):
        ax.plot(fx, (2.5 - 0.5) * (1 - np.exp(-fx ** 2 / D0 ** 2)) + 0.5, label="D0=%g" % D0)
    ax.set_xlabel("radial frequency (cycles/pixel)"); ax.set_ylabel("H"); ax.set_title("gL=0.5, gH=2.5, c=1"); ax.legend()
    C.savefig(plt, fig, "q3_filter_profile.png")

    # --- numerical checks of the model: the log turns products into sums; linearity of the filter in log domain
    rng = np.random.default_rng(2026)
    Lm = np.exp(np.linspace(-3, 0, 256))[:, None] * np.ones((1, 256))        # slowly varying illumination
    Rm = np.exp(0.3 * rng.standard_normal((256, 256)))
    zmodel = np.log(Lm * Rm)
    R["model_check"] = {"max|ln(L*R)-(ln L+ln R)|": float(np.abs(zmodel - (np.log(Lm) + np.log(Rm))).max())}
    C.save_json(R, "q3.json")
    print("q3 done")


if __name__ == "__main__":
    main()
