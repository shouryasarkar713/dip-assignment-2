"""Question 6 - motion blur synthesis, naive inverse, Wiener deconvolution, blur estimation on real photos.

Coordinates: x = column index (right), y = row index (down); theta is measured from +x towards +y (clockwise on screen).
The PSF is the unit-mass line segment of length L through the origin, discretised by midpoint sampling of t in [-L/2, L/2]
and bilinear splatting onto the (periodic) pixel grid.  Boundary convention: circular, for blur AND deconvolution.
Frequencies: fx, fy in cycles/pixel, OTF H(fx,fy)=sinc(L(fx cos(theta) + fy sin(theta))), sinc(z)=sin(pi z)/(pi z).
Run: python src/q6_motion.py -> results/q6.json, figures/q6_*.png
"""
import math
import os

import numpy as np
from PIL import Image

import common as C
import q4_edge_preserving as Q4
import q5_wiener as Q5

SEED = 2026
L_TRUE, THETA_TRUE = 25.0, 30.0
SNR_DB = (20, 30, 40, 50)


# ------------------------------------------------------------------ model
def motion_psf(L, theta_deg, shape, oversample=32):
    th = math.radians(theta_deg)
    n = max(int(math.ceil(L * oversample)), 2)
    t = (np.arange(n) + 0.5) / n * L - L / 2.0           # midpoint rule over [-L/2, L/2]
    x, y = t * math.cos(th), t * math.sin(th)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    M, N = shape
    psf = np.zeros(shape)
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        np.add.at(psf, ((y0 + dy) % M, (x0 + dx) % N), w / n)
    return psf


def otf_analytic(L, theta_deg, shape):
    """Continuous-PSF OTF sampled on the DFT grid (fftfreq ordering): axis 0 <-> fy, axis 1 <-> fx."""
    th = math.radians(theta_deg)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.sinc(L * (fx * math.cos(th) + fy * math.sin(th)))


def interp_factor(shape):
    """Frequency response of the bilinear splatting (triangle) kernel: sinc^2(fx) sinc^2(fy)."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.sinc(fx) ** 2 * np.sinc(fy) ** 2


def blur_circular(img2d, H):
    return np.fft.irfft2(np.fft.rfft2(img2d) * H[:, :img2d.shape[1] // 2 + 1], s=img2d.shape)


def rH(H, shape):
    return H[:, :shape[1] // 2 + 1]


def inverse_filter(G_hat, H, floor=1e-8):
    """F = G/H with |H| floored at `floor` (sign preserved; H is real for the point-symmetric PSF)."""
    Hs = np.where(np.abs(H) < floor, np.where(H < 0, -floor, floor), H)
    return G_hat / Hs


def wiener_deconv(g, H, K):
    """F = conj(H) G / (|H|^2 + K) for a real image g (rfft2), circular boundary."""
    Hh = rH(H, g.shape)
    return np.fft.irfft2(np.conj(Hh) * np.fft.rfft2(g) / (np.abs(Hh) ** 2 + K), s=g.shape)


def snr_noise_sigma(f, snr_db):
    """sigma of white noise so that 10log10(sum f^2 / sum n^2) = snr_db (expected value, per element)."""
    return math.sqrt(np.sum(f ** 2) / f.size / 10 ** (snr_db / 10.0))


# ------------------------------------------------------------------ blur estimation (cepstrum + spectral comb)
def hann2(shape):
    return np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]


def flat_log_spectrum(gray, nb=128):
    """log power spectrum of the Hann-windowed, mean-removed image, divided by its radial mean (spectral whitening)."""
    g = (gray - gray.mean()) * hann2(gray.shape)
    P = np.abs(np.fft.fft2(g)) ** 2
    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.fftfreq(gray.shape[1])[None, :]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    idx = np.minimum((rho / 0.5 * nb).astype(int), nb)          # radial bins
    s = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nb + 1)
    n = np.bincount(idx.ravel(), minlength=nb + 1)
    rad = s / np.maximum(n, 1)
    S = np.log(P + 1e-12 * P.max()) - np.log(rad[idx] + 1e-30)
    return S, P, rho


def cepstrum_estimate(gray, lmin=4, lmax=150):
    """Power-cepstrum estimate of (L, theta): strongest negative peak of the background-subtracted cepstrum
    at quefrency radius in [lmin, lmax].  Returns dict with L, theta (deg in [0,180)), z-score of the peak."""
    g = (gray - gray.mean()) * hann2(gray.shape)
    logP = np.log(np.abs(np.fft.fft2(g)) ** 2 + 1e-12)
    # exclude the lowest frequencies, which carry scene structure, by flattening with the radial mean (as above)
    S, _, rho = flat_log_spectrum(gray)
    S = S.copy()
    S[rho < 0.02] = 0.0
    c = np.fft.fftshift(np.real(np.fft.ifft2(S)))
    cy, cx = np.array(c.shape) // 2
    yy, xx = np.indices(c.shape)
    q = np.hypot(yy - cy, xx - cx)
    ann = (q >= lmin) & (q <= lmax)
    # remove the smooth radial trend of the cepstrum (median per integer radius)
    qi = np.round(q).astype(int)
    med = np.zeros(qi.max() + 1)
    for r in range(lmin, min(lmax + 1, len(med))):
        med[r] = np.median(c[qi == r])
    cd = c - med[np.minimum(qi, len(med) - 1)]
    vals = cd[ann]
    i = np.argmin(np.where(ann, cd, np.inf))
    py, px = np.unravel_index(i, c.shape)
    # sub-pixel refinement (parabola along each axis)
    def para(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    dy = para(cd[py - 1, px], cd[py, px], cd[py + 1, px])
    dx = para(cd[py, px - 1], cd[py, px], cd[py, px + 1])
    ey, ex = py + dy - cy, px + dx - cx
    L = math.hypot(ey, ex)
    th = math.degrees(math.atan2(ey, ex)) % 180.0
    z = float((cd[py, px] - vals.mean()) / vals.std())
    return {"L": L, "theta_deg": th, "peak_z_score": z, "peak_value": float(cd[py, px]), "n_annulus": int(ann.sum())}, cd


class CombCtx:
    """Spectral-comb scorer on a fixed random subsample (<= nmax) of the band 0.03 < |f| < 0.4 cycles/pixel of the whitened
    log-spectrum S (the subsample is seeded, so scores are deterministic)."""

    def __init__(self, S, nmax=60000, seed=0):
        fy = np.fft.fftfreq(S.shape[0])[:, None]
        fx = np.fft.fftfreq(S.shape[1])[None, :]
        rho = np.sqrt(fx ** 2 + fy ** 2)
        idx = np.flatnonzero(((rho > 0.03) & (rho < 0.4)).ravel())
        if len(idx) > nmax:
            idx = np.random.default_rng(seed).choice(idx, nmax, replace=False)
        self.fx = np.broadcast_to(fx, S.shape).ravel()[idx]
        self.fy = np.broadcast_to(fy, S.shape).ravel()[idx]
        self.s = S.ravel()[idx]

    def score(self, L, theta_deg):
        """Mean whitened log-spectrum on the predicted zero lines (L (fx cos + fy sin) = integer m != 0) minus the mean on
        the half-way lines: NEGATIVE when dark stripes sit on the predicted lines."""
        th = math.radians(theta_deg)
        u = L * (self.fx * math.cos(th) + self.fy * math.sin(th))
        frac = np.abs(u - np.round(u))                        # 0 on a zero line, 0.5 half-way
        big = np.abs(u) >= 0.9
        on, off = big & (frac < 0.08), big & (frac > 0.42)
        if on.sum() < 50 or off.sum() < 50:
            return 0.0
        return float(self.s[on].mean() - self.s[off].mean())


def comb_score(S, L, theta_deg):
    return CombCtx(S).score(L, theta_deg)


def refine_comb(S, L0, th0, dL=0.5, dth=3.0, ctx=None):
    ctx = CombCtx(S) if ctx is None else ctx
    best = (ctx.score(L0, th0), L0, th0)
    for th in np.arange(th0 - dth, th0 + dth + 1e-9, 0.25):
        for L in np.arange(L0 * (1 - 0.15), L0 * (1 + 0.15) + 1e-9, dL):
            sc = ctx.score(L, th % 180)
            if sc < best[0]:
                best = (sc, L, th % 180)
    return best


def estimate_blur(gray, lmin=4, lmax=150):
    est, cd = cepstrum_estimate(gray, lmin, lmax)
    S, P, rho = flat_log_spectrum(gray)
    ctx = CombCtx(S)
    sc, L2, th2 = refine_comb(S, est["L"], est["theta_deg"], ctx=ctx)
    est["comb_refined"] = {"L": float(L2), "theta_deg": float(th2), "score": float(sc), "score_at_cepstrum_estimate": ctx.score(est["L"], est["theta_deg"])}
    est["_ctx"] = ctx
    return est, S, cd


def angdiff(a, b):
    d = (a - b + 90) % 180 - 90
    return d


def main():
    plt = C.fig_style()
    C.ensure_dirs()
    R = {"env": C.env_info(), "seed": SEED, "conventions": "x=column,y=row(down); theta from +x towards +y; circular boundary; sinc(z)=sin(pi z)/(pi z)"}
    ref = C.load_rgb(os.path.join(C.DERIVED, "reference_work_1536x1024.png")) / 255.0 if os.path.exists(os.path.join(C.DERIVED, "reference_work_1536x1024.png")) else None
    if ref is None:
        raise SystemExit("run q5_wiener.py first (creates data/derived/reference_work_1536x1024.png)")
    M, N, _ = ref.shape
    shape = (M, N)

    # ---------------- 6.1 PSF / OTF checks
    psf = motion_psf(L_TRUE, THETA_TRUE, shape)
    H = np.fft.fft2(psf)
    Ha = otf_analytic(L_TRUE, THETA_TRUE, shape)
    Hi = Ha * interp_factor(shape)
    # axis aligned integer L: exact box (weights 1/L at integer pixels would need half-weights at the ends -> Dirichlet check)
    psf0 = motion_psf(8.0, 0.0, (64, 64), oversample=64)
    H0 = np.fft.fft2(psf0)
    fx0 = np.fft.fftfreq(64)
    zeros_axis = [float(abs(H0[0, k])) for k in (8, 16, 24)]       # f = m/L with L=8 -> k = 64*m/8 = 8m
    fyg = np.fft.fftfreq(M)[:, None]
    fxg = np.fft.fftfreq(N)[None, :]
    alias = np.zeros(shape)
    for ky in range(-4, 5):                                   # DFT of the splatted PSF = sum over aliases of sinc x triangle
        for kx in range(-4, 5):
            xx_, yy_ = fxg - kx, fyg - ky
            alias = alias + np.sinc(L_TRUE * (xx_ * math.cos(math.radians(THETA_TRUE)) + yy_ * math.sin(math.radians(THETA_TRUE)))) * np.sinc(xx_) ** 2 * np.sinc(yy_) ** 2
    R["q6_1_psf"] = {
        "max|H-aliased_sum(sinc*triangle)|_(+-4_aliases)": float(np.abs(H - alias).max()),
        "max|H-Ha*interp|_(k=0_term_only)_|f|<0.1": float(np.abs(H - Hi)[np.sqrt(fxg ** 2 + fyg ** 2) < 0.1].max()),
        "L": L_TRUE, "theta_deg": THETA_TRUE, "sum": float(psf.sum()), "min": float(psf.min()), "support_pixels": int((psf > 0).sum()),
        "max_|imag(H)|": float(np.abs(H.imag).max()),
        "max|H-Ha*interp|_(analytic_sinc_times_triangle_factor)": float(np.abs(H - Hi).max()),
        "max|H-Ha|_(raw_sinc)": float(np.abs(H - Ha).max()),
        "rel_max|H-Ha|_in_|f|<0.1": float(np.abs(H - Ha)[(np.sqrt(np.fft.fftfreq(M)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2) < 0.1)].max()),
        "axis_aligned_L=8_theta=0_|H|_at_f=m/8_(m=1,2,3)": zeros_axis,
        "H_dc": float(H[0, 0].real)}
    # zero lines: predicted spacing 1/L along the direction cosine; verify with the min of |H| along a ray
    th = math.radians(THETA_TRUE)
    ray_f = np.linspace(0.005, 0.45, 4000)
    # sample analytic and discrete H along direction (cos, sin) by nearest grid point
    iy = np.round(ray_f * math.sin(th) * M).astype(int) % M
    ix = np.round(ray_f * math.cos(th) * N).astype(int) % N
    ray_H = np.abs(H.real[iy, ix])
    pred = np.arange(1, int(0.45 * L_TRUE) + 1) / L_TRUE
    dev = []
    for z0 in pred:
        w = (ray_f > z0 - 0.4 / L_TRUE) & (ray_f < z0 + 0.4 / L_TRUE)
        j = np.argmin(ray_H[w])
        dev.append(float(ray_f[w][j] - z0))
    R["q6_1_psf"]["ray_zero_positions_pred_m/L"] = [float(p) for p in pred]
    R["q6_1_psf"]["ray_zero_position_error_(cycles/pixel)"] = dev
    R["q6_1_psf"]["adjacent_zero_spacing_pred_1/L"] = 1.0 / L_TRUE

    # ---------------- 6.2 / 6.3 synthetic experiment
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(ref.shape)
    blurred = np.stack([blur_circular(ref[..., c], H) for c in range(3)], -1)
    R["synthetic"] = {"image": "reference_work_1536x1024 (0..1)", "blur": {"L": L_TRUE, "theta_deg": THETA_TRUE},
                      "blurred_noise_free_PSNR_vs_clean": C.psnr(ref * 255, blurred * 255), "blurred_noise_free_SSIM": C.ssim(ref * 255, blurred * 255)}
    sig_sum = np.sum(ref ** 2)
    levels = {}
    Ks = 10.0 ** np.arange(-8.0, 0.001, 0.25)
    Kfine = 10.0 ** np.arange(-8.0, 0.001, 0.05)
    sweeps = {}
    naive = {}
    wiener_best = {}
    for snr in SNR_DB:
        sg = snr_noise_sigma(ref, snr)
        noisy = blurred + sg * z
        achieved = 10 * math.log10(sig_sum / np.sum((sg * z) ** 2))
        # naive inverse with floor 1e-8 (effectively none) and with floor 1e-3
        out_raw = np.stack([np.fft.irfft2(inverse_filter(np.fft.rfft2(noisy[..., c]), rH(H, shape), 1e-8), s=shape) for c in range(3)], -1)
        out_f3 = np.stack([np.fft.irfft2(inverse_filter(np.fft.rfft2(noisy[..., c]), rH(H, shape), 1e-3), s=shape) for c in range(3)], -1)
        # Wiener sweep
        Gs = [np.fft.rfft2(noisy[..., c]) for c in range(3)]
        Hh = rH(H, shape)
        A2 = np.abs(Hh) ** 2
        cur = []
        for K in Kfine:
            o = np.stack([np.fft.irfft2(np.conj(Hh) * Gs[c] / (A2 + K), s=shape) for c in range(3)], -1)
            cur.append(C.psnr(ref * 255, o * 255))
        cur = np.array(cur)
        i = int(np.argmax(cur))
        Kopt = float(Kfine[i])
        ob = np.stack([np.fft.irfft2(np.conj(Hh) * Gs[c] / (A2 + Kopt), s=shape) for c in range(3)], -1)
        wiener_best[snr] = ob
        sweeps[snr] = cur
        ktheory = 10 ** (-snr / 10.0)
        levels[snr] = {"sigma_noise_(0..1)": sg, "achieved_SNR_dB": achieved,
                       "noisy_blurred_PSNR": C.psnr(ref * 255, noisy * 255), "naive_inverse_floor1e-8_PSNR": C.psnr(ref * 255, out_raw * 255),
                       "naive_inverse_floor1e-3_PSNR": C.psnr(ref * 255, out_f3 * 255),
                       "naive_inverse_rms_error_(0..1,_unclipped)": float(np.sqrt(np.mean((out_raw - ref) ** 2))),
                       "K_opt": Kopt, "PSNR_at_Kopt": float(cur[i]), "SSIM_at_Kopt": C.ssim(ref * 255, ob * 255),
                       "K_theory_=10^(-SNR/10)": ktheory, "Kopt/Ktheory": Kopt / ktheory,
                       "PSNR_at_K=1e-8_(inverse-like)": float(cur[0]), "PSNR_at_K=1_(under-deconvolved)": float(cur[-1]),
                       "PSNR_at_Ktheory": float(np.interp(math.log10(ktheory), np.log10(Kfine), cur)),
                       "PSNR_at_selected_K": {("%g" % K): float(np.interp(math.log10(K), np.log10(Kfine), cur)) for K in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)}}
        if snr == 30:
            naive_out = out_raw
            noisy30 = noisy
            Gs30 = Gs
        if snr == 20:
            pass
        print("snr", snr, "Kopt", Kopt, flush=True)
    R["q6_3_levels"] = levels
    R["q6_3_K_grid"] = {"log10_K_from": -8, "to": 0, "step": 0.05}
    # noise amplification of the inverse filter: mean of 1/|H|^2 over the grid (variance gain), with and without floors
    Hh_all = np.abs(H)
    R["q6_2_noise_gain"] = {"min|H|_over_grid": float(Hh_all.min()), "frac_|H|<1e-2": float((Hh_all < 1e-2).mean()), "frac_|H|<1e-3": float((Hh_all < 1e-3).mean()),
                            "variance_gain_mean(1/|H|^2)_floor1e-3": float(np.mean(1 / np.maximum(Hh_all, 1e-3) ** 2)),
                            "variance_gain_mean(1/|H|^2)_floor1e-8": float(np.mean(1 / np.maximum(Hh_all, 1e-8) ** 2)),
                            "variance_gain_mean(1/|H|^2)_K=1e-3": float(np.mean(Hh_all ** 2 / (Hh_all ** 2 + 1e-3) ** 2))}

    # ---------------- figures: spectra with predicted zeros marked
    gl = C.luma(noisy30 * 255)
    Fg = np.fft.fft2(gl)
    Fr = np.fft.fft2(C.luma(naive_out * 255))
    def show(a, ax, title):
        im = ax.imshow(np.log10(np.abs(np.fft.fftshift(a)) + 1e-3), cmap="magma", extent=[-0.5, 0.5, 0.5, -0.5]); ax.set_title(title, fontsize=9)
        return im
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    show(Fg, ax[0], "log10|G|: blurred + noise (30 dB)")
    im = show(Fr, ax[1], "log10|G/H| raw inverse (unclipped, floor 1e-8)")
    show(np.fft.fft2(C.luma(ref * 255)), ax[2], "log10|F|: clean image")
    # predicted zero lines and nearest sampled frequencies
    thr_ = math.radians(THETA_TRUE)
    cth, sth = math.cos(thr_), math.sin(thr_)
    for a_ in ax[:2]:
        for m in range(-int(0.7 * L_TRUE), int(0.7 * L_TRUE) + 1):
            if m == 0:
                continue
            c_ = m / L_TRUE
            # line fx cos + fy sin = c : param by s along the line direction (-sin, cos)
            s_ = np.linspace(-0.7, 0.7, 2)
            fx_ = c_ * cth - s_ * sth
            fy_ = c_ * sth + s_ * cth
            a_.plot(fx_, fy_, "c-", lw=0.4, alpha=0.7)
        a_.set_xlim(-0.5, 0.5); a_.set_ylim(0.5, -0.5); a_.set_xlabel("fx (cycles/px)"); a_.set_ylabel("fy")
    # nearest grid samples to the m=+-1..+-3 zero lines
    pts = []
    for m in (-3, -2, -1, 1, 2, 3):
        c_ = m / L_TRUE
        for v in range(-N // 2, N // 2, 24):
            fx_ = v / N
            fy_ = (c_ - fx_ * cth) / sth
            u = int(round(fy_ * M))
            if abs(u) < M // 2:
                pts.append((v / N, u / M))
    pts = np.array(pts)
    ax[0].plot(pts[:, 0], pts[:, 1], "w.", ms=2.5)
    ax[1].plot(pts[:, 0], pts[:, 1], "w.", ms=2.5)
    ax[1].set_xlabel("cyan: predicted zero lines m/L (|m|<=17); white dots: sampled frequencies nearest to |m|<=3")
    C.savefig(plt, fig, "q6_spectra.png")
    # K sweep figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for snr in SNR_DB:
        ax[0].semilogx(Kfine, sweeps[snr], label="SNR %d dB" % snr)
        ax[0].plot(levels[snr]["K_opt"], levels[snr]["PSNR_at_Kopt"], "ko", ms=4)
        ax[0].axvline(levels[snr]["K_theory_=10^(-SNR/10)"], color="gray", lw=0.4, ls=":")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("PSNR vs clean (dB)"); ax[0].legend(); ax[0].set_title("Wiener deconvolution K sweep (dots: optimum; dotted: K=10^(-SNR/10))")
    ax[1].loglog([levels[s]["K_theory_=10^(-SNR/10)"] for s in SNR_DB], [levels[s]["K_opt"] for s in SNR_DB], "o-", label="K_opt (measured)")
    ax[1].loglog([levels[s]["K_theory_=10^(-SNR/10)"] for s in SNR_DB], [levels[s]["K_theory_=10^(-SNR/10)"] for s in SNR_DB], "k--", label="K = 10^(-SNR/10)")
    ax[1].set_xlabel("noise-to-signal power ratio 10^(-SNR/10)"); ax[1].set_ylabel("K"); ax[1].legend(); ax[1].set_title("optimum K vs noise level")
    C.savefig(plt, fig, "q6_ksweep.png")
    # image grid at 30 dB
    crop = (slice(380, 580), slice(500, 800))
    Hh = rH(H, shape)
    A2 = np.abs(Hh) ** 2
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.8))
    imgs_ = [("clean", ref), ("blurred+noise 30 dB", np.clip(noisy30, 0, 1)), ("naive inverse G/H", np.clip(naive_out, 0, 1))]
    for K in (1e-6, levels[30]["K_opt"], 1e-1):
        o = np.stack([np.fft.irfft2(np.conj(Hh) * Gs30[c] / (A2 + K), s=shape) for c in range(3)], -1)
        imgs_.append(("Wiener K=%.2g%s" % (K, " (optimum)" if K == levels[30]["K_opt"] else ""), np.clip(o, 0, 1)))
    imgs_.append(("Wiener optimum (full frame)", np.clip(wiener_best[30], 0, 1)))
    for a_, (t, im) in zip(ax.ravel(), imgs_[:7]):
        a_.imshow((im[crop] * 255).astype(np.uint8)); a_.set_title(t, fontsize=9); a_.axis("off")
    ax.ravel()[7].imshow((np.clip(wiener_best[30], 0, 1) * 255).astype(np.uint8)); ax.ravel()[7].set_title("Wiener optimum, full frame"); ax.ravel()[7].axis("off")
    C.savefig(plt, fig, "q6_deconv_grid.png")
    R["q6_3_grid_crop_(rows,cols)"] = [[380, 580], [500, 800]]

    # ---------------- estimator validation on synthetic blur of the sharper photo D (grey, native crop)
    D = C.load_rgb(os.path.join(C.PHOTOS, C.PHOTO_FILES["D"]))
    Dg = C.luma(D)
    CR = {"r0": 700, "c0": 1000, "size": 1024}
    cropD = Dg[CR["r0"]:CR["r0"] + CR["size"], CR["c0"]:CR["c0"] + CR["size"]]
    # first the estimator on the sharp crop itself
    est_D, S_D, cd_D = estimate_blur(cropD)
    val = []
    rngv = np.random.default_rng(SEED + 5)
    for (L, th) in [(8, 0), (8, 90), (8, 30), (15, 30), (15, 45), (15, 0.7), (15, 89.3), (25, 30), (25, 120), (40, 30), (40, 160), (70, 5), (70, 100)]:
        Hc = np.fft.fft2(motion_psf(L, th, cropD.shape))
        bl = blur_circular(cropD, Hc)
        for noise_sig in (0.0, 2.0):
            nz = bl + noise_sig * rngv.standard_normal(bl.shape)
            e, _, _ = estimate_blur(nz)
            val.append({"L_true": L, "theta_true": th, "noise_sigma_(0-255)": noise_sig, "L_cepstrum": e["L"], "theta_cepstrum": e["theta_deg"],
                        "L_comb": e["comb_refined"]["L"], "theta_comb": e["comb_refined"]["theta_deg"], "z": e["peak_z_score"],
                        "theta_err_comb": float(angdiff(e["comb_refined"]["theta_deg"], th)), "L_err_comb": e["comb_refined"]["L"] - L})
    R["q6_4_validation_on_synthetic_blur_of_photoD_crop"] = {"crop": CR, "rows": val, "estimate_on_unblurred_crop": {k: est_D[k] for k in ("L", "theta_deg", "peak_z_score")} | {"comb": est_D["comb_refined"]}}
    C.save_json(R, "q6.json")
    print("q6 part 1 done", flush=True)
    real_photo_part(R, plt)
    C.save_json(R, "q6.json")
    print("q6 done")
    return R


CROPS = {"window": (850, 100), "door": (700, 1900), "floor": (1700, 1500), "wall_clock": (200, 1000)}
CROP = 1024
TILE = 512


def comb_null(ctx, L, rng, n=300):
    """Distribution of the comb score at random (theta, L' in [0.7L,1.3L]) - what a stripe-free spectrum gives."""
    return np.array([ctx.score(L * rng.uniform(0.7, 1.3), rng.uniform(0, 180)) for _ in range(n)])


def angular_profile(S, theta_deg, half_width_deg=3.0):
    """Mean whitened log-spectrum in a thin wedge around the direction theta, versus |f| (cycles/pixel)."""
    fy = np.fft.fftfreq(S.shape[0])[:, None]
    fx = np.fft.fftfreq(S.shape[1])[None, :]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.degrees(np.arctan2(fy, fx)) % 180.0
    d = np.abs(angdiff(ang, theta_deg))
    sel = (d < half_width_deg) & (rho > 0.004) & (rho < 0.5)
    bins = np.linspace(0.004, 0.5, 120)
    idx = np.digitize(rho[sel], bins)
    vals = S[sel]
    prof = np.array([vals[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    return 0.5 * (bins[1:] + bins[:-1]), prof


def tile_average_spectrum(gray, tiles=None, tile=TILE, margin=100):
    """Mean of the whitened log-spectra of non-overlapping tile x tile windows covering the frame (averaging over many
    different scene contents suppresses scene-specific structure, keeps what is common to all tiles: blur and scene axes)."""
    H, W = gray.shape
    acc, n = 0.0, 0
    idx = [(r0, c0) for r0 in range(margin, H - tile, tile) for c0 in range(margin, W - tile, tile)]
    for k, (r0, c0) in enumerate(idx):
        if tiles is not None and k not in tiles:
            continue
        S, _, _ = flat_log_spectrum(gray[r0:r0 + tile, c0:c0 + tile], nb=64)
        acc = acc + S
        n += 1
    return acc / n, n


def _band_mean(S, ang, rho, t, lo=0.02, hi=0.12, hw=2.0):
    return S[(np.abs(angdiff(ang, t)) < hw) & (rho > lo) & (rho < hi)].mean()


def direction_from_difference(S_blur, S_ctrl):
    """Blur direction from the angular profile of (blurred - control) tile-averaged whitened log-spectra: the control
    cancels the scene's own axis-aligned anisotropy (its edges lie along x and y).  Blur along theta removes energy of
    frequencies along theta, so after the whitening the difference has its MAXIMUM perpendicular to the blur."""
    n = S_blur.shape[0]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.degrees(np.arctan2(fy, fx)) % 180.0
    th = np.arange(0, 180, 1.0)
    dif = np.array([_band_mean(S_blur, ang, rho, t) - _band_mean(S_ctrl, ang, rho, t) for t in th])
    k = np.convolve(np.r_[dif[-3:], dif, dif[:3]], np.ones(5) / 5, mode="same")[3:-3]            # circular 5-point smoothing
    i = int(np.argmax(k))
    am, a0, ap = k[(i - 1) % 180], k[i], k[(i + 1) % 180]
    d = am - 2 * a0 + ap
    perp = th[i] + (0.5 * (am - ap) / d if d != 0 else 0.0)
    return (perp - 90.0) % 180.0, th, dif


def double_ratio(S_blur, S_ctrl, theta_deg, hw=4.0):
    """Q(f) = [S_b(f along theta) - S_b(f perpendicular)] - [S_c(along) - S_c(perp)]  ~  log|H(f)|^2 along the blur
    direction (perpendicular direction unaffected), scene anisotropy cancelled by the control."""
    n = S_blur.shape[0]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.degrees(np.arctan2(fy, fx)) % 180.0
    bins = np.linspace(0.01, 0.3, 59)
    rc = 0.5 * (bins[1:] + bins[:-1])

    def prof(S, t):
        sel = np.abs(angdiff(ang, t)) < hw
        return np.array([S[sel & (rho >= bins[j]) & (rho < bins[j + 1])].mean() for j in range(len(bins) - 1)])
    return rc, (prof(S_blur, theta_deg) - prof(S_blur, theta_deg + 90)) - (prof(S_ctrl, theta_deg) - prof(S_ctrl, theta_deg + 90))


def length_from_first_lobe(rc, Q, fmax=0.05, kappa=0.02, Ls=None):
    """Box-length fit: minimise sum (Q(f) - log((sinc^2(L f)+kappa)/(1+kappa)))^2 over the first lobe f <= fmax
    (beyond it non-uniformity and the spectral floor flatten the curve; no periodic zeros survive)."""
    Ls = np.arange(3.0, 100.01, 0.5) if Ls is None else Ls
    sel = rc <= fmax
    sse = np.array([np.sum((Q[sel] - np.log((np.sinc(L * rc[sel]) ** 2 + kappa) / (1 + kappa))) ** 2) for L in Ls])
    i = int(np.argmin(sse))
    ok = Ls[sse <= sse[i] * 1.25]
    return float(Ls[i]), (float(ok.min()), float(ok.max())), float(sse[i]), Ls, sse


def dip_contrast(rc, Q, L, mmax=5):
    """Stripe contrast of a log-transfer curve at the predicted zeros: Q(m/L) minus the mean of Q at the two half-way points
    (m-1/2)/L and (m+1/2)/L; negative when there is a dip at m/L."""
    qa = lambda f: float(np.interp(f, rc, Q))
    return [qa(m / L) - 0.5 * (qa((m - 0.5) / L) + qa((m + 0.5) / L)) for m in range(1, mmax + 1)]


def deblur_real(crop_rgb, L, th, K, pad=160):
    """Wiener deconvolution of an RGB crop (0..255) with the estimated PSF; reflect padding (so the circular model
    does not wrap), same K for all channels."""
    h, w, _ = crop_rgb.shape
    from scipy.fft import next_fast_len
    Pr, Pc = next_fast_len(h + 2 * pad), next_fast_len(w + 2 * pad)
    r0, c0 = (Pr - h) // 2, (Pc - w) // 2
    H = np.fft.fft2(motion_psf(L, th, (Pr, Pc)))
    out = np.zeros_like(crop_rgb)
    for c in range(3):
        g = np.pad(crop_rgb[..., c], ((r0, Pr - h - r0), (c0, Pc - w - c0)), mode="reflect")
        out[..., c] = wiener_deconv(g, H, K)[r0:r0 + h, c0:c0 + w]
    return out


def real_photo_part(R, plt):
    rng = np.random.default_rng(SEED + 11)
    photos = {k: C.load_rgb(os.path.join(C.PHOTOS, C.PHOTO_FILES[k])) for k in ("C", "D")}
    gray = {k: C.luma(v) for k, v in photos.items()}
    # ---------------- (1) single-crop automatic estimator (cepstrum + comb) on native-resolution crops
    rows = []
    store = {}
    for key in ("C", "D"):
        for name, (r0, c0) in CROPS.items():
            g = gray[key][r0:r0 + CROP, c0:c0 + CROP]
            est, S, cd = estimate_blur(g)
            null = comb_null(est["_ctx"], est["comb_refined"]["L"], rng)
            sc = est["comb_refined"]["score"]
            rows.append({"photo": key, "crop": name, "r0": r0, "c0": c0, "cepstrum_L": est["L"], "cepstrum_theta": est["theta_deg"], "cepstrum_peak_z": est["peak_z_score"],
                         "comb_L": est["comb_refined"]["L"], "comb_theta": est["comb_refined"]["theta_deg"], "comb_score": sc,
                         "comb_score_null_mean": float(null.mean()), "comb_score_null_std": float(null.std()), "comb_z_vs_null": float((sc - null.mean()) / null.std()),
                         "at_search_bound": bool(est["L"] < 4.5 or est["L"] > 145),
                         "gradient_energy_ratio_horizontal/vertical": float(np.mean(np.diff(g, axis=1) ** 2) / np.mean(np.diff(g, axis=0) ** 2))})
            store[(key, name)] = (est, S, cd)
            print("single-crop", key, name, round(est["L"], 1), round(est["theta_deg"], 1), "z", round(rows[-1]["comb_z_vs_null"], 1), flush=True)
    R["q6_4_single_crop_estimates"] = rows
    # ---------------- (2) tile-averaged spectra, control-normalised direction and first-lobe length
    SC, nC = tile_average_spectrum(gray["C"])
    SD, nD = tile_average_spectrum(gray["D"])
    theta_hat, th_grid, dif = direction_from_difference(SC, SD)
    rc, Qf = double_ratio(SC, SD, theta_hat)
    L_hat, L_rng, sse_min, Ls, sse = length_from_first_lobe(rc, Qf)
    # null for the stripe structure: periodicity of Q beyond the first lobe
    R["q6_4_tile_estimate"] = {"tile": TILE, "n_tiles_C": nC, "n_tiles_D": nD, "theta_hat_deg": float(theta_hat), "L_hat_px": L_hat, "L_range_(sse<=1.25 min)": list(L_rng),
                               "diff_at_0deg": float(dif[0]), "diff_at_90deg": float(dif[90]), "diff_min": float(dif.min()), "diff_max": float(dif.max()),
                               "Q_first_points": [float(x) for x in Qf[:8]], "Q_floor_median_0.05-0.15": float(np.median(Qf[(rc > 0.05) & (rc < 0.15)])),
                               "Q_dip_contrast_at_m/L_hat_m=1..5_(value at m/L minus mean of the two half-way points)": dip_contrast(rc, Qf, L_hat),
                               "box_model_dip_contrast_m=1..5_(kappa=0.02)": dip_contrast(rc, np.log((np.sinc(L_hat * rc) ** 2 + 0.02) / 1.02), L_hat)}
    # ---- validation of the tile estimator on synthetic blur of D (split tiles: blur one half, use the other half as control)
    idxD = [(r0, c0) for r0 in range(100, 3072 - TILE, TILE) for c0 in range(100, 4080 - TILE, TILE)]
    Dg = gray["D"]
    vrows = []
    for (Lt, tht) in [(30, 2), (30, 45), (20, 0), (40, 90), (30, 150)]:
        Hc = np.fft.fft2(motion_psf(Lt, tht, (TILE, TILE)))
        acc_b, nb_ = 0.0, 0
        acc_c, nc_ = 0.0, 0
        for k, (r0, c0) in enumerate(idxD):
            t = Dg[r0:r0 + TILE, c0:c0 + TILE]
            if k % 2 == 0:
                # blur the tile with a circular model after reflect padding the neighbouring pixels (take a bigger window)
                big = Dg[max(r0 - 64, 0):r0 + TILE + 64, max(c0 - 64, 0):c0 + TILE + 64]
                Hb = np.fft.fft2(motion_psf(Lt, tht, big.shape))
                tb = blur_circular(big, Hb)
                rr0, cc0 = r0 - max(r0 - 64, 0), c0 - max(c0 - 64, 0)
                tb = tb[rr0:rr0 + TILE, cc0:cc0 + TILE]
                S, _, _ = flat_log_spectrum(tb, nb=64)
                acc_b, nb_ = acc_b + S, nb_ + 1
            else:
                S, _, _ = flat_log_spectrum(t, nb=64)
                acc_c, nc_ = acc_c + S, nc_ + 1
        thv, _, _ = direction_from_difference(acc_b / nb_, acc_c / nc_)
        rcv, Qv = double_ratio(acc_b / nb_, acc_c / nc_, thv)
        Lv, rngv, _, _, _ = length_from_first_lobe(rcv, Qv)
        vrows.append({"L_true": Lt, "theta_true": tht, "theta_est": float(thv), "theta_err": float(angdiff(thv, tht)), "L_est": Lv, "L_range": list(rngv), "L_err_%": float(100 * (Lv - Lt) / Lt)})
        print("validation", vrows[-1], flush=True)
    R["q6_4_tile_estimator_validation_(split_tiles_of_D)"] = vrows
    # ---------------- figures
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.6))
    for i, (nm, S) in enumerate((("C (blurred)", SC), ("D (sharper)", SD))):
        ax[i, 0].imshow(np.fft.fftshift(np.clip(S, -2.5, 1.2)), cmap="gray", extent=[-0.5, 0.5, 0.5, -0.5])
        ax[i, 0].set_xlim(-0.2, 0.2)
        ax[i, 0].set_ylim(0.2, -0.2)
        ax[i, 0].set_title("%s: tile-averaged whitened log-spectrum (%d tiles of 512x512)" % (nm, nC), fontsize=8)
        ax[i, 0].set_xlabel("fx (cycles/pixel)")
        ax[i, 0].set_ylabel("fy")
    th_r = math.radians(theta_hat)
    ax[0, 0].plot([-0.2 * math.cos(th_r), 0.2 * math.cos(th_r)], [-0.2 * math.sin(th_r), 0.2 * math.sin(th_r)], "c--", lw=1)
    ax[0, 0].text(0.01, 0.17, "estimated blur direction theta = %.1f deg" % theta_hat, color="c", fontsize=8)
    ax[0, 1].plot(th_grid, dif)
    ax[0, 1].axvline((theta_hat + 90) % 180, color="r", ls=":")
    ax[0, 1].set_title("angular profile of (C - D) in the band 0.02-0.12 c/px: max at %.1f deg, i.e. blur direction %.1f deg" % ((theta_hat + 90) % 180, theta_hat), fontsize=8)
    ax[0, 1].set_xlabel("frequency direction phi (deg)")
    ax[1, 1].plot(rc, Qf, "k", label="Q(f): double ratio along theta_hat")
    for Lm, st in ((L_hat, "-"), (L_rng[0], ":"), (L_rng[1], ":")):
        ax[1, 1].plot(rc, np.log((np.sinc(Lm * rc) ** 2 + 0.02) / 1.02), "r" + st, lw=1, label="box model L=%.1f px" % Lm)
    for m in range(1, 8):
        ax[1, 1].axvline(m / L_hat, color="c", lw=0.4)
    ax[1, 1].set_xlim(0, 0.3)
    ax[1, 1].set_ylim(-6, 1)
    ax[1, 1].set_xlabel("|f| (cycles/pixel); cyan = m/L_hat, spacing 1/L_hat = %.4f" % (1 / L_hat))
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("estimated |H(f)|^2 along the blur direction (log) with fitted box: first lobe only", fontsize=8)
    ax[0, 2].plot(Ls, sse)
    ax[0, 2].axvline(L_hat, color="r")
    ax[0, 2].set_xlabel("L (px)")
    ax[0, 2].set_title("first-lobe fit error vs L (minimum %.1f px)" % L_hat, fontsize=8)
    # single-crop spectra for reference
    est, S, cd = store[("C", "window")]
    ax[1, 2].imshow(np.fft.fftshift(np.clip(S, -3, 3)), cmap="gray", extent=[-0.5, 0.5, 0.5, -0.5])
    ax[1, 2].set_xlim(-0.2, 0.2)
    ax[1, 2].set_ylim(0.2, -0.2)
    ax[1, 2].set_title("single 1024x1024 crop of C (window): no clean stripe pattern, only the dominant axis-aligned scene structure", fontsize=8)
    C.savefig(plt, fig, "q6_real_spectra.png")
    # ---------------- deblurring with the adopted (= automatic tile-based) estimate
    primary = "window"
    r0, c0 = CROPS[primary]
    crop = photos["C"][r0:r0 + CROP, c0:c0 + CROP]
    L_ad, th_ad = L_hat, theta_hat
    sig_hat = []
    for c in range(3):
        rr = Q5.patch_noise(crop[..., c], patch=24)
        thr = np.percentile(rr[:, 2], 10)
        sig_hat.append(float(np.median(rr[rr[:, 2] <= thr][:, 0])))
    var_sig = [float(crop[..., c].var()) for c in range(3)]
    K_est = float(np.mean([sig_hat[c] ** 2 / var_sig[c] for c in range(3)]))
    Ks = [K_est / 10, K_est, K_est * 10, K_est * 100, K_est * 1000]
    outs = [deblur_real(crop, L_ad, th_ad, K) for K in Ks]
    R["q6_4_deblur"] = {"primary_crop": primary, "crop_origin_(r0,c0)": [r0, c0], "crop_size": CROP,
                        "automatic_estimate_tile_based": {"L": L_hat, "theta_deg": theta_hat}, "adopted": {"L": L_ad, "theta_deg": th_ad, "K": K_est * 10, "K/K_est": 10.0, "note": "L and theta = automatic tile-based estimate, unchanged; K = 10 x the NSR-formula value K_est, chosen by visual inspection of the sweep (no ground truth exists)"},
                        "sigma_hat_RGB_(0-255)": sig_hat, "signal_variance_RGB": var_sig, "K_estimated_(sigma^2/var)": K_est,
                        "K_sweep": [{"K": K, "K/K_est": K / K_est, "fraction_outside_0_255_%": float(100 * ((o < 0) | (o > 255)).mean()), "min": float(o.min()), "max": float(o.max()),
                                     "rms_change_vs_input": float((o - crop).std())} for K, o in zip(Ks, outs)]}
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    z_ = (slice(0, 520), slice(0, 520))
    ax[0, 0].imshow(np.clip(crop[z_], 0, 255).astype(np.uint8))
    ax[0, 0].set_title("photo C crop (blurred, native resolution)")
    for a, K, o in zip(ax.ravel()[1:], Ks, outs):
        a.imshow(np.clip(o[z_], 0, 255).astype(np.uint8))
        a.set_title("Wiener, L=%.1f theta=%.1f K=%.2e (%g x K_est)" % (L_ad, th_ad, K, round(K / K_est, 3)), fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    C.savefig(plt, fig, "q6_real_deblur.png")
    from PIL import Image
    Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).save(os.path.join(C.DERIVED, "photoC_window_crop_native.png"))
    for lab, o in zip(("Kest_div10", "Kest", "Kest_x10", "Kest_x100", "Kest_x1000"), outs):
        Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).save(os.path.join(C.DERIVED, "photoC_window_wiener_%s.png" % lab))
    # second crop (door) and a lightly blurred region (floor) with the same global PSF: shows the non-uniformity
    extra = {}
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    for j, name in enumerate(("door", "floor", "wall_clock")):
        rr0, cc0 = CROPS[name]
        cr = photos["C"][rr0:rr0 + CROP, cc0:cc0 + CROP]
        o = deblur_real(cr, L_ad, th_ad, K_est * 10)
        ax[0, j].imshow(np.clip(cr[:520, :520], 0, 255).astype(np.uint8))
        ax[0, j].set_title("C %s crop (blurred)" % name)
        ax[1, j].imshow(np.clip(o[:520, :520], 0, 255).astype(np.uint8))
        ax[1, j].set_title("same PSF (L=%.1f, theta=%.1f), K=%.1e" % (L_ad, th_ad, K_est * 10))
        extra[name] = {"fraction_outside_0_255_%": float(100 * ((o < 0) | (o > 255)).mean())}
    for a in ax.ravel():
        a.axis("off")
    C.savefig(plt, fig, "q6_real_other_crops.png")
    R["q6_4_deblur"]["other_crops_K=10Kest"] = extra
    # control D: apply the same PSF to the sharper photo (qualitative demonstration of the model's effect on an unblurred image)
    cropD = photos["D"][r0:r0 + CROP, c0:c0 + CROP]
    oD = deblur_real(cropD, L_ad, th_ad, K_est * 10)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(np.clip(cropD[z_], 0, 255).astype(np.uint8))
    ax[0].set_title("photo D crop (1/50 s), not deblurred")
    ax[1].imshow(np.clip(oD[z_], 0, 255).astype(np.uint8))
    ax[1].set_title("D after applying the PSF estimated from C (control: ringing, no gain)")
    for a in ax:
        a.axis("off")
    C.savefig(plt, fig, "q6_real_control_D.png")
    # cepstrum maps (single-crop estimator) for the record
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for a, key in zip(ax, ("C", "D")):
        est_, _, cd_ = store[(key, "window")]
        cy_, cx_ = np.array(cd_.shape) // 2
        w = 160
        v = np.percentile(np.abs(cd_), 99.9)
        a.imshow(cd_[cy_ - w:cy_ + w, cx_ - w:cx_ + w], cmap="coolwarm", vmin=-v, vmax=v, extent=[-w, w, w, -w])
        a.plot([est_["L"] * math.cos(math.radians(est_["theta_deg"])), -est_["L"] * math.cos(math.radians(est_["theta_deg"]))],
               [est_["L"] * math.sin(math.radians(est_["theta_deg"])), -est_["L"] * math.sin(math.radians(est_["theta_deg"]))], "k+", ms=12)
        a.set_title("photo %s window crop: detrended cepstrum; + = most negative peak (L=%.1f, theta=%.1f) at the lower search bound" % (key, est_["L"], est_["theta_deg"]), fontsize=8)
    C.savefig(plt, fig, "q6_real_cepstrum.png")


if __name__ == "__main__":
    main()
