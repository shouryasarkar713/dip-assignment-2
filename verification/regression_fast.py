"""Fast regression: recompute key quantities from scratch (NOT reading any stored result as input) and compare them with
the stored authoritative results/*.json using explicit tolerances.  Also runs the unit tests.

    python verification/regression_fast.py            # exit code 0 iff everything passes

Compares: numerical identities (tight tolerances), deterministic experiment outputs (rtol 1e-6 unless stated).
Runtime-performance measurements are never compared.  Takes a few minutes (Q5/Q6 parts reload the 1536x1024 working image).
"""
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

results = []


def check(name, got, ref, rtol=1e-6, atol=1e-9):
    ok = bool(abs(got - ref) <= atol + rtol * max(abs(got), abs(ref)))
    results.append((name, ok, got, ref))
    print("%-4s %-70s got=%.10g stored=%.10g" % ("PASS" if ok else "FAIL", name, got, ref), flush=True)


def check_true(name, cond, detail=""):
    results.append((name, bool(cond), detail, ""))
    print("%-4s %-70s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def J(n):
    return json.load(open(os.path.join(ROOT, "results", n)))


def part_unit_tests():
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", os.path.join(ROOT, "tests"), "-t", os.path.join(ROOT, "tests")], capture_output=True, text=True, cwd=ROOT)
    tail = (r.stderr or "").strip().splitlines()[-3:]
    check_true("unit tests (python -m unittest discover tests)", r.returncode == 0, " | ".join(tail))


def part_q1():
    import q1_dft as Q
    R = J("q1.json")
    img = C.load_gray(os.path.join(C.PROVIDED, "q1_testimage.png"))
    F = Q.dft2(img)
    check("Q1 custom-vs-numpy max abs error N=512 (rtol 0.5: rounding noise)", float(np.abs(F - np.fft.fft2(img)).max()), R["q1_1_dft_error"]["full_512_max_abs_err"], rtol=0.5, atol=1e-8)
    check_true("Q1 error below 1e-7", np.abs(F - np.fft.fft2(img)).max() < 1e-7)
    check("Q1 conjugacy classes N=512", Q.conjugacy_class_counts(512)["conjugacy_classes"], R["q1_1a_conj_symmetry"]["counts_512"]["conjugacy_classes"])
    # shift theorem for one shift
    fs = np.roll(img, (100, -37), axis=(0, 1))
    rel = float(np.abs(Q.dft2(fs) - Q.circ_shift_phase(F, 100, -37)).max() / np.abs(F).max())
    check_true("Q1 shift theorem relative error < 1e-12", rel < 1e-12, "%.3g" % rel)
    # phase-only reconstruction statistics (deterministic)
    fh = np.real(Q.idft2(np.exp(1j * np.angle(F))))
    check("Q1.2 kurtosis of phase-only reconstruction", Q.kurtosis(fh), R["q1_2"]["kurtosis_fhat_flat"], rtol=1e-8)
    m = np.abs(F).ravel()[1:]
    check("Q1.2 exact corr(f_hat,f) = E|F|/sqrt(E|F|^2)", float(m.mean() / np.sqrt((m ** 2).mean())), R["q1_2"]["corr(fhat_flat,f)_(pred_|r|<0.1)"], rtol=1e-8)
    comps = Q.SINUSOIDS
    pk = Q.pick_peaks(Q.dft2(Q.synth(512, comps)), 512, 4)
    check("Q1.1c recovered amplitude of strongest sinusoid", pk[0]["A"], comps[0][2], atol=1e-9)


def part_q2():
    import q2_filtering as Q
    R = J("q2.json")
    P = 1024
    Fs = Q.filter_spectrum(Q.step_image()[0], P)
    s_img, lo = Q.step_image()
    for D0 in (40, 80):
        g = Q.apply_H(Fs, Q.make_filter("ideal", P, D0), s_img.shape)
        st = Q.profile_stats(g, 256, lo, 256)
        ref = [x for x in R["q2_2a"]["ideal"] if x["D0"] == D0][0]["measured"]
        check("Q2.2a ILPF D0=%d overshoot %%" % D0, st["overshoot_%"], ref["overshoot_%"])
        check("Q2.2a ILPF D0=%d first-peak distance" % D0, st["first_peak_distance_px"], ref["first_peak_distance_px"])
    h = Q.kernel_1d(Q.make_filter("butterworth", P, 40, n=2))
    check("Q2.2b Butterworth n=2 |min h|/max h", abs(h.min()) / h.max(), [r for r in R["q2_2b"]["rows"] if r["n"] == 2][0]["|min h|/max h"])
    check("Q2.1c inverse design D0", float(R["q2_1c"]["D0_solution"]), 20.0)
    H = Q.make_filter("butterworth", 512, 20, n=2)
    check("Q2.1c H(10)=16/17", float(H[256, 266]), 16 / 17, rtol=1e-12)
    check("Q2.1c H(40)=1/17", float(H[256, 296]), 1 / 17, rtol=1e-12)
    pf, dc, Pc = Q.power_fraction_table(C.load_gray(os.path.join(C.PROVIDED, "characters_test_pattern.tif")))
    check("Q2.1b radius for 95 % power", pf["95"]["radius_bins_(padded_P=1376)"], R["q2_1b"]["table"]["95"]["radius_bins_(padded_P=1376)"])
    cam = C.load_gray(os.path.join(C.PROVIDED, "cameraman.tif"))
    noisy = cam + np.random.default_rng(Q.SEED).normal(0, 20.0, cam.shape)
    Fn = Q.filter_spectrum(noisy, 1024)
    o = Q.apply_H(Fn, Q.make_filter("butterworth", 1024, 150, n=2), cam.shape)
    check("Q2.3a BLPF n=2 PSNR at D0=150 (seed 2026)", C.psnr(cam, o), R["q2_3a"]["PSNR_at_D0"]["BLPF n=2"]["150"], rtol=1e-9)
    gf = Q.apply_H(Q.filter_spectrum(cam, 1024), Q.make_filter("gaussian", 1024, 160), cam.shape)
    import scipy.ndimage as ndi
    gs = ndi.gaussian_filter(cam, Q.sigma_spatial(1024, 160), mode="constant", truncate=8.0)
    check("Q2.2d Gaussian equivalence error D0=160", float(np.abs(gf - gs).max()), R["q2_2d"]["rows"][2]["max_abs_err_truncate8_zero_boundary"], rtol=1e-6)


def part_q3():
    import q3_homomorphic as Q
    R = J("q3.json")
    rgb = C.load_rgb(os.path.join(C.PROVIDED, "tunnel_image.jpg")) / 255.0
    rgb = Q.block_mean(rgb, Q.FACTOR)
    Y = C.luma(rgb)
    o = Q.normalise(Q.homomorphic(Y, 0.5, 2.5, 0.08))
    ref = [x for x in R["grid"] if x["gL"] == 0.5 and x["gH"] == 2.5 and x["D0_cycles_per_pixel"] == 0.08][0]
    r0, r1, c0, c1 = R["regions"]["shadow_box_(r0,r1,c0,c1)"]
    st = Q.region_stats(o, (r0, r1, c0, c1), None)
    check("Q3 shadow local std (gL=.5,gH=2.5,D0=.08)", st["local_std_9x9_(0-255)"], ref["shadow"]["local_std_9x9_(0-255)"])
    check("Q3 shadow mean level", st["mean_level_(0-255)"], ref["shadow"]["mean_level_(0-255)"])
    he = Q.hist_equalise(Y)
    check("Q3 HE shadow mean level", float(255 * he[r0:r1, c0:c1].mean()), R["hist_equalisation"]["shadow"]["mean_level_(0-255)"])


def part_q4():
    import q4_edge_preserving as Q
    R = J("q4.json")
    gt = C.load_gray(os.path.join(C.PROVIDED, "q4_groundtruth.png"))
    rng = np.random.default_rng(Q.SEED)
    noisy = gt + rng.normal(0, Q.SIGMA_MAIN, gt.shape)
    check("Q4 noisy PSNR", C.psnr(gt, noisy), R["noisy_input"]["PSNR_dB_(clip_for_metric)"], rtol=1e-9)
    row = [r for r in R["q4_1b_performance"]["rows"] if r["N"] == 7][0]
    check("Q4 Kuwahara N=7 PSNR", C.psnr(gt, Q.kuwahara(noisy, 7)), row["kuwahara"]["PSNR"], rtol=1e-9)
    check("Q4 Kuwahara N=7 SSIM", C.ssim(gt, Q.kuwahara(noisy, 7)), row["kuwahara"]["SSIM"], rtol=1e-9)
    sh = Q.noise_sigma_immerkaer(noisy)
    check("Q4 soft-Kuwahara N=7 PSNR", C.psnr(gt, Q.soft_kuwahara(noisy, 7, sh ** 2)), row["soft_kuwahara"]["PSNR"], rtol=1e-9)
    o = Q.bilateral(noisy, row["sigma_s_bilateral"], 2 * Q.SIGMA_MAIN)
    check("Q4 bilateral N=7 matched PSNR", C.psnr(gt, o), row["bilateral"]["PSNR"], rtol=1e-9)
    check("Q4 analytic NCR at weight ratio 0.5 = 1/sqrt(12 ln 2)", 1 / math.sqrt(12 * math.log(2)), R["q4_3_analytic"]["rho_half_NCR_(closed_form)"], rtol=1e-12)


def part_q5():
    import q5_wiener as Q
    R = J("q5.json")
    src = C.load_rgb(os.path.join(C.PROVIDED, "reference_image.jpg"))        # same float 2x2 block mean as q5_wiener.py (not the 8-bit PNG copy)
    ref = src[:src.shape[0] // 2 * 2, :src.shape[1] // 2 * 2].reshape(src.shape[0] // 2, 2, src.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    z = np.random.default_rng(Q.SEED).standard_normal(ref.shape)
    for s in (5, 20):
        noisy = ref + s * z
        wk = Q.per_channel(lambda c: Q.wiener_denoise(c / 255.0, s / 255.0) * 255.0, noisy)
        row = [r for r in R["q5_1_table"] if r["sigma"] == s][0]["wiener_known_sigma"]
        check("Q5 Wiener known-sigma PSNR sigma=%d" % s, C.psnr(ref, wk), row["PSNR"], rtol=1e-7)
        check("Q5 Wiener known-sigma SSIM sigma=%d" % s, C.ssim(ref, wk), row["SSIM"], rtol=1e-7)


def part_q6():
    import q6_motion as Q
    R = J("q6.json")
    ref = C.load_rgb(os.path.join(C.DERIVED, "reference_work_1536x1024.png")) / 255.0
    shape = ref.shape[:2]
    H = np.fft.fft2(Q.motion_psf(Q.L_TRUE, Q.THETA_TRUE, shape))
    z = np.random.default_rng(Q.SEED).standard_normal(ref.shape)
    snr = 30
    sg = Q.snr_noise_sigma(ref, snr)
    noisy = np.stack([Q.blur_circular(ref[..., c], H) for c in range(3)], -1) + sg * z
    lv = R["q6_3_levels"]["30"]
    Kopt = lv["K_opt"]
    Hh = Q.rH(H, shape)
    A2 = np.abs(Hh) ** 2
    Gs = [np.fft.rfft2(noisy[..., c]) for c in range(3)]
    ps = []
    for K in (Kopt / 10 ** 0.05, Kopt, Kopt * 10 ** 0.05):
        o = np.stack([np.fft.irfft2(np.conj(Hh) * Gs[c] / (A2 + K), s=shape) for c in range(3)], -1)
        ps.append(C.psnr(ref * 255, o * 255))
    check("Q6 PSNR at stored K_opt (SNR 30 dB)", ps[1], lv["PSNR_at_Kopt"], rtol=1e-7)
    check_true("Q6 stored K_opt is a local maximum of PSNR(K)", ps[1] >= ps[0] and ps[1] >= ps[2], "%.4f %.4f %.4f" % tuple(ps))
    check("Q6 achieved SNR (dB)", 10 * math.log10(np.sum(ref ** 2) / np.sum((sg * z) ** 2)), lv["achieved_SNR_dB"], rtol=1e-9)


def main():
    t0 = time.time()
    for fn in (part_unit_tests, part_q1, part_q2, part_q3, part_q4, part_q5, part_q6):
        try:
            fn()
        except Exception as ex:   # a crash is a failure, never a silent skip
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, False, "exception", repr(ex)))
            print("FAIL %s raised %r" % (fn.__name__, ex))
    n_ok = sum(1 for r in results if r[1])
    print("\n%d / %d checks passed in %.0f s" % (n_ok, len(results), time.time() - t0))
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
