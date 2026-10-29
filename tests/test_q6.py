import math
import unittest

import _path  # noqa: F401
import numpy as np

import q6_motion as Q


class TestMotion(unittest.TestCase):
    def test_psf_normalised_nonnegative_real_otf(self):
        for L, th in [(7, 0), (12.5, 33), (20, 90), (9, 135)]:
            p = Q.motion_psf(L, th, (64, 64))
            self.assertAlmostEqual(p.sum(), 1.0, places=12)
            self.assertGreaterEqual(p.min(), 0)
            self.assertLess(np.abs(np.fft.fft2(p).imag).max(), 1e-12)            # point symmetric -> real OTF

    def test_axis_aligned_psf_confined_to_one_row(self):
        p = Q.motion_psf(5, 0, (16, 16))
        self.assertEqual(np.count_nonzero(p[1:]), 0)
        self.assertAlmostEqual(p[0, :].sum(), 1.0)

    def test_otf_matches_sinc_times_triangle_factor(self):
        # DFT of the bilinearly splatted line = sum over aliases k of  sinc(L s(f-k)) sinc^2(fx-kx) sinc^2(fy-ky);
        # near DC the k=0 term (sinc x triangle factor) alone is accurate.
        shape = (96, 128)
        L, th = 11.0, 37.0
        p = Q.motion_psf(L, th, shape, oversample=64)
        H = np.fft.fft2(p)
        fy = np.fft.fftfreq(shape[0])[:, None]
        fx = np.fft.fftfreq(shape[1])[None, :]
        near = np.sqrt(fx ** 2 + fy ** 2) < 0.1
        Hi = Q.otf_analytic(L, th, shape) * Q.interp_factor(shape)
        self.assertLess(np.abs(H - Hi)[near].max(), 2e-3)
        tot = 0
        for ky in range(-4, 5):
            for kx in range(-4, 5):
                x, y = fx - kx, fy - ky
                tot = tot + np.sinc(L * (x * math.cos(math.radians(th)) + y * math.sin(math.radians(th)))) * np.sinc(x) ** 2 * np.sinc(y) ** 2
        self.assertLess(np.abs(H - tot).max(), 2e-4)

    def test_zero_spacing_along_direction(self):
        # axis aligned, L=8 on a 64 grid: |H| vanishes at f=m/8 -> bins 8m ; half-way (bin 4) is not a zero
        p = Q.motion_psf(8, 0, (64, 64), oversample=64)
        H = np.fft.fft2(p)
        for m in (1, 2, 3):
            self.assertLess(abs(H[0, 8 * m]), 1e-3)
        self.assertGreater(abs(H[0, 4]), 0.1)

    def test_wiener_with_tiny_K_inverts_noise_free_blur(self):
        img = np.random.default_rng(0).random((64, 64))
        H = np.fft.fft2(Q.motion_psf(5, 30, img.shape))
        g = Q.blur_circular(img, H)
        f = Q.wiener_deconv(g, H, 1e-14)
        self.assertLess(np.sqrt(np.mean((f - img) ** 2)), 5e-3)

    def test_noise_makes_naive_inverse_fail_and_wiener_help(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 1, 128)
        img = 0.5 + 0.3 * np.outer(np.sin(5 * x), np.cos(3 * x))
        H = np.fft.fft2(Q.motion_psf(9, 20, img.shape))
        g = Q.blur_circular(img, H) + 0.01 * rng.standard_normal(img.shape)
        naive = np.fft.irfft2(Q.inverse_filter(np.fft.rfft2(g), Q.rH(H, img.shape), 1e-8), s=img.shape)
        w = Q.wiener_deconv(g, H, 1e-3)
        e = lambda a: np.sqrt(np.mean((a - img) ** 2))
        self.assertGreater(e(naive), 5 * e(w))
        self.assertGreater(e(Q.wiener_deconv(g, H, 10.0)), e(w))                 # too much K: under-deconvolved

    def test_snr_definition(self):
        f = np.random.default_rng(2).random((64, 64, 3))
        sg = Q.snr_noise_sigma(f, 20.0)
        n = sg * np.random.default_rng(3).standard_normal(f.shape)
        self.assertAlmostEqual(10 * math.log10(np.sum(f ** 2) / np.sum(n ** 2)), 20.0, delta=0.1)

    def test_estimator_known_angle_and_length_incl_near_axis(self):
        rng = np.random.default_rng(4)
        N = 512
        fx = np.fft.fftfreq(N)
        rho = np.sqrt(fx[:, None] ** 2 + fx[None, :] ** 2)
        rho[0, 0] = 1
        tex = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) / rho))     # 1/f texture
        tex = (tex - tex.min()) / np.ptp(tex) * 255
        for L, th in [(15, 30), (20, 90), (20, 0.8), (20, 89.2), (30, 150)]:
            g = Q.blur_circular(tex, np.fft.fft2(Q.motion_psf(L, th, tex.shape)))
            e, _, _ = Q.estimate_blur(g, lmin=4, lmax=50)
            c = e["comb_refined"]
            self.assertLess(abs(Q.angdiff(c["theta_deg"], th)), 2.0, msg=(L, th, c))
            self.assertLess(abs(c["L"] - L), 0.12 * L + 1.0, msg=(L, th, c))


if __name__ == "__main__":
    unittest.main()
