import unittest

import _path  # noqa: F401
import numpy as np
import scipy.ndimage as ndi

import q2_filtering as Q


class TestFilters(unittest.TestCase):
    P = 256

    def test_values_at_cutoff(self):
        D0 = 20
        i, j = self.P // 2, self.P // 2 + D0
        self.assertEqual(Q.make_filter("ideal", self.P, D0)[i, j], 1.0)
        self.assertAlmostEqual(Q.make_filter("butterworth", self.P, D0, n=3)[i, j], 0.5)
        self.assertAlmostEqual(Q.make_filter("gaussian", self.P, D0)[i, j], np.exp(-0.5))
        self.assertAlmostEqual(Q.make_filter("gaussian", self.P, D0 * Q.GAUSS_MATCH)[i, j], 0.5)

    def test_highpass_is_one_minus_lowpass_and_dc(self):
        for kw in (dict(kind="ideal"), dict(kind="butterworth", n=4), dict(kind="gaussian")):
            lp = Q.make_filter(P=self.P, D0=15, **kw)
            hp = Q.make_filter(P=self.P, D0=15, highpass=True, **kw)
            self.assertLess(np.abs(lp + hp - 1).max(), 1e-15)
            self.assertEqual(hp[self.P // 2, self.P // 2], 0.0)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            Q.make_filter("butterworth", 64, 10)
        with self.assertRaises(ValueError):
            Q.make_filter("ideal", 64, 0)

    def test_inverse_design(self):
        # H(10)=16/17, H(40)=1/17 -> n=2, D0=20; smallest n with H(2 D0)<=1% is 4
        H = Q.make_filter("butterworth", 512, 20, n=2)
        self.assertAlmostEqual(H[256, 256 + 10], 16 / 17)
        self.assertAlmostEqual(H[256, 256 + 40], 1 / 17)
        vals = {n: Q.make_filter("butterworth", 512, 20, n=n)[256, 256 + 40] for n in range(1, 7)}
        self.assertEqual(min(n for n, v in vals.items() if v <= 0.01), 4)

    def test_constant_image_interior_preserved_by_wide_lowpass(self):
        img2 = np.full((128, 128), 7.0)
        out2 = Q.filter_image(img2, Q.make_filter("gaussian", 256, 40))
        self.assertAlmostEqual(out2[64, 64], 7.0, places=3)

    def test_output_real_zero_phase(self):
        img = np.random.default_rng(0).random((40, 40))
        H = Q.make_filter("butterworth", 80, 10, n=2)
        pad = np.zeros((80, 80))
        pad[:40, :40] = img
        full = np.fft.ifft2(np.fft.fft2(pad) * np.fft.ifftshift(H))
        self.assertLess(np.abs(full.imag).max(), 1e-12)

    def test_step_overshoot_ideal_1d_strip_matches_gibbs(self):
        # x-only image, background padded: ideal LPF step overshoot = 1/2+Si(pi)/pi-1 = 8.95 %
        P, N = 1024, 512
        row = np.full(N, 50.0)
        row[128:384] = 200.0
        img = np.full((P, P), 50.0)
        img[:, :N] = row
        H = Q.make_filter("ideal", P, 40)
        g = np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.ifftshift(H)))[512, :N]
        self.assertAlmostEqual(100 * (g.max() - 200) / 150, 8.95, delta=0.8)
        Hg = Q.make_filter("gaussian", P, 40)                                  # non-negative kernel: no overshoot
        gg = np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.ifftshift(Hg)))[512, :N]
        self.assertLessEqual(gg.max(), 200 + 1e-9)

    def test_butterworth_kernel_ratio_grows_with_order(self):
        def ratio(n):
            h = Q.kernel_1d(Q.make_filter("butterworth", 1024, 40, n=n))
            return abs(h.min()) / h.max()
        r = [ratio(n) for n in (1, 2, 5, 20)]
        self.assertLess(r[0], 1e-4)
        self.assertTrue(all(a < b for a, b in zip(r, r[1:])))

    def test_gaussian_equivalence_sigma_s(self):
        P, N = 256, 128
        img = np.random.default_rng(0).random((N, N)) * 255
        D0 = 12
        out = Q.filter_image(img, Q.make_filter("gaussian", P, D0))
        ref = ndi.gaussian_filter(img, Q.sigma_spatial(P, D0), mode="constant", truncate=8.0)
        self.assertLess(np.abs(out - ref).max(), 1e-6)

    def test_power_fraction_monotone(self):
        img = np.random.default_rng(1).random((32, 32)) * 255
        t, dc, P = Q.power_fraction_table(img)
        r = [t[k]["radius_bins_(padded_P=%d)" % P] for k in ("90", "95", "99", "99.5")]
        self.assertEqual(r, sorted(r))


if __name__ == "__main__":
    unittest.main()
