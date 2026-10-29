import unittest

import _path  # noqa: F401
import numpy as np

import q4_edge_preserving as Q


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_bilateral_constant_and_normalised(self):
        img = np.full((20, 20), 42.0)
        self.assertLess(np.abs(Q.bilateral(img, 2.0, 10.0) - 42).max(), 1e-12)

    def test_bilateral_large_sigma_r_is_gaussian(self):
        img = self.rng.random((24, 24)) * 255
        a = Q.bilateral(img, 1.5, 1e9)
        r = 5
        x = np.arange(-r, r + 1)
        k = np.exp(-x ** 2 / (2 * 1.5 ** 2))
        k /= k.sum()
        p = np.pad(img, r, mode="reflect")
        t = sum(k[i] * p[:, i:i + 24] for i in range(2 * r + 1))
        g = sum(k[i] * t[i:i + 24, :] for i in range(2 * r + 1))
        self.assertLess(np.abs(a - g).max(), 1e-9)

    def test_bilateral_preserves_clean_step_for_small_sigma_r(self):
        img = np.zeros((20, 20))
        img[:, 10:] = 100.0
        self.assertLess(np.abs(Q.bilateral(img, 2.0, 1.0) - img).max(), 1e-6)

    def test_bilateral_brute_force_single_pixel(self):
        img = self.rng.random((15, 15)) * 255
        ss, sr, r = 1.5, 40.0, 5
        o = Q.bilateral(img, ss, sr)
        y, x = 7, 7
        num = den = 0.0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w = np.exp(-(dx * dx + dy * dy) / (2 * ss * ss)) * np.exp(-(img[y + dy, x + dx] - img[y, x]) ** 2 / (2 * sr * sr))
                num += w * img[y + dy, x + dx]
                den += w
        self.assertAlmostEqual(o[y, x], num / den, places=9)

    def test_kuwahara_constant_and_step(self):
        self.assertLess(np.abs(Q.kuwahara(np.full((16, 16), 5.0), 5) - 5).max(), 1e-12)
        step = np.zeros((20, 20))
        step[:, 10:] = 100.0
        self.assertLess(np.abs(Q.kuwahara(step, 7) - step).max(), 1e-9)              # edge kept exactly

    def test_kuwahara_brute_force_pixel(self):
        img = self.rng.random((21, 21)) * 255
        N, k = 7, 3
        y, x = 10, 10
        wins = [img[y - k:y + 1, x - k:x + 1], img[y - k:y + 1, x:x + k + 1],
                img[y:y + k + 1, x - k:x + 1], img[y:y + k + 1, x:x + k + 1]]
        best = min(wins, key=lambda w: w.var())
        self.assertAlmostEqual(Q.kuwahara(img, N)[y, x], best.mean(), places=9)

    def test_soft_limits(self):
        img = self.rng.random((24, 24)) * 255
        hard = Q.kuwahara(img, 7)
        self.assertLess(np.abs(Q.soft_kuwahara(img, 7, 1e-9) - hard).max(), 1e-6)
        means, _ = Q._quadrants(img, 7)
        self.assertLess(np.abs(Q.soft_kuwahara(img, 7, 1e12) - means.mean(0)).max(), 1e-6)

    def test_even_window_rejected(self):
        with self.assertRaises(ValueError):
            Q.kuwahara(np.zeros((10, 10)), 6)

    def test_noise_estimator(self):
        n = np.random.default_rng(3).normal(0, 12.0, (256, 256))
        self.assertAlmostEqual(Q.noise_sigma_immerkaer(n), 12.0, delta=0.5)

    def test_kuwahara_flat_variance_ratio_close_to_1_over_m2(self):
        f = np.random.default_rng(4).normal(0, 10, (200, 200))
        r = Q.flat_variance_ratio(lambda x: Q.kuwahara(x, 7), f)
        self.assertAlmostEqual(r, 1 / 16, delta=0.02)

    def test_range_weight_ratio_closed_form_by_monte_carlo(self):
        rng = np.random.default_rng(5)
        dI, ncr = 100.0, 0.5
        sg = ncr * dI
        sr = 2 * sg
        n1, n2 = rng.normal(0, sg, 10 ** 6), rng.normal(0, sg, 10 ** 6)
        mc = np.exp(-(dI + n2 - n1) ** 2 / (2 * sr ** 2)).mean() / np.exp(-(n2 - n1) ** 2 / (2 * sr ** 2)).mean()
        self.assertAlmostEqual(mc, np.exp(-1 / (12 * ncr ** 2)), delta=0.01)


if __name__ == "__main__":
    unittest.main()
