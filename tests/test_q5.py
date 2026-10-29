import unittest

import _path  # noqa: F401
import numpy as np

import q5_wiener as Q


class TestWiener(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 96)
        self.clean = 0.5 + 0.3 * np.outer(np.sin(6 * x), np.cos(4 * x)) + 0.2 * (np.add.outer(x, x) > 1)
        self.sig = 0.06
        self.noisy = self.clean + self.sig * rng.standard_normal(self.clean.shape)

    def mse(self, a):
        return np.mean((a - self.clean) ** 2)

    def test_denoising_reduces_mse(self):
        out = Q.wiener_denoise(self.noisy, self.sig)
        self.assertLess(self.mse(out), 0.5 * self.mse(self.noisy))

    def test_zero_noise_returns_input(self):
        out = Q.wiener_denoise(self.clean, 1e-9)
        self.assertLess(np.abs(out - self.clean).max(), 1e-3)

    def test_gain_between_zero_and_one(self):
        _, p = Q.wiener_denoise(self.noisy, self.sig, return_parts=True)
        self.assertTrue(np.all(p["W"] > 0) and np.all(p["W"] <= 1 + 1e-12))

    def test_more_assumed_noise_means_more_smoothing(self):
        lo = Q.wiener_denoise(self.noisy, self.sig, noise_scale=0.5)
        hi = Q.wiener_denoise(self.noisy, self.sig, noise_scale=2.0)
        self.assertGreater(np.std(np.diff(lo, axis=1)), np.std(np.diff(hi, axis=1)))

    def test_underestimated_noise_leaves_residual_noise(self):
        lo = Q.wiener_denoise(self.noisy, self.sig, noise_scale=0.25)
        cor = Q.wiener_denoise(self.noisy, self.sig)
        self.assertGreater(np.std(np.diff(lo, axis=1)), np.std(np.diff(cor, axis=1)))

    def test_lowpass_preserves_constants(self):
        c = np.full((32, 32), 0.7)
        self.assertLess(np.abs(Q.lowpass_rect(c, 0.1, "gaussian") - 0.7).max(), 1e-12)   # symmetric padding keeps constants

    def test_plane_detrended_std_white_noise(self):
        n = np.random.default_rng(1).normal(0, 3.0, (24, 24)) + 5 + 0.1 * np.arange(24)[None, :]
        self.assertAlmostEqual(Q.plane_detrended_std(n)[0], 3.0, delta=0.6)


if __name__ == "__main__":
    unittest.main()
