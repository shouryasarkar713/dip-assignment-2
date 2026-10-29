import unittest

import _path  # noqa: F401
import numpy as np

import q3_homomorphic as Q


class TestHomomorphic(unittest.TestCase):
    def test_log_turns_product_into_sum(self):
        rng = np.random.default_rng(0)
        L = rng.random((8, 8)) + 0.1
        R = rng.random((8, 8)) + 0.1
        self.assertLess(np.abs(np.log(L * R) - (np.log(L) + np.log(R))).max(), 1e-14)

    def test_filter_limits(self):
        H = Q.hom_filter_grid((64, 64), 0.4, 2.5, 0.05)
        self.assertAlmostEqual(H[0, 0], 0.4)                     # DC gain = gamma_L
        self.assertAlmostEqual(H[32, 32], 2.5, places=6)         # Nyquist corner ~ gamma_H
        self.assertTrue(np.all(H >= 0.4 - 1e-12) and np.all(H <= 2.5 + 1e-12))

    def test_identity_when_gains_are_one(self):
        Y = np.random.default_rng(0).random((24, 24)) * 0.9 + 0.05
        out = Q.homomorphic(Y, 1.0, 1.0, 0.1)
        self.assertLess(np.abs(out - Y).max(), 1e-12)

    def test_constant_image_maps_to_power_of_dc_gain(self):
        Y = np.full((16, 16), 0.3)
        out = Q.homomorphic(Y, 0.5, 2.0, 0.05)
        expect = np.exp(0.5 * np.log(0.3 + Q.EPS)) - Q.EPS
        self.assertLess(np.abs(out - expect).max(), 1e-10)

    def test_illumination_flattening_on_synthetic_model(self):
        rng = np.random.default_rng(0)
        L = np.exp(np.linspace(-3, 0, 128))[:, None] * np.ones((1, 128))
        R = np.exp(0.2 * rng.standard_normal((128, 128)))
        I = L * R
        I = I / I.max()
        out = Q.homomorphic(I, 0.3, 1.5, 0.05)
        rows = lambda a: a.mean(axis=1)
        self.assertLess(np.ptp(np.log(rows(out) + 1e-9)), 0.7 * np.ptp(np.log(rows(I) + 1e-9)))

    def test_hist_equalise_monotone_and_range(self):
        Y = np.random.default_rng(0).random((64, 64)) ** 3
        E = Q.hist_equalise(Y)
        self.assertGreaterEqual(E.min(), 0)
        self.assertLessEqual(E.max(), 1.0 + 1e-12)
        order = np.argsort(Y.ravel(), kind="stable")
        self.assertTrue(np.all(np.diff(E.ravel()[order]) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
