import unittest

import _path  # noqa: F401
import numpy as np

import common as C


class TestMetrics(unittest.TestCase):
    def test_psnr_known_value(self):
        a = np.zeros((8, 8))
        b = np.full((8, 8), 10.0)
        self.assertAlmostEqual(C.psnr(a, b), 10 * np.log10(255 ** 2 / 100), places=10)

    def test_psnr_identical_is_inf(self):
        a = np.random.default_rng(0).random((8, 8)) * 255
        self.assertEqual(C.psnr(a, a), float("inf"))

    def test_psnr_clips_test_image_only(self):
        a = np.full((4, 4), 255.0)
        b = np.full((4, 4), 300.0)
        self.assertEqual(C.psnr(a, b), float("inf"))        # clipped to 255 for the metric

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            C.psnr(np.zeros((4, 4)), np.zeros((4, 5)))
        with self.assertRaises(ValueError):
            C.ssim_gray(np.zeros((16, 16)), np.zeros((17, 16)))

    def test_ssim_identity_and_symmetry(self):
        rng = np.random.default_rng(1)
        a = rng.random((40, 40)) * 255
        b = np.clip(a + rng.normal(0, 20, a.shape), 0, 255)
        self.assertAlmostEqual(C.ssim_gray(a, a), 1.0, places=12)
        self.assertAlmostEqual(C.ssim_gray(a, b), C.ssim_gray(b, a), places=12)
        self.assertLess(C.ssim_gray(a, b), 1.0)

    def test_ssim_matches_skimage_if_available(self):
        try:
            from skimage.metrics import structural_similarity as sk
        except ImportError:
            self.skipTest("scikit-image not installed (only used as an external check)")
        rng = np.random.default_rng(2)
        a = rng.random((64, 64)) * 255
        b = np.clip(a + rng.normal(0, 15, a.shape), 0, 255)
        ref = sk(a, b, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        self.assertAlmostEqual(C.ssim_gray(a, b), ref, places=10)


if __name__ == "__main__":
    unittest.main()
