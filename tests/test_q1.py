import unittest

import _path  # noqa: F401
import numpy as np

import q1_dft as Q


class TestDFT(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_matches_literal_double_sum(self):
        x = self.rng.normal(size=(12, 12))
        self.assertLess(np.abs(Q.dft2(x) - Q.dft2_direct(x)).max(), 1e-10)     # independent of W and of numpy

    def test_matches_numpy(self):
        x = self.rng.normal(size=(64, 64))
        self.assertLess(np.abs(Q.dft2(x) - np.fft.fft2(x)).max(), 1e-9)

    def test_inverse_roundtrip(self):
        x = self.rng.normal(size=(32, 32))
        self.assertLess(np.abs(Q.idft2(Q.dft2(x)) - x).max(), 1e-12)

    def test_parseval(self):
        x = self.rng.normal(size=(32, 32))
        F = Q.dft2(x)
        self.assertAlmostEqual(np.sum(x ** 2), np.sum(np.abs(F) ** 2) / 32 ** 2, places=9)

    def test_conjugate_symmetry_real_only(self):
        x = self.rng.normal(size=(16, 16))
        F = Q.dft2(x)
        self.assertLess(np.abs(F - np.conj(Q.conj_partner(F))).max(), 1e-11)
        z = x + 1j * self.rng.normal(size=(16, 16))
        Fz = Q.dft2(z)
        self.assertGreater(np.abs(Fz - np.conj(Q.conj_partner(Fz))).max(), 1e-3)      # must fail for complex input

    def test_conjugacy_class_counts(self):
        self.assertEqual(Q.conjugacy_class_counts(512)["conjugacy_classes"], 131074)
        self.assertEqual(Q.conjugacy_class_counts(512)["real_dof"], 512 ** 2)
        self.assertEqual(Q.conjugacy_class_counts(8)["conjugacy_classes"], 34)         # (64+4)/2
        self.assertEqual(Q.conjugacy_class_counts(7)["fixed_points"], 1)               # odd N: only DC
        N = 6
        seen = set()
        n = 0
        for u in range(N):                                                             # independent brute-force orbit count
            for v in range(N):
                if (u, v) not in seen:
                    seen.add((u, v))
                    seen.add(((-u) % N, (-v) % N))
                    n += 1
        self.assertEqual(Q.conjugacy_class_counts(6)["conjugacy_classes"], n)

    def test_shift_theorem(self):
        x = self.rng.normal(size=(16, 16))
        for a, b in [(3, 5), (-2, 7), (15, 1)]:
            lhs = Q.dft2(np.roll(x, (a, b), axis=(0, 1)))
            self.assertLess(np.abs(lhs - Q.circ_shift_phase(Q.dft2(x), a, b)).max(), 1e-10)

    def test_convolution_theorem_against_direct_sum(self):
        N = 12
        x = self.rng.normal(size=(N, N))
        K = self.rng.normal(size=(3, 3))
        g = Q.embed_kernel(K, N, (1, 1))
        via = np.real(Q.idft2(Q.dft2(x) * Q.dft2(g)))
        ref = np.zeros((N, N))                     # literal definition (f*g)[r,c] = sum f[a,b] g[(r-a)%N,(c-b)%N]
        for r in range(N):
            for c in range(N):
                for a in range(N):
                    for b in range(N):
                        ref[r, c] += x[a, b] * g[(r - a) % N, (c - b) % N]
        self.assertLess(np.abs(via - ref).max(), 1e-10)

    def test_sinusoid_recovery(self):
        comps = [(3, 4, 2.0, 0.3), (-5, 2, 1.0, -1.0), (7, -6, 0.5, 2.0)]
        f = Q.synth(64, comps)
        pk = Q.pick_peaks(Q.dft2(f), 64, 3)
        got = {(p["u"], p["v"]): p for p in pk}
        for u, v, A, ph in comps:
            if u < 0 or (u == 0 and v < 0):
                u, v, ph = -u, -v, -ph
            self.assertIn((u, v), got)
            self.assertAlmostEqual(got[(u, v)]["A"], A, places=9)
            self.assertAlmostEqual(Q.wrap_pi(got[(u, v)]["phi"] - ph), 0, places=9)

    def test_padding_equivalence_and_aliasing(self):
        N, M = 20, 5
        x = self.rng.normal(size=(N, N))
        K = self.rng.normal(size=(M, M))
        full = np.zeros((N + M - 1,) * 2)
        for m in range(M):
            for n in range(M):
                full[m:m + N, n:n + N] += K[m, n] * x

        def circ(P):
            xp = np.zeros((P, P))
            xp[:N, :N] = x
            kp = np.zeros((P, P))
            kp[:M, :M] = K
            return np.real(np.fft.ifft2(np.fft.fft2(xp) * np.fft.fft2(kp)))
        self.assertLess(np.abs(circ(N + M - 1) - full).max(), 1e-10)           # minimal exact size
        P = N + M - 2                                                          # one sample less aliases
        self.assertGreater(np.abs(circ(P)[:P, :P] - full[:P, :P]).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
