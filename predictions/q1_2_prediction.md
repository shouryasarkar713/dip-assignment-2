# Q1.2 pre-implementation analytical prediction (written BEFORE any Q1.2 code was run)

Timestamp of writing: 2026-10-27 11:17:12 +0530  (file written before src/q1_dft.py existed)

Notation: image f(x,y), N x N (N=512), unnormalised DFT F(u,v)=sum f e^{-2 pi i (ux+vy)/N},
F = |F| e^{i phi}.  Inverse has 1/N^2.

## (a) Phase of the test image + flat magnitude c
Reconstruction  f_hat = IDFT[ c e^{i phi} ] = c * IDFT[ F/|F| ].

1. Shift theorem: an impulse at (x0,y0) has F = e^{-2 pi i (u x0+v y0)/N}, i.e. |F|=1 and
   phi linear in (u,v) with slope given by the position.  Hence for a single impulse the
   flat-magnitude reconstruction returns the impulse in the SAME place: position is carried
   by phase alone.
2. Superposition does not hold for phase (F/|F| is nonlinear in f), so for a general image
   F/|F| = F * (1/|F|) and, by the convolution theorem,
       f_hat = c * ( f  (*)  w ),   w = IDFT[ 1/|F| ]  (zero-phase, real, even kernel),
   a circular convolution of f with a *whitening* (inverse-magnitude) kernel.
3. For natural images |F| ~ 1/|k|^alpha, alpha ~ 1-2, so 1/|F| ~ |k|^alpha is a strong
   high-pass.  Predicted appearance: low-frequency tonal content / smooth regions vanish,
   edges, contours and fine texture are retained at their original positions and are
   strongly emphasised -> an "edge map"-like, sign-alternating, heavy-tailed (sparse,
   high-kurtosis) image, with noisy-looking texture; the object is recognisable by its outlines.
4. Exact (checkable) consequences:
   - f_hat is real (flat magnitude is even, phase of a real image is odd -> Hermitian).
   - Parseval: sum f_hat^2 = c^2 (every bin has modulus c: (1/N^2) sum |c|^2 N^2 = c^2).
   - DC term: phi(0,0)=0 for a positive-mean image, so mean(f_hat)=c/N^2 (negligible).
   - The radially averaged power spectrum of f_hat is flat (= c^2) by construction.
   Heuristic (not exact): the smoothed |f_hat| correlates positively with the gradient magnitude
   of f (Pearson r > 0.3), and the correlation of f_hat with f itself is close to 0 (|r| < 0.1)
   because the DC/low-frequency content that dominates the variance of f is removed.

### White-noise magnitude variant
If c is replaced by the magnitude R(u,v) of a white-noise image, R is Rayleigh-like with
E[R]=s sqrt(pi/2), E[R^2]=2 s^2, independent of phi.  Then f_hat_n = IDFT[R e^{i phi}] and by
Parseval the correlation coefficient with the flat-magnitude reconstruction is
   rho = E[R] / sqrt(E[R^2]) = sqrt(pi)/2 = 0.886  (tolerance: +-0.01).
So f_hat_n = 0.886-correlated version of f_hat plus speckle: same structure, extra random
modulation (|R| is conjugate-symmetric so only half the bins are independent).

### Why flat magnitude isolates structure better than swapping with another natural image
Swapping gives IDFT[ |G| e^{i phi_F} ] = f (*) IDFT[|G|/|F|]: the kernel is not white; |G| ~ 1/k^alpha again
imposes the other image's smooth, low-frequency-dominated amplitude envelope and its own
coherent amplitude features, so the result is a blend (the second image's tonal distribution
modulating the first image's structure).  Flat magnitude removes every amplitude cue.

## (b) Original magnitude, zero phase
f_hat = IDFT[ |F| ]: |F| is real, non-negative and even -> f_hat is real and even,
f_hat(x,y) = f_hat(-x,-y) (mod N).  All bins add in phase at the origin, so the global maximum
is at (0,0) (the corners of the un-shifted array) with value sum|F|/N^2 >= |mean|.  No object
localisation survives (no linear phase = no position info): predicted result is a blurry,
centrally symmetric, low-pass-looking "blob" concentrated around the origin (periodic wrap), with
the structure of the *autocorrelation-like* function (square-root of the power spectrum), i.e. it
looks like a smooth halo of the image's spectral envelope, not like the picture.
