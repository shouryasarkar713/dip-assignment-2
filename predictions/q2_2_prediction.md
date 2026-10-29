# Q2.2 pre-implementation analytical prediction: ringing of the ideal low-pass filter
# (written BEFORE any Q2 code was run)

Timestamp of writing: 2026-10-27 11:17:29 +0530  (file written before src/q2_filtering.py existed)

Setup: s(x,y) = A=50 background, B=200 centred 256x256 square in a 512x512 image, zero-padded to
P=2N=1024, spectrum centred, H radially symmetric, D0 in padded-grid bins (cycles per P pixels).

## 1-D reduction
Far from the square's top/bottom edges (a row through the middle, away from the corners), s varies
only with x, so its 2-D spectrum is confined to the line v=0; only H(u,0) matters:
   ILPF: H(u,0)=1 for |u|<=D0, 0 otherwise  ->  1-D ideal low-pass with cutoff fc = D0/P cycles/pixel.
Impulse response h(x)=2 fc sinc(2 fc x)= sin(2 pi fc x)/(pi x).  Unit-step response
   F(t) = 1/2 + Si(2 pi fc t)/pi,   Si(z)=int_0^z sin(w)/w dw,  t = x - x_edge (pixels).
Single edge at x_e:  g(x) = A + (B-A) F(x-x_e).
Square with edges x1<x2 (width W=256): g(x)=A+(B-A)[F(x-x1)-F(x-x2)]  (closed form, two Si terms).

## Predictions
(i) Overshoot: F peaks where d/dt F = sin(2 pi fc t)/(pi t)=0 first -> 2 pi fc t = pi, t* = 1/(2 fc)=P/(2 D0).
    F(t*) = 1/2 + Si(pi)/pi = 1/2 + 1.851937/pi = 1.08949  ->  overshoot 8.95 % of the jump (B-A),
    INDEPENDENT of D0 (a Gibbs phenomenon: it does not shrink as the bandwidth grows).
(ii) First-peak distance t* = P/(2 D0): D0=10: 51.2 px, 20: 25.6 px, 40: 12.8 px, 80: 6.4 px.
    Discrete-grid refinement: the discrete ideal filter keeps 2 D0 + 1 bins (u=-D0..D0), equivalent
    to fc_eff=(D0+1/2)/P, so t*_disc ~ P/(2 D0+1): 48.8, 25.0, 12.64, 6.35 px.
Expected deviations (stated beforehand): (a) the second edge 256 px away adds a tail
    ~ (B-A)/(pi * 2 pi fc W) cos(.) -> for D0=10 about +-2 % of the jump, smaller for larger D0;
    (b) the zero-padding boundary (jump 50 -> 0 at the image border) contributes its own Gibbs
    ripple of amplitude ~0.09*50/(150)=3 % of the jump near the edge and decaying ~1/distance;
    (c) overshoot is read from pixel samples (spacing 1) so the peak height is under-estimated for
    large D0 (peak width ~ t*), and the max over the whole row may be set by another ripple.
Undershoot on the dark side mirrors the overshoot at -t*.

## Gaussian low-pass
H(u,0)=exp(-u^2/(2 D0'^2)) -> spatial Gaussian with sigma_s = P/(2 pi D0') pixels.  Its unit-step
response is a Gaussian CDF 1/2 + 1/2 erf(t/(sigma_s sqrt2)), monotone: overshoot = 0 % (a
non-negative kernel summing to one cannot exceed max(s)).  Predicted measured overshoot: 0.0 %.

## Corner profile
Near a corner the image is not 1-D: its spectrum has energy off the v=0 axis, where the radial H
cuts at |u|<sqrt(D0^2-v^2) < D0.  The relevant kernel is the 2-D jinc, the response is a product-like
mixture of x- and y-ripples with *different effective cutoffs*, so the 8.95 % / P/(2 D0) 1-D
prediction no longer holds (expect lower/modulated overshoot, ripples curving around the corner).
