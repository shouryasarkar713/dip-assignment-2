"""Shared helpers: paths, image I/O, metrics (PSNR, SSIM written from scratch), JSON/env utilities.

Everything numerical in the questions is implemented in NumPy in the q*.py files; this module
only holds bookkeeping and the evaluation metrics.
"""
import json
import os
import platform
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
PROVIDED = os.path.join(DATA, "provided")
PHOTOS = os.path.join(DATA, "photos")
DERIVED = os.path.join(DATA, "derived")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
PREDICTIONS = os.path.join(ROOT, "predictions")

PHOTO_FILES = {
    "A": "IMG_20261026_195540763.jpg",
    "B": "IMG_20261026_195556198.jpg",
    "C": "IMG_20261026_195635983.jpg",
    "D": "IMG_20261026_195702582.jpg",
}


def ensure_dirs():
    for d in (RESULTS, FIGS, DERIVED):
        os.makedirs(d, exist_ok=True)


def load_gray(path):
    """Load as float64 grayscale in the native 0..255 scale."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def save_png(arr, path):
    a = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def luma(rgb):
    """ITU-R BT.601 luma, same scale as the input."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


# ----------------------------------------------------------------------------- metrics
def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def psnr(ref, test, data_range=255.0, clip=True):
    """PSNR in dB.  `test` is clipped to [0, data_range] (clip only for metric evaluation)."""
    ref = np.asarray(ref, float)
    test = np.asarray(test, float)
    if ref.shape != test.shape:
        raise ValueError("PSNR: shape mismatch %s vs %s" % (ref.shape, test.shape))
    if clip:
        test = np.clip(test, 0, data_range)
    m = np.mean((ref - test) ** 2)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / m))


def _gauss1d(size=11, sigma=1.5):
    r = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-r ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _sep_valid(img, k):
    """'valid' separable correlation with 1-D kernel k along both axes (shifted-slice sums)."""
    n = len(k)
    h, w = img.shape
    tmp = np.zeros((h, w - n + 1))
    for i in range(n):
        tmp += k[i] * img[:, i:i + w - n + 1]
    out = np.zeros((h - n + 1, w - n + 1))
    for i in range(n):
        out += k[i] * tmp[i:i + h - n + 1, :]
    return out


def ssim_gray(ref, test, data_range=255.0, size=11, sigma=1.5, K1=0.01, K2=0.03):
    """Mean SSIM (Wang et al. 2004), Gaussian 11x11 sigma=1.5 window, 'valid' region, population
    covariance.  Both images are clipped to [0, data_range] first."""
    x = np.clip(np.asarray(ref, float), 0, data_range)
    y = np.clip(np.asarray(test, float), 0, data_range)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("SSIM: need two equally shaped 2-D arrays")
    k = _gauss1d(size, sigma)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    mx, my = _sep_valid(x, k), _sep_valid(y, k)
    sxx = _sep_valid(x * x, k) - mx * mx
    syy = _sep_valid(y * y, k) - my * my
    sxy = _sep_valid(x * y, k) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return float(s.mean())


def ssim(ref, test, data_range=255.0):
    """SSIM for 2-D or H x W x C images (channel-wise mean)."""
    ref = np.asarray(ref, float)
    if ref.ndim == 2:
        return ssim_gray(ref, test, data_range)
    return float(np.mean([ssim_gray(ref[..., c], np.asarray(test)[..., c], data_range)
                          for c in range(ref.shape[2])]))


# ----------------------------------------------------------------------------- bookkeeping
def env_info():
    import scipy
    import matplotlib
    cpu = platform.processor()
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "cpu_reported_by_python": cpu,
        "logical_cpus": os.cpu_count(),
        "cpu_marketing_name_measured_on_dev_machine": "13th Gen Intel(R) Core(TM) i5-1340P (12 cores/16 threads)",
        "ram_measured_on_dev_machine_GB": 15.6,
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def save_json(obj, name):
    ensure_dirs()
    p = os.path.join(RESULTS, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1)
    return p


def load_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t


def fig_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
                         "axes.grid": False, "axes.titlesize": 9, "figure.constrained_layout.use": True})
    return plt


def savefig(plt, fig, name):
    ensure_dirs()
    p = os.path.join(FIGS, name)
    fig.savefig(p)
    plt.close(fig)
    return p
