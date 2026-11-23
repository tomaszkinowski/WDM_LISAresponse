"""Where the EMRI's power sits in frequency (the f = 2-5 mHz of appendix C).

The Fourier power of the EMRI's h+ and hx over the whole tapered record
(parameters.py), each normalised to unit total power: the peak frequency, the peak
of the power per unit ln f, the median, the 5-95 % range, the fraction in each of
BANDS, and each harmonic track's span and power. Printed and written to
numbers/emri_power.txt.
"""

from pathlib import Path

import numpy as np

import _emri_comparison as emri
from parameters import DT, N_SIGNAL

HERE = Path(__file__).parent
EDGES = np.logspace(-5, np.log10(0.5 / DT), 241)       # log-spaced bins, Hz
BANDS = (0.0, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, np.inf)     # Hz, for the band fractions


def power(x):
    """Frequencies and the normalised one-sided Fourier power of x."""
    p = np.abs(np.fft.rfft(x)) ** 2
    p[1:] *= 2.0
    return np.fft.rfftfreq(len(x), DT), p / p.sum()


def describe(name, f, p):
    """The peak, median, 5-95 % range and band fractions of one spectrum."""
    cdf = np.cumsum(p)
    q05, q50, q95 = (f[np.searchsorted(cdf, q)] for q in (0.05, 0.5, 0.95))
    sums, _ = np.histogram(f, bins=EDGES, weights=p)
    centres = np.sqrt(EDGES[1:] * EDGES[:-1])
    density = sums / np.diff(np.log(EDGES))
    bands = "  ".join(f"{lo * 1e3:g}-{hi * 1e3:g} mHz: {p[(f >= lo) & (f < hi)].sum():.3f}"
                      for lo, hi in zip(BANDS[:-1], BANDS[1:]))
    return [f"{name:<6} peak |FFT|^2 at {f[np.argmax(p)] * 1e3:.3f} mHz, "
            f"peak dP/dln f at {centres[np.argmax(density)] * 1e3:.3f} mHz, "
            f"median {q50 * 1e3:.3f} mHz, 5-95 % {q05 * 1e3:.3f}-{q95 * 1e3:.3f} mHz",
            f"       {bands}"]


def tracks(f, p, step=1e-5, floor=1e-6):
    """Each harmonic track's span, power and peak: the runs of step-wide bins
    holding more than floor times the fullest bin."""
    edges = np.arange(EDGES[0], EDGES[-1], step)
    binned = np.histogram(f, edges, weights=p)[0]
    on = binned > floor * binned.max()
    starts = np.flatnonzero(np.diff(np.r_[0, on.astype(int)]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[on.astype(int), 0]) == -1) + 1
    lines = []
    for i, j in zip(starts, stops):
        sel = (f >= edges[i]) & (f < edges[j])
        lines.append(f"  {edges[i] * 1e3:.2f}-{edges[j] * 1e3:.2f} mHz  power "
                     f"{p[sel].sum():.3f}  peak {f[sel][np.argmax(p[sel])] * 1e3:.3f} mHz")
    return lines


if __name__ == "__main__":
    h_plus, h_cross = emri.waveform(N_SIGNAL)
    lines = []
    for name, x in (("plus", h_plus), ("cross", h_cross)):
        f, p = power(np.asarray(x))
        lines += describe(name, f, p) + tracks(f, p)
    text = "\n".join(lines)
    print(text)
    (HERE / "numbers").mkdir(exist_ok=True)
    (HERE / "numbers" / "emri_power.txt").write_text(text + "\n")
