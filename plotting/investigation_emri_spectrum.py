"""Investigation: where the EMRI's power sits in frequency.

The Fourier power of the EMRI's h+ and hx over the whole tapered record
(parameters.py), each normalised to unit total power. Top: power per unit ln f, on
log-spaced bins, so equal areas are equal power. Bottom: the cumulative fraction.
The peak frequency, the median and the 5-95 % range of each are printed.
Written to figures/investigations/emri_spectrum.pdf.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

import _emri_comparison as emri
from _colours import HAIRLINE, INK
from parameters import DT, N_SIGNAL

HERE = Path(__file__).parent
EDGES = np.logspace(-5, np.log10(0.5 / DT), 241)       # log-spaced bins, Hz
BANDS = (0.0, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, np.inf)     # Hz, for the printed fractions


def power(x):
    """Frequencies and the normalised one-sided Fourier power of x."""
    p = np.abs(np.fft.rfft(x)) ** 2
    p[1:] *= 2.0
    return np.fft.rfftfreq(len(x), DT), p / p.sum()


def binned(f, p):
    """Bin centres and power per unit ln f on EDGES."""
    sums, _ = np.histogram(f, bins=EDGES, weights=p)
    return np.sqrt(EDGES[1:] * EDGES[:-1]), sums / np.diff(np.log(EDGES))


def describe(name, f, p):
    """The peak, median, 5-95 % range and band fractions of one spectrum."""
    cdf = np.cumsum(p)
    q05, q50, q95 = (f[np.searchsorted(cdf, q)] for q in (0.05, 0.5, 0.95))
    centres, density = binned(f, p)
    bands = "  ".join(f"{lo * 1e3:g}-{hi * 1e3:g} mHz: {p[(f >= lo) & (f < hi)].sum():.3f}"
                      for lo, hi in zip(BANDS[:-1], BANDS[1:]))
    print(f"{name:<6} peak |FFT|^2 at {f[np.argmax(p)] * 1e3:.3f} mHz, "
          f"peak dP/dln f at {centres[np.argmax(density)] * 1e3:.3f} mHz, "
          f"median {q50 * 1e3:.3f} mHz, 5-95 % {q05 * 1e3:.3f}-{q95 * 1e3:.3f} mHz")
    print(f"       {bands}")
    return centres, density, cdf


def tracks(f, p, step=1e-5, floor=1e-6):
    """Each harmonic track's span, power and peak: the runs of step-wide bins
    holding more than floor times the fullest bin."""
    edges = np.arange(EDGES[0], EDGES[-1], step)
    on = np.histogram(f, edges, weights=p)[0] > floor * np.histogram(f, edges, weights=p)[0].max()
    starts = np.flatnonzero(np.diff(np.r_[0, on.astype(int)]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[on.astype(int), 0]) == -1) + 1
    for i, j in zip(starts, stops):
        sel = (f >= edges[i]) & (f < edges[j])
        print(f"  {edges[i] * 1e3:.2f}-{edges[j] * 1e3:.2f} mHz  power {p[sel].sum():.3f}  "
              f"peak {f[sel][np.argmax(p[sel])] * 1e3:.3f} mHz")


def main():
    plt.style.use(HERE / "paper.mplstyle")
    h_plus, h_cross = emri.waveform(N_SIGNAL)
    series = dict(plus=(r"$h_+$", h_plus, emri.POLARISATION_COLOURS["plus"]),
                  cross=(r"$h_\times$", h_cross, emri.POLARISATION_COLOURS["cross"]))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.375, 3.8), sharex=True,
                                      layout="constrained", height_ratios=(1.6, 1))
    for name, (label, x, colour) in series.items():
        f, p = power(x)
        centres, density, cdf = describe(name, f, p)
        top.plot(centres, density, color=colour, lw=1.0, label=label)
        bottom.plot(f[1:], cdf[1:], color=colour, lw=1.0)
        tracks(f, p)

    top.set(xscale="log", yscale="log", ylabel=r"$dP/d\ln f$ (normalised)",
            ylim=(1e-6, None))
    bottom.set(xscale="log", xlabel=r"$f$ [mHz]", ylabel="cumulative power",
               xlim=(4e-4, 6e-3), ylim=(0, 1.02))
    mhz = np.arange(1, 7)
    bottom.xaxis.set_major_locator(FixedLocator(mhz * 1e-3))
    bottom.xaxis.set_major_formatter(FixedFormatter([str(m) for m in mhz]))
    bottom.xaxis.set_minor_locator(NullLocator())
    for ax in (top, bottom):
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)
    top.legend(fontsize="small", frameon=False, labelcolor=INK)

    path = HERE / "figures" / "investigations" / "emri_spectrum.pdf"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
