"""Investigation beside figure 13: why second-generation X shows high-frequency
structure that first-generation X and a single link do not, on a toy signal.

With frozen, equal arms, X1 = (1 - D^2) and X2 = (1 - D^4) X1, with D the one-way
delay L/c. For a harmonic at f, x(t) - x(t - tau) = g xdot(t - tau/2) exactly, with
g = sin(pi f tau) / (pi f): a derivative, times a gain that tends to tau when
pi f tau << 1. The derivative weights the n-th harmonic by n against the
fundamental, so X1 carries n and X2 n^2.

The signal is a sum of harmonics of F0 whose band powers match the EMRI link's
(figure 12), so the delays and derivatives are exact and analytic. Left, over the
same third of a day as figure 13's zoom: the signal, and X1 and X2 from the exact
delays (coloured) and in the small-angle limit g = tau (thin, dashed). Right: the exact
transfer functions |1 - D^2| and |(1 - D^2)(1 - D^4)| against that limit's
2 pi f tau, with the harmonics marked. The derivative form with the exact g is
checked against the exact delays and printed. Written to
figures/investigations/fig13_tdi_harmonics_toy.pdf.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import DARK, HAIRLINE, INK, MUTED, PALETTE

HERE = Path(__file__).parent

ARM = 2.5e9 / 299792458.0                       # one-way light travel time, s
TAU_1, TAU_2 = 2 * ARM, 4 * ARM                 # the X1 and X2 differencing delays
F0 = 5e-4                                       # fundamental, Hz
# Powers of the harmonics at 0.5-4.5 mHz: the EMRI link's fraction in 0-1, 1-2,
# 2-3 and 3-5 mHz (0.10, 0.73, 0.12, 0.05), shared equally within each band.
POWERS = np.array([0.10] + [0.73 / 2] * 2 + [0.12 / 2] * 2 + [0.05 / 4] * 4)
AMPLITUDES = np.sqrt(POWERS)
FREQUENCIES = F0 * np.arange(1, len(POWERS) + 1)
PHASES = np.random.default_rng(1).uniform(0, 2 * np.pi, len(POWERS))
T = np.arange(0.0, 86400.0 / 3, 1.0)            # a third of a day, 1 s steps

COLOURS = dict(x=DARK, X1=PALETTE[0], X2=PALETTE[1])
GUIDE = dict(color=DARK, lw=0.4, ls=(0, (4, 2)))


def signal(t, derivative=0, gain=1.0):
    """The sum of harmonics, or its ``derivative``-th time derivative, each
    harmonic scaled by ``gain`` (one value per harmonic)."""
    omega = 2 * np.pi * FREQUENCIES
    phase = np.outer(t, omega) + PHASES + derivative * np.pi / 2
    return np.sum(gain * AMPLITUDES * omega ** derivative * np.sin(phase), axis=1)


def gain(tau):
    """Each harmonic's exact gain g = sin(pi f tau) / (pi f) of a differencing by tau."""
    return np.sin(np.pi * FREQUENCIES * tau) / (np.pi * FREQUENCIES)


def exact():
    """x, X1 = (1 - D^2) x and X2 = (1 - D^4) X1, from exact delays."""
    x = signal(T)
    x1 = x - signal(T - TAU_1)
    x2 = x1 - (signal(T - TAU_2) - signal(T - TAU_2 - TAU_1))
    return x, x1, x2


def as_derivatives(g1, g2):
    """X1 and X2 as derivatives times the gains g1 and g2 of their two differencings,
    at the midpoint of the delays."""
    x1 = signal(T - TAU_1 / 2, 1, g1)
    x2 = signal(T - (TAU_1 + TAU_2) / 2, 2, g1 * g2)
    return x1, x2


def main():
    plt.style.use(HERE / "paper.mplstyle")
    fig = plt.figure(figsize=(7.0, 4.4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=(1.5, 1))
    axes = left.subplots(3, 1, sharex=True)

    x, x1, x2 = exact()
    for name, ex, de in zip(("X1", "X2"), (x1, x2), as_derivatives(gain(TAU_1), gain(TAU_2))):
        print(f"{name}, derivative with the exact gain: max |difference| / max |exact| = "
              f"{np.max(np.abs(ex - de)) / np.max(np.abs(ex)):.1e}")
    d1, d2 = as_derivatives(TAU_1, TAU_2)
    t_days = T / 86400.0
    axes[0].plot(t_days, x, color=COLOURS["x"], lw=0.6, label=r"$x$, a single link")
    axes[0].set_ylabel("$x$")
    for ax, name, ex, de, tex in ((axes[1], "X1", x1, d1, r"$X_1$"),
                                  (axes[2], "X2", x2, d2, r"$X_2$")):
        ax.plot(t_days, ex, color=COLOURS[name], lw=1.2, label=f"{tex}, exact delays")
        ax.plot(t_days, de, **GUIDE)
        ax.set_ylabel(tex)
        print(f"{name}, small-angle derivative: max |difference| / max |exact| = "
              f"{np.max(np.abs(ex - de)) / np.max(np.abs(ex)):.1e}")
    axes[-1].set(xlabel=r"$t$ [days]", xlim=(t_days[0], t_days[-1]))

    # Transfer functions, exact against the small-angle limit.
    ax = right.subplots()
    f = np.logspace(-4, -1.3, 800)
    h1 = np.abs(2 * np.sin(np.pi * f * TAU_1))
    h2 = h1 * np.abs(2 * np.sin(np.pi * f * TAU_2))
    for h, law, name in ((h1, 2 * np.pi * f * TAU_1, "X1"),
                         (h2, (2 * np.pi * f) ** 2 * TAU_1 * TAU_2, "X2")):
        ax.loglog(f * 1e3, h, color=COLOURS[name], lw=1.4)
        ax.loglog(f * 1e3, law, **GUIDE | dict(lw=0.7))
    for fn in FREQUENCIES * 1e3:
        ax.axvline(fn, color=MUTED, lw=0.3, alpha=0.6, zorder=0)
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set(xlabel=r"$f$ [mHz]", ylabel="gain against the link",
           xlim=(f[0] * 1e3, f[-1] * 1e3), ylim=(1e-3, 10))
    handles = [line for a in axes for line in a.get_lines() if line.get_label()[0] != "_"]
    handles.append(plt.Line2D([], [], label=r"small-angle approximation: $\tau_1 \dot{x}$, $\tau_1 \tau_2 \ddot{x}$",
                              **GUIDE | dict(lw=0.8)))
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize="small",
               frameon=False)

    for a in (*axes, ax):
        a.grid(color=HAIRLINE, lw=0.4)
        a.set_axisbelow(True)

    path = HERE / "figures" / "investigations" / "fig13_tdi_harmonics_toy.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
