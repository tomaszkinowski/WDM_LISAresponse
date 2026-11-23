"""Figure 5: the Meyer window in the frequency domain (top) and the time domain
(bottom) for three steepness parameters d.

Both panels are dimensionless: sqrt(dF) Phi(f) = M(f/dF) against f/dF, and
sqrt(dT) phi(t) against t/dT, each with unit norm in its own variable. With
u = f/dF, tau = t/dT and dF dT = 1/2,

    sqrt(dT) phi(t) = (1/sqrt 2) int du M(u) exp(i pi u tau).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from WDM.code.utils.Meyer import Meyer

from _colours import PALETTE, HAIRLINE

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

A_FRAC = 0.25
B_FRAC = 1. - 2. * A_FRAC
D_VALS = [1, 4, 16]
LINESTYLES = ["-", "--", "-."]
TAU_MAX = 16                # plotted range of t/dT

# With dOmega = 2A + B = 1, Meyer(u) is M(u) itself.
u = np.linspace(-1., 1., 4001)
u_half = np.linspace(0., A_FRAC + B_FRAC, 4001)     # M is even with this support
tau = np.linspace(-TAU_MAX, TAU_MAX, 3201)


def scaled_phi(d):
    """sqrt(dT) phi(t) on the tau grid, by trapezoid quadrature over u >= 0."""
    M = np.asarray(Meyer(u_half, d=d, A=A_FRAC, B=B_FRAC))
    integrand = M[None, :] * np.cos(np.pi * u_half[None, :] * tau[:, None])
    return np.sqrt(2.) * np.trapezoid(integrand, u_half, axis=1)


def main():
    fig, (ax_f, ax_t) = plt.subplots(nrows=2, figsize=(3.375, 3.2))   # one journal column

    for d, c, ls in zip(D_VALS, PALETTE, LINESTYLES):
        ax_f.plot(u, Meyer(u, d=d, A=A_FRAC, B=B_FRAC), c=c, ls=ls)
        ax_t.plot(tau, scaled_phi(d), c=c, ls=ls, label=rf"$d={d}$")

    ax_f.set_xlim(-1., 1.)
    ax_f.set_xticks(np.arange(-1., 1.01, 0.25))
    ax_f.xaxis.set_major_formatter(lambda x, _: rf"${x:g}$")
    ax_f.set_ylim(0., 1.05)
    ax_f.set_xlabel(r"$f/\Delta F$")
    ax_f.set_ylabel(r"$\sqrt{\Delta F}\,\Phi(f)$")

    ax_t.set_xlim(-TAU_MAX, TAU_MAX)
    ax_t.set_xticks(np.arange(-15, 16, 5))
    ax_t.set_xlabel(r"$t/\Delta T$")
    ax_t.set_ylabel(r"$\sqrt{\Delta T}\,\phi(t)$")
    ax_t.legend(loc="upper right")

    for ax in (ax_f, ax_t):
        ax.grid(color=HAIRLINE, lw=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout(h_pad=0.6)
    fig.savefig(HERE / "figures" / "fig5_meyer_window.pdf")


if __name__ == "__main__":
    main()
