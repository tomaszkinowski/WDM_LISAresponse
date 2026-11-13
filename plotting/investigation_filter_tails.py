"""Investigation (not a paper figure): how fast the time-delay filters fall off
with lag, and how the Meyer window sets that. Two figures, each with T_ell (top)
and T'_ell (bottom), the largest |value| over delays -dT/2 <= delta < dT/2:
  filter_tails_d       d = 1, 2, 4, 8, 16 at A_frac = 0.25
  filter_tails_A_frac  A_frac = 0.1, 0.25, 0.4, 0.45 at d = 4

The window is C^(d-1) where it meets zero, so the product of neighbouring windows
behind T'_ell has tails falling as ell^-(d+1), and |Phi|^2 behind T_ell, smoother,
as ell^-(2d+1). The thin grey lines are those power laws, drawn through each
curve at ELL_GUIDE where it is still above the numerical floor (~1e-16). A_frac
leaves the power law alone but widens the core: a narrow transition in frequency
is long in time. The vertical line marks L = 25, the kernel's cutoff.

The filters come from filter_functions, which depends only on the window shape.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _colours import HAIRLINE, MUTED, PALETTE
from parameters import MAX_LAG_L

HERE = Path(__file__).parent
OUT = HERE / "figures" / "investigations"

ELLS = np.arange(1, 301)
N_DELTA = 21
ELL_GUIDE = 60                                  # where each guide line meets its curve
FLOOR = 1e-15                                   # below this the sums are rounding

SWEEPS = {
    "d": dict(windows=[(0.25, d) for d in (1, 2, 4, 8, 16)],
              label=lambda a, d: rf"$d = {d}$", figure="filter_tails_d"),
    "A_frac": dict(windows=[(a, 4) for a in (0.1, 0.25, 0.4, 0.45)],
                   label=lambda a, d: rf"$A_\mathrm{{frac}} = {a:g}$",
                   figure="filter_tails_A_frac"),
}


def filter_tails(A_frac, d):
    """max over delta of |T_ell| and |T'_ell| for every lag in ELLS."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from WDM.code.time_delay_filters.filters import filter_functions

    # x = ell - delta/dT over delays -dT/2 <= delta < dT/2; index 0 is T', 1 is T
    shifts = np.linspace(-0.5, 0.5, N_DELTA, endpoint=False)
    filters = np.abs(np.asarray(filter_functions(ELLS[:, None] - shifts[None, :], d, A_frac)))
    return filters[1].max(axis=1), filters[0].max(axis=1)


def plot(sweep):
    spec = SWEEPS[sweep]
    fig, (ax_T, ax_Tp) = plt.subplots(nrows=2, sharex=True, figsize=(3.375, 4.2),
                                      layout="constrained")
    for (A_frac, d), colour in zip(spec["windows"], PALETTE):
        T, T_prime = filter_tails(A_frac, d)
        for ax, values, power in ((ax_T, T, 2 * d + 1), (ax_Tp, T_prime, d + 1)):
            ax.plot(ELLS, values, color=colour, lw=1.0, label=spec["label"](A_frac, d))
            # The predicted power law, through the curve at ELL_GUIDE.
            anchor = values[ELL_GUIDE - 1]
            if anchor > 1e3 * FLOOR:
                guide = np.array([ELL_GUIDE / 2, ELL_GUIDE * 3])
                ax.plot(guide, anchor * (guide / ELL_GUIDE) ** -power, color=MUTED,
                        lw=0.6, ls=(0, (2, 2)), zorder=1)

    for ax, name in ((ax_T, r"$\max_\delta |T_\ell(\delta)|$"),
                     (ax_Tp, r"$\max_\delta |T'_\ell(\delta)|$")):
        ax.axvline(MAX_LAG_L, color=MUTED, lw=0.6, zorder=1)
        ax.axhspan(1e-18, FLOOR, color=HAIRLINE, lw=0, zorder=0)
        ax.set(xscale="log", yscale="log", ylim=(1e-17, 2), ylabel=name)
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)
    frame = dict(fontsize="small", frameon=True, fancybox=False, framealpha=1,
                 facecolor="white", edgecolor=HAIRLINE)
    guide = Line2D([], [], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    handles, labels = ax_T.get_legend_handles_labels()
    ax_T.legend(handles + [guide], labels + [r"$\propto \ell^{-(2d+1)}$"], loc="lower left",
                **frame).get_frame().set_linewidth(0.6)
    ax_Tp.legend([guide], [r"$\propto \ell^{-(d+1)}$"], loc="lower left",
                 **frame).get_frame().set_linewidth(0.6)
    ax_Tp.set(xlim=(1, ELLS[-1]), xlabel=r"lag $\ell$")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{spec['figure']}.pdf")
    print(f"wrote {OUT / spec['figure']}.pdf")


if __name__ == "__main__":
    plt.style.use(HERE / "paper.mplstyle")
    for sweep in SWEEPS:
        plot(sweep)
