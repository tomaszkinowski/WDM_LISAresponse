"""Figure 6: the time-delay filters T_ell (top) and T'_ell (bottom) against the
delay, with the band -dT/2 <= delta < dT/2 shaded.

Ported from WDM_GW_wavelets/examples/time_delay_filters_demo.ipynb.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import jax

from WDM.code.discrete_wavelet_transform.WDM import WDM_transform
from WDM.code.time_delay_filters.filters import filter_functions

from _colours import PALETTE, HAIRLINE

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

# The filters depend only on the window shape, so any tiling gives the same curves;
# this one just supplies the default d and A_frac.
wdm = WDM_transform(dt=1.0, Nf=16, N=512)

# Each ell takes the next PALETTE slot, so at most five.
ELLS = [-15, -5, 0, 5, 15]
X_MAX = 16                  # plotted range of delta, in units of dT

x = np.linspace(-X_MAX, X_MAX, 40 * X_MAX + 1)          # delta / dT

# filter_functions takes l - delta/dT; index 0 is T', index 1 is T
filters = {ell: np.asarray(filter_functions(ell - x, wdm.d, wdm.A_frac)) for ell in ELLS}
Tl = {ell: f[1] for ell, f in filters.items()}
Tprimel = {ell: f[0] for ell, f in filters.items()}

fig, (ax_T, ax_Tp) = plt.subplots(nrows=2, sharex=True, figsize=(3.375, 3.375))
fig.subplots_adjust(hspace=0)

for ax in (ax_T, ax_Tp):
    ax.axvspan(-0.5, 0.5, color=HAIRLINE, lw=0, zorder=0)

for ell, c in zip(ELLS, PALETTE):
    ax_T.plot(x, Tl[ell], c=c, label=rf"${ell}$")
    ax_Tp.plot(x, Tprimel[ell], c=c)

ax_T.set_ylim(-0.28, 1.35)
ax_T.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_T.set_ylabel(r"$T_\ell(\delta)$")
# A blank-handle first entry puts "ell =" in front of the whole row.
handles, labels = ax_T.get_legend_handles_labels()
ax_T.legend([Line2D([], [], lw=0)] + handles, [r"$\ell=$"] + labels, ncols=len(ELLS) + 1,
            loc="upper center", handlelength=1.0, columnspacing=0.6, handletextpad=0.3,
            borderpad=0.3, frameon=True, fancybox=False, framealpha=1, facecolor="white",
            edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)

ax_Tp.set_ylim(0, 0.095)
ax_Tp.set_yticks([0.02, 0.04, 0.06, 0.08])
ax_Tp.set_ylabel(r"$T^\prime_\ell(\delta)$")
ax_Tp.set_xlim(-X_MAX, X_MAX)
ax_Tp.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_Tp.set_xlabel(r"$\delta/\Delta T$")

fig.align_ylabels((ax_T, ax_Tp))
fig.savefig(HERE / "figures" / "fig6_time_delay_filters.pdf")
