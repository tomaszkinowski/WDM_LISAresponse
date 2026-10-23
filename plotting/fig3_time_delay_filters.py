"""Figure 3: the time-delay filters T_ell (top) and T'_ell (bottom) against the
delay, with the band -dT/2 <= delta < dT/2 shaded.

Ported from WDM_GW_wavelets/examples/time_delay_filters_demo.ipynb.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from WDM.code.discrete_wavelet_transform.WDM import WDM_transform
from WDM.code.time_delay_filters.filters import (time_delay_filter_Tl_reference,
                                                 time_delay_filter_Tprimel_reference)

from _colours import PALETTE, HAIRLINE

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

wdm = WDM_transform(dt=1.0, Nf=16, N=512)

# Each ell takes the next PALETTE slot, so at most five.
ELLS = [-15, -5, 0, 5, 15]
# The filters repeat every N*dt = 32 dT, so +-16 dT is exactly one period: every
# ell in [-16, 16) peaks inside it and nothing is drawn twice.
X_MAX = 16                  # plotted range of delta, in units of dT

delta_vals = np.linspace(-X_MAX, X_MAX, 40 * X_MAX + 1) * wdm.dT

Tl = {ell: np.array([time_delay_filter_Tl_reference(ell, delta, wdm.freqs, wdm.window_FD,
                                                    wdm.dT, wdm.df)
                     for delta in delta_vals])
      for ell in ELLS}
Tprimel = {ell: np.array([time_delay_filter_Tprimel_reference(ell, delta, wdm.freqs,
                                                              wdm.window_FD, wdm.dT, wdm.dF,
                                                              wdm.N, wdm.df)
                          for delta in delta_vals])
           for ell in ELLS}

fig, (ax_T, ax_Tp) = plt.subplots(nrows=2, sharex=True, figsize=(3.375, 3.375))
fig.subplots_adjust(hspace=0)

for ax in (ax_T, ax_Tp):
    ax.axvspan(-0.5, 0.5, color=HAIRLINE, lw=0, zorder=0)

x = delta_vals / wdm.dT
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
fig.savefig(HERE / "figures" / "fig3_time_delay_filters.pdf")
