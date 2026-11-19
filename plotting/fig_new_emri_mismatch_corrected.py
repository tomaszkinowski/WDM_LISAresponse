"""New figure (unnumbered): figure 3's EMRI panel on its own - the mismatch of the
split_amp_sinc_first response against the time-domain reference on link 1 -> 2, as a
function of the tiling Delta T - with figure 8's production response for the same
EMRI behind it, muted. Hollow markers, the 3600-36000 s band and the Delta F axis
are figure 8's.

Plots the caches fig8_mismatch_vs_dT.py and investigation_fig8_delay_rate.py write.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import fig8_mismatch_vs_dT as fig8
from _colours import DARK, HAIRLINE, INK, MUTED, PALETTE
from parameters import DT

HERE = Path(__file__).parent
CORRECTED = HERE / "cache" / "investigation_fig8_split_amp_sinc_first.npz"
PRODUCTION = fig8.CACHE


def main():
    plt.style.use(HERE / "paper.mplstyle")
    corrected, production = dict(np.load(CORRECTED)), dict(np.load(PRODUCTION))
    assert np.array_equal(corrected["Nf"], production["Nf"])
    delta_t = corrected["Nf"] * DT
    leaks = delta_t < fig8.emri_shade_edge()

    fig, ax = plt.subplots(figsize=(3.375, 2.5), layout="constrained")
    ax.axvspan(*fig8.DT_BAND, color=PALETTE[3], alpha=0.3, lw=0, zorder=0.5)
    markers = dict(marker="o", ms=2.5, mec="white", mew=0.4)
    hollow = dict(ls="none", marker="o", ms=2.5, mfc="white", mew=0.7, zorder=5)
    series = ((production["EMRI"], MUTED, 0.8, r"0$^{\mathrm{th}}$-order response"),
              (corrected["EMRI"], DARK, 1.2, r"1$^{\mathrm{st}}$-order response"))
    for values, colour, lw, _ in series:
        ax.plot(delta_t, values, color=colour, lw=lw, zorder=4, **markers)
        ax.plot(delta_t[leaks], values[leaks], mec=colour, **hollow)

    ax.set(xscale="log", yscale="log", xlim=(delta_t[0] / 1.3, delta_t[-1] * 1.3),
           xlabel=r"$\Delta T$ [s]", ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)
    top = ax.secondary_xaxis("top", functions=(lambda x: 5e2 / np.asarray(x, float),
                                               lambda x: 5e2 / np.asarray(x, float)))
    top.set_xlabel(r"$\Delta F$ [mHz]")

    handles = [Line2D([], [], color=colour, lw=lw, **markers)
               for _, colour, lw, _ in series]
    handles.append(Line2D([], [], mec=INK, **hollow))
    labels = [label for *_, label in series] + [r"power leaks into $m = 0$"]
    fig.legend(handles, labels, loc="outside lower center", fontsize="small", ncols=2,
               handlelength=1.4, handletextpad=0.5, columnspacing=0.9, frameon=False)

    path = HERE / "figures" / "fig_new_emri_mismatch_corrected.pdf"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
