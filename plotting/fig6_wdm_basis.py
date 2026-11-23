"""Figure 6: three WDM wavelets in the time domain (top), the frequency domain
(right), and the tiles they occupy in the time-frequency plane (centre).

Ported from cell 11 of WDM_GW_wavelets/examples/plotting_the_mother_wavelet.ipynb.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches, patheffects

from WDM.code.discrete_wavelet_transform.WDM import WDM_transform

from _colours import PALETTE, HAIRLINE

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

wdm = WDM_transform(dt=1.0, Nf=16, N=512)

N_VALS = [8, 16, 24]
M_VALS = [1, 4, 7]
LINESTYLES = ["-", "--", "-."]

# Tile edges sit halfway between the wavelet centres.
t_ticks = [(i + 0.5) * wdm.dT for i in range(wdm.Nt)]
f_ticks = [(i + 0.5) * wdm.dF for i in range(wdm.Nf)]


def tile_boxes(n, m):
    """The (t0, t1, f0, f1) box(es) wavelet (n, m) occupies in the plot.

    n = 0 wraps round the end of the time axis, so it is drawn as two boxes.
    m = 0 packs the zero-frequency (n < Nt/2) and Nyquist (n >= Nt/2) wavelets
    into one row, each twice as wide in time.
    """
    assert 0 <= n < wdm.Nt, f"n={n} is out of bounds for Nt={wdm.Nt}"
    assert 0 <= m < wdm.Nf, f"m={m} is out of bounds for Nf={wdm.Nf}"
    T, half = wdm.T, wdm.Nt // 2

    if m > 0:
        f0, f1 = f_ticks[m - 1], f_ticks[m]
        if n == 0:
            return [(0, t_ticks[0], f0, f1), (t_ticks[-1], T, f0, f1)]
        return [(t_ticks[n - 1], t_ticks[n], f0, f1)]

    if n < half:
        f0, f1, k = 0.0, f_ticks[0], n
    else:
        f0, f1, k = f_ticks[-1], wdm.f_Ny, n - half
    if k == 0:
        return [(0, t_ticks[1], f0, f1), (t_ticks[-1], T, f0, f1)]
    return [(t_ticks[2 * k - 1], t_ticks[2 * k + 1], f0, f1)]


fig = plt.figure(figsize=(3.375, 3.375))   # one journal column
gs = gridspec.GridSpec(2, 3, width_ratios=[3, 0.1, 1], height_ratios=[1, 3])

# Time-domain wavelets (top-left)
ax_time = fig.add_subplot(gs[0, 0])
for n, m, c, ls in zip(N_VALS, M_VALS, PALETTE, LINESTYLES):
    ax_time.plot(wdm.times, wdm.gnm(n, m), c=c, ls=ls, lw=0.6)
ax_time.set_xlim(0, wdm.T)
ax_time.set_xticks([])
ax_time.set_yticks([])
ax_time.set_ylabel(r"$g_{nm}(t)$")

# Frequency-domain wavelets (bottom-right)
ax_freq = fig.add_subplot(gs[1, 2])
mask = wdm.freqs >= 0.0
for n, m, c, ls in zip(N_VALS, M_VALS, PALETTE, LINESTYLES):
    ax_freq.plot(np.abs(wdm.Gnm(n, m))[mask], wdm.freqs[mask], c=c, ls=ls, lw=0.6)
ax_freq.set_ylim(0, wdm.f_Ny)
ax_freq.set_xticks([])
ax_freq.set_yticks([])
ax_freq.xaxis.set_label_position("top")
ax_freq.set_xlabel(r"$|\tilde{g}_{nm}(f)|$")

# Time-frequency tiling (bottom-left)
ax_tf = fig.add_subplot(gs[1, 0])
for t in t_ticks:
    ax_tf.axvline(t, color=HAIRLINE, lw=0.5)
for f in f_ticks:
    ax_tf.axhline(f, color=HAIRLINE, lw=0.5)
ax_tf.set_xlim(0, wdm.T)
ax_tf.set_ylim(0, wdm.f_Ny)
ax_tf.set_xticks(np.linspace(0, wdm.T, 5))
ax_tf.set_yticks(np.linspace(0, wdm.f_Ny, 5))
ax_tf.set_xlabel(r"Time $t$ [seconds]")
ax_tf.set_ylabel(r"Frequency $f$ [Hertz]")

for n, m, c in zip(N_VALS, M_VALS, PALETTE):
    boxes = tile_boxes(n, m)
    for t0, t1, f0, f1 in boxes:
        ax_tf.add_patch(patches.Rectangle((t0, f0), t1 - t0, f1 - f0, linewidth=1,
                                          edgecolor=c, facecolor=c, alpha=0.9, zorder=3))
    if m > 0:
        t0, t1, f0, f1 = boxes[0]
        ax_tf.text(t1, f0, rf"$(n,m)=({n},{m})$", color=c, rotation=90, fontsize=8, zorder=4,
                  path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])

plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.08, right=0.92, top=0.95, bottom=0.08)
fig.savefig(HERE / "figures" / "fig6_wdm_basis.pdf")

# The same figure with the tiling axes in units of dT and dF.
n_ticks = np.arange(0, wdm.Nt + 1, wdm.Nt // 4)
m_ticks = np.arange(0, wdm.Nf + 1, wdm.Nf // 4)
ax_tf.set_xticks(n_ticks * wdm.dT, [str(n) for n in n_ticks])
ax_tf.set_yticks(m_ticks * wdm.dF, [str(m) for m in m_ticks])
ax_tf.set_xlabel(r"Time $t/\Delta T$")
ax_tf.set_ylabel(r"Frequency $f/\Delta F$")
fig.savefig(HERE / "figures" / "fig6_wdm_basis_dimensionless.pdf")
