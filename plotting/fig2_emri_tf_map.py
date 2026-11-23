"""Figure 2: the EMRI's h+ (parameters.py) on the WDM grid at
Delta T = 1e4 s (Nf = 2500), as |w_nm| on a log scale spanning DYNAMIC_RANGE below
its largest coefficient. The grid is figure 8's: padded and windowed as figure 14, cut to the rows
holding neither padding nor a tapered sample, m = 0 dropped. Only the bands the
signal reaches are drawn.

Reads the waveform cache _emri_comparison.waveform writes, generating it if absent.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import _emri_comparison as emri
from parameters import DT, MAX_LAG_L, N_SIGNAL, WINDOW_KWARGS

HERE = Path(__file__).parent
DYNAMIC_RANGE = 1e-3    # lowest |w| drawn, relative to the largest; also which bands count as reached
F_MARGIN = 0.05         # fraction of the reached band range added above and below

CMAP = "magma"


def wdm_plus(h_plus, h_cross, nf=emri.NF):
    """|w_nm| of h+ on the cut nf-band grid, the band centres' times in days from
    the signal's start, and the band centres' frequencies in Hz."""
    import WDM
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    info = utils.wdm_padding_info(N_SIGNAL, nf, max_lag_L=MAX_LAG_L, taper_samples=taper)
    _, h_plus, _ = utils.apply_zero_padding(
        DT * np.arange(N_SIGNAL), h_plus, h_cross, info["pad_left"], info["pad_right"])
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=nf, N=info["target_length"])
    grid = np.abs(np.asarray(wdm(h_plus)))
    cut = utils.wdm_cut_info(info)
    rows = np.arange(wdm.Nt)[cut["time_slice"]]
    bands = np.arange(nf)[cut["freq_slice"]]
    days = (rows * wdm.dT - info["pad_left"] * DT) / 86400.0
    return grid[cut["time_slice"], cut["freq_slice"]], days, bands * wdm.dF, wdm


def draw(h_plus, h_cross, path):
    """The map for a whole-record h+ and hx (unpadded, from the signal's start)."""
    plt.style.use(HERE / "paper.mplstyle")
    grid, days, freqs, wdm = wdm_plus(h_plus, h_cross)
    vmax = grid.max()
    vmin = DYNAMIC_RANGE * vmax

    reached = np.flatnonzero(grid.max(axis=0) >= vmin)
    lo, hi = freqs[reached[0]], freqs[reached[-1]]
    pad = F_MARGIN * (hi - lo)
    half_t, half_f = 0.5 * wdm.dT / 86400.0, 0.5 * wdm.dF

    fig, ax = plt.subplots(figsize=(3.375, 2.5), layout="constrained")
    image = ax.imshow(grid.T, origin="lower", aspect="auto", interpolation="antialiased",
                      cmap=CMAP, norm=LogNorm(vmin=vmin, vmax=vmax),
                      extent=((days[0] - half_t), (days[-1] + half_t),
                              (freqs[0] - half_f) * 1e3, (freqs[-1] + half_f) * 1e3))
    ax.set(xlim=(days[0] - half_t, days[-1] + half_t),
           ylim=(max(lo - pad, freqs[0] - half_f) * 1e3, (hi + pad) * 1e3),
           xlabel=r"$t$ [days]", ylabel=r"$f$ [mHz]")
    bar = fig.colorbar(image, ax=ax, location="right", aspect=30, pad=0.02, extend="min")
    bar.set_label(r"$|w^+_{nm}|$")
    bar.ax.tick_params(which="both", direction="out")

    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"wrote {path}  (bands {lo * 1e3:.2f}-{hi * 1e3:.2f} mHz, "
          f"{grid.shape[0]} rows x {grid.shape[1]} bands)")


def main():
    draw(*emri.waveform(N_SIGNAL), HERE / "figures" / "fig2_emri_tf_map.pdf")


if __name__ == "__main__":
    main()
