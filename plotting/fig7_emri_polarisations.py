"""Figure 7: the EMRI's h+ and hx (parameters.py), drawn as figures 12 and 13 draw
the response: the whole record over a third-of-a-day zoom, both on the span those
figures keep (the cut record at Delta T = 1e4 s), on the signal's own clock.

Reads the waveform cache _emri_comparison.waveform writes, generating it if absent.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import _emri_comparison as emri
from _colours import HAIRLINE, INK
from parameters import DT, MAX_LAG_L, N_SIGNAL, WINDOW_KWARGS

HERE = Path(__file__).parent
NAMES = dict(plus=r"$h_+$", cross=r"$h_\times$")


def cut_span(nf=emri.NF):
    """The samples figures 12 and 13 keep at nf bands, as [first, stop) of the
    unpadded record."""
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    info = utils.wdm_padding_info(N_SIGNAL, nf, max_lag_L=MAX_LAG_L, taper_samples=taper)
    rows = utils.wdm_cut_info(info)["time_slice"]
    return rows.start * nf - info["pad_left"], rows.stop * nf - info["pad_left"]


def main():
    plt.style.use(HERE / "paper.mplstyle")
    nf = emri.NF
    first, stop = cut_span(nf)
    wave = dict(zip(NAMES, (h[first:stop] for h in emri.waveform(N_SIGNAL))))
    t = DT * np.arange(first, stop) / 86400.0
    scale = 10.0 ** np.floor(np.log10(max(np.max(np.abs(h)) for h in wave.values())))

    fig, (top, zoom) = plt.subplots(2, 1, figsize=(3.375, 3.6), layout="constrained",
                                    height_ratios=(1, 1.7))

    # Each row's extent. The louder polarisation goes underneath, here and on the zoom.
    order = sorted(wave, key=lambda k: -np.max(np.abs(wave[k])))
    t_rows = t.reshape(-1, nf)[:, nf // 2]
    for kind in order:
        rows = wave[kind].reshape(-1, nf) / scale
        top.fill_between(t_rows, rows.min(axis=1), rows.max(axis=1), lw=0,
                         color=emri.POLARISATION_COLOURS[kind])
    top.axvline(emri.ZOOM_DAY, color=INK, lw=0.6)
    top.set(xlim=(t[0], t[-1]), xlabel=r"$t$ [days]")
    top.xaxis.tick_top()
    top.xaxis.set_label_position("top")

    zoom_span = (emri.ZOOM_DAY, emri.ZOOM_DAY + emri.ZOOM_DAYS)
    window = (t >= zoom_span[0]) & (t < zoom_span[1])
    handles = {}
    for kind in order:
        handles[kind], = zoom.plot(t[window], wave[kind][window] / scale, lw=0.4,
                                   color=emri.POLARISATION_COLOURS[kind], label=NAMES[kind])
    zoom.set(xlim=zoom_span, xlabel=r"$t$ [days]",
             ylabel=rf"strain [$10^{{{int(np.log10(scale))}}}$]")
    emri.zoom_ticks(zoom)
    top.set_ylabel(zoom.get_ylabel())

    for ax in (top, zoom):
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)

    # Lines from the zoomed span on the top panel to the zoom panel's corners.
    y_low = top.get_ylim()[0]
    for x in zoom_span:
        fig.add_artist(ConnectionPatch(
            (x, y_low), (x, zoom.get_ylim()[1]), coordsA=top.transData,
            coordsB=zoom.transData, color=INK, lw=0.4))

    legend = fig.legend(handles=[handles[kind] for kind in NAMES], loc="outside lower center",
                        ncols=2, fontsize="small", frameon=False, handlelength=1.8)
    for handle in legend.legend_handles:
        handle.set_linewidth(1.5)
    path = HERE / "figures" / "fig7_emri_polarisations.pdf"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
