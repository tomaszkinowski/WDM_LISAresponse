"""What figures 9 and 11 share: the EMRI on figure 6's grid, the four codes compared
on it, the printed mismatches, and the figure drawn for each quantity - the whole
record and a half-day zoom. Investigations, in figures/investigations/: the same
with the source's h+ and hx over the zoom added, and the residuals over the zoom.

The four are our wavelet response (WDM), our time-domain reference (TD),
lisagwresponse (GWR) and fastlisaresponse (FLR). The external codes return the
fractional frequency y; comparison.common.convert integrates it to our delay, a
change of units only. Their static-arm formula differs from ours by
1 - Omega.v/c (comparison.common.velocity_factor): the mismatches are printed with
and without that factor, and the series drawn without.

Every series is taken onto the wavelet grid and back with m = 0 zeroed, the band
every mismatch here drops, and kept on the rows that hold neither padding nor a
tapered sample.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from _colours import DARK, HAIRLINE, INK, MUTED, PALETTE
from parameters import (DT, EMRI_PARAMS, MAX_LAG_L, N_SIGNAL, NUM_INTERP_POINTS,
                        WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = 2500                                       # Delta T = 1e4 s
ZOOM_DAY = 200.0                                # zoom start, days after the signal starts
ZOOM_DAYS = 0.5
WAVEFORM_CACHE = HERE / "cache" / "emri_waveform.npz"

METHODS = ("WDM", "TD", "GWR", "FLR")
# The residual overlay and the polarisations. No Galactic binary appears in these
# figures, so the slots its tones hold elsewhere are free.
RESIDUAL_COLOUR = PALETTE[0]
POLARISATION_COLOURS = dict(plus=PALETTE[1], cross=PALETTE[2])
# lisagwresponse is drawn wide underneath, so the others stay visible on top of it.
STYLES = dict(WDM=dict(color=DARK, lw=1.2, zorder=3),
              TD=dict(color=PALETTE[4], lw=1.2, ls=(0, (3, 1.5)), zorder=4),
              GWR=dict(color=PALETTE[3], lw=3.0, zorder=2),
              FLR=dict(color=MUTED, lw=1.2, ls=(0, (1, 1.2)), zorder=5))


def signal_start():
    """The signal's first sample, as figure 6: 200 d into the orbit file."""
    from lisaorbits.file import read_orbit_file

    data = read_orbit_file(str(ORBIT_FILE))
    return float(data.t0) + 200 * 86400.0


def generate(n_signal):
    """The EMRI's h+ and hx, tapered and unpadded, from the signal's start."""
    from WDM_LISAresponse.code.waveforms import waveforms

    tick = time.perf_counter()
    td_gen = waveforms.emri_generator()
    _, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "emri", t0=signal_start(), dt=DT, N=n_signal,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    print(f"EMRI generated ({time.perf_counter() - tick:.0f} s)", flush=True)
    return h_plus, h_cross


def waveform(n_signal):
    """generate(n_signal), cached for the whole record."""
    if n_signal != N_SIGNAL:
        return generate(n_signal)
    if not WAVEFORM_CACHE.exists():
        h_plus, h_cross = generate(n_signal)
        WAVEFORM_CACHE.parent.mkdir(exist_ok=True)
        np.savez(WAVEFORM_CACHE, h_plus=h_plus, h_cross=h_cross)
    cached = np.load(WAVEFORM_CACHE)
    return cached["h_plus"], cached["h_cross"]


def prepare(n_signal, nf, link_convention=None, extra_rows=0):
    """The EMRI, padded onto an nf-band grid as in figure 6 (200 d into the orbit
    file), and the transform, response and cut on that grid. ``extra_rows`` widens
    the guard band at each end, for a second time shift (TDI)."""
    from lisaorbits import InterpolatedOrbits
    from lisaorbits.file import read_orbit_file

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    data = read_orbit_file(str(ORBIT_FILE))
    orbit_times = data.t0 + data.dt * np.arange(data.size)
    orbits = InterpolatedOrbits(orbit_times, data.positions, data.velocities)
    start = signal_start()
    signal_times = start + DT * np.arange(n_signal)
    h_plus, h_cross = waveform(n_signal)

    taper = waveforms.tukey_taper_samples(n_signal, WINDOW_KWARGS["alpha"])
    info = utils.wdm_padding_info(n_signal, nf, max_lag_L=MAX_LAG_L, taper_samples=taper,
                                  extra_rows_left=extra_rows, extra_rows_right=extra_rows)
    _, h_plus, h_cross = utils.apply_zero_padding(
        signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=nf, N=info["target_length"])
    t0 = start - info["pad_left"] * DT
    kwargs = {} if link_convention is None else dict(link_convention=link_convention)
    response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                            num_interp_points=NUM_INTERP_POINTS, **kwargs)

    # What comparison.conventions_* read from a ComparisonConfig, on this grid.
    grid = SimpleNamespace(
        N=info["target_length"], dt=DT, t0=t0, orbits_model="esa",
        orbit_file=str(ORBIT_FILE),
        padded_times=t0 + DT * np.arange(info["target_length"]),
        orbit_span=lambda: (float(orbit_times[0]), float(orbit_times[-1])))

    return SimpleNamespace(nf=nf, n_signal=n_signal, start=start, info=info,
                           cut=utils.wdm_cut_info(info), h_plus=h_plus, h_cross=h_cross,
                           wdm=wdm, response=response, grid=grid,
                           theta=EMRI_PARAMS["theta"], phi=EMRI_PARAMS["phi"])


def flr_orbits():
    """fastlisaresponse's own orbit provider, on the same file."""
    from lisatools.detector import Orbits

    orbits = Orbits(str(ORBIT_FILE))
    orbits.configure(linear_interp_setup=True)
    return orbits


def finish(setup, grids, external):
    """Mismatches and time series from our two wavelet grids and the external codes'
    series (fractional frequency on the padded grid), as the cached result."""
    from WDM_LISAresponse.code.comparison import common
    from WDM_LISAresponse.code.utils import utils

    s = setup
    factor = common.velocity_factor(s.response.orbits_class, s.grid.padded_times,
                                    s.theta, s.phi)
    for name, series in external.items():
        grids[name] = common.convert(series[None], s.grid, s.wdm)[0]
        grids[name + "_v"] = common.convert(series[None], s.grid, s.wdm, factor)[0]

    rows = s.cut["time_slice"]
    out = dict(Nf=s.nf, t_start=s.start, pad_left=s.info["pad_left"],
               n_signal=s.n_signal, rows=np.array([rows.start, rows.stop]))
    for reference in ("TD", "WDM"):
        for name in [m for m in METHODS if m != reference] + ["GWR_v", "FLR_v"]:
            result = utils.wdm_mismatch(utils.apply_wdm_cut(grids[name], s.cut),
                                        utils.apply_wdm_cut(grids[reference], s.cut))
            out[f"mismatch_{name}_{reference}"] = result["mismatch"]
            out[f"rel_L2_{name}_{reference}"] = result["rel_L2"]

    # Back to time with m = 0 zeroed, kept on the cut rows only.
    for name in METHODS:
        g = np.array(grids[name])
        g[:, 0] = 0.0
        out[name] = np.asarray(s.wdm.idwt(g))[rows.start * s.nf:rows.stop * s.nf]
    out["t_first"] = (rows.start * s.nf - s.info["pad_left"]) * DT
    return out


def summary(res, labels, quantity):
    """Every pair's mismatch and relative L2 over the cut record, the external codes'
    with and without the velocity factor 1 - Omega.v/c."""
    nf = int(res["Nf"])
    width = max(len(label) for label in labels.values()) + 2
    lines = [f"Nf = {nf}, Delta T = {nf * DT:g} s, {quantity}, whole cut record",
             f"{'':<{width + 2}}{'mismatch':>10}{'rel L2':>10}"
             f"{'mismatch, v-corrected':>32}{'rel L2, v-corrected':>30}"]
    for reference in ("TD", "WDM"):
        lines.append(f"against {labels[reference]}")
        for name in METHODS:
            if name == reference:
                continue
            m, l2 = (float(res[f"{kind}_{name}_{reference}"]) for kind in ("mismatch", "rel_L2"))
            row = f"  {labels[name]:<{width}}{m:>10.2e}{l2:>10.2e}"
            if name in ("GWR", "FLR"):
                m_v, l2_v = (float(res[f"{kind}_{name}_v_{reference}"])
                             for kind in ("mismatch", "rel_L2"))
                row += f"{m_v:>14.2e} (x{m_v / m:5.3f})   {l2_v:>12.2e} (x{l2_v / l2:5.3f})"
            lines.append(row)
    return "\n".join(lines)


def plot_signals(res, path, labels, symbol, wave=None, residual=None, zoom_day=ZOOM_DAY):
    """The whole record, one translucent shade per code, over a zoom of ZOOM_DAYS;
    with ``wave``, a third panel of the source's h+ and hx over the same zoom.
    ``symbol`` is the quantity's LaTeX symbol. ``residual``, a pair of codes,
    overlays their difference on the zoom, magnified by the power of ten that brings
    it to the signal's size."""
    plt.style.use(HERE / "paper.mplstyle")
    nf = int(res["Nf"])
    t = (res["t_first"] + DT * np.arange(len(res["WDM"]))) / 86400.0
    scale = 10.0 ** np.floor(np.log10(np.max(np.abs(res["TD"]))))

    if wave is None:
        fig, (top, zoom) = plt.subplots(2, 1, figsize=(3.375, 3.6), layout="constrained",
                                        height_ratios=(1, 1.7))
    else:
        fig, (top, zoom, source) = plt.subplots(3, 1, figsize=(3.375, 4.8),
                                                layout="constrained",
                                                height_ratios=(1, 1.7, 1.3))

    # Each row's extent. The axis sits on top, so the lines down to the zoom cross
    # no labels.
    t_rows = t.reshape(-1, nf)[:, nf // 2]
    for name in METHODS:
        rows = res[name].reshape(-1, nf) / scale
        top.fill_between(t_rows, rows.min(axis=1), rows.max(axis=1), lw=0, alpha=0.35,
                         color=STYLES[name]["color"], zorder=STYLES[name]["zorder"])
    top.axvline(zoom_day, color=INK, lw=0.6)
    top.set(xlim=(t[0], t[-1]), xlabel=r"$t$ [days]")
    top.xaxis.tick_top()
    top.xaxis.set_label_position("top")

    zoom_span = (zoom_day, zoom_day + ZOOM_DAYS)
    window = (t >= zoom_span[0]) & (t < zoom_span[1])
    # Half a day holds ~100 cycles, so thin solid lines: dashes and a wide underlay would
    # only fill the band in.
    handles = []
    for name in METHODS:
        handles += zoom.plot(t[window], res[name][window] / scale, label=labels[name],
                             color=STYLES[name]["color"], lw=0.4,
                             zorder=STYLES[name]["zorder"])
    if residual is not None:
        a, b = residual
        difference = (res[a] - res[b])[window]
        gain = 10.0 ** np.floor(np.log10(np.max(np.abs(res[a][window]))
                                         / np.max(np.abs(difference))))
        handles += zoom.plot(
            t[window], gain * difference / scale, color=RESIDUAL_COLOUR, lw=0.4, zorder=6,
            label=rf"({labels[a]} $-$ {labels[b]}) $\times 10^{{{int(np.log10(gain))}}}$")
    zoom.set(xlim=zoom_span, xlabel=r"$t$ [days]",
             ylabel=rf"${symbol}$ [$10^{{{int(np.log10(scale))}}}$ s]")
    top.set_ylabel(zoom.get_ylabel())
    axes = [top, zoom]

    if wave is not None:
        # The source over the same span, on the signal's own clock (the response sees
        # it up to ~500 s earlier or later, the light travel time across the orbit).
        t_wave = DT * np.arange(len(wave[0])) / 86400.0
        in_zoom = (t_wave >= zoom_span[0]) & (t_wave < zoom_span[1])
        strain = 10.0 ** np.floor(np.log10(max(np.max(np.abs(h[in_zoom])) for h in wave)))
        for h, kind, tex in zip(wave, ("plus", "cross"), (r"$h_+$", r"$h_\times$")):
            handles += source.plot(t_wave[in_zoom], h[in_zoom] / strain, lw=0.4,
                                   color=POLARISATION_COLOURS[kind], label=tex)
        source.set(xlim=zoom_span, xlabel=r"$t$ [days]",
                   ylabel=rf"strain [$10^{{{int(np.log10(strain))}}}$]")
        source.sharex(zoom)
        zoom.set_xlabel("")
        zoom.tick_params(labelbottom=False)
        axes.append(source)

    for ax in axes:
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)

    # Lines from the zoomed span on the top panel to the zoom panel's corners.
    y_low = top.get_ylim()[0]
    for x in zoom_span:
        fig.add_artist(ConnectionPatch(
            (x, y_low), (x, zoom.get_ylim()[1]), coordsA=top.transData,
            coordsB=zoom.transData, color=INK, lw=0.4))

    legend = fig.legend(handles=handles, loc="outside lower center", ncols=2,
                        fontsize="small", frameon=False, handlelength=1.8,
                        columnspacing=1.0)
    for handle in legend.legend_handles:
        handle.set_linewidth(1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


# What each residual panel compares, and against what: the time-domain reference
# split in two, as the external codes' residual is far below the wavelet response's.
PANELS = (("TD", ("WDM",)), ("TD", ("GWR", "FLR")), ("WDM", ("TD", "GWR", "FLR")))


def plot_residuals(res, path, labels, zoom_day=ZOOM_DAY):
    """The residuals over the zoom, in seconds, each panel on its own scale: the
    wavelet response and then the external codes against the time-domain reference,
    and the other three against the wavelet response."""
    plt.style.use(HERE / "paper.mplstyle")
    t = (res["t_first"] + DT * np.arange(len(res["WDM"]))) / 86400.0
    zoom_span = (zoom_day, zoom_day + ZOOM_DAYS)
    window = (t >= zoom_span[0]) & (t < zoom_span[1])

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.375, 5.2), sharex=True,
                             layout="constrained")
    for ax, (reference, names) in zip(axes, PANELS):
        residuals = {name: (res[name] - res[reference])[window] for name in names}
        largest = max(np.max(np.abs(r)) for r in residuals.values())
        scale = 10.0 ** np.floor(np.log10(largest))
        for name, residual in residuals.items():
            ax.plot(t[window], residual / scale, **STYLES[name])
        ax.set_title(f"against the {labels[reference]}", loc="left", fontsize="small",
                     color=INK)
        ax.set_ylabel(rf"residual [$10^{{{int(np.log10(scale))}}}$ s]")
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)

    axes[-1].set(xlim=zoom_span, xlabel=r"$t$ [days]")
    handles = [plt.Line2D([], [], **STYLES[name]) for name in METHODS]
    fig.legend(handles, [labels[name] for name in METHODS], loc="outside lower center",
               ncols=2, fontsize="small", frameon=False, handlelength=2.2,
               columnspacing=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


def main(compute, stem, names, labels, quantity, symbol, residual=None):
    """The command line both figures share. ``stem`` (e.g. fig9_emri_link) names the
    cache and the mismatch file; ``names`` the figure's file and the residuals' file.
    The figure goes to figures/, and to figures/investigations/ again with the
    waveform panel (``_waveform``) beside the residuals. ``residual`` is passed to
    plot_signals.

    No arguments: compute (or reuse the cache) at Nf = NF and plot. ``--short``: 2**19
    samples (24 d), printed and drawn into figures/investigations/ without caching.
    ``--nf NF``: the whole record at another tiling, cached and drawn as an
    investigation."""
    investigations = HERE / "figures" / "investigations"
    if "--short" in sys.argv:
        res = compute(2 ** 19, NF)
        print(summary(res, labels, quantity))
        wave = waveform(2 ** 19)
        plot_signals(res, investigations / f"{names[0]}_waveform_short.pdf", labels,
                     symbol, wave, residual, zoom_day=10.0)
        plot_residuals(res, investigations / f"{names[1]}_short.pdf", labels,
                       zoom_day=10.0)
        return

    nf, cache = NF, HERE / "cache" / f"{stem}_comparison.npz"
    figures, suffix = HERE / "figures", ""
    if "--nf" in sys.argv:
        nf = int(sys.argv[sys.argv.index("--nf") + 1])
        cache = HERE / "cache" / f"investigation_{stem.split('_')[0]}_nf{nf}.npz"
        figures, suffix = investigations, f"_nf{nf}"

    if cache.exists():
        res = dict(np.load(cache))
    else:
        res = compute(N_SIGNAL, nf)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, **res)
    text = summary(res, labels, quantity)
    print(text)
    figures.mkdir(parents=True, exist_ok=True)
    (figures / f"{stem}_mismatches{suffix}.txt").write_text(text + "\n")
    plot_signals(res, figures / f"{names[0]}{suffix}.pdf", labels, symbol,
                 residual=residual)
    plot_signals(res, investigations / f"{names[0]}_waveform{suffix}.pdf", labels, symbol,
                 waveform(N_SIGNAL), residual)
    plot_residuals(res, investigations / f"{names[1]}{suffix}.pdf", labels)
