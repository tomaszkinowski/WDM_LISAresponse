"""Figure 10: the mismatch of the split_amp_sinc_first response
(investigation_fig8_delay_rate.py) over the (f, Delta T) plane, on link 1 -> 2;
``--kind K`` maps another variant of that script instead (e.g. production), with
its own cache and PDFs suffixed _K. The median maps of production (left panel,
fig10_mismatch_map_production.pdf) and split_amp_sinc_first (right panel,
fig10_mismatch_map.pdf) go to figures/; the others to figures/investigations/.

The tilings are figure 14's. The frequency axis is split into F_CELLS log cells;
at each tiling a cell's tones sit in the band m nearest its centre, at
f = (m + r) Delta F for N_R offsets r spaced evenly over the band, and the cell
is coloured by the median of their mismatches (and, in a second figure, the
largest). Cells where any of its tones leaks into the dropped m = 0 band are
left blank. Each tone is compared against the time-domain response on the same
tiling's grid, as for figure 14's bounds, interpolating the waveform with 12 taps
below TAPS_SPLIT and 20 above: at 12 the reference's own error reaches 2e-9 at
0.05 Hz and 3e-4 at 0.09 Hz (dt = 4 s), while 20 taps hold it near 1e-12 up to
0.055 Hz, so the cells stop at 0.05 Hz.

../prototypes/delay_rate_correction must be on PYTHONPATH. Run with no arguments
to plot from the cache; ``--part K/COUNT`` computes tilings K, K + COUNT, ...
into cache/parts/ and stops; ``--merge COUNT`` assembles them and plots. The colour
bar sits on top, and the right axis is Delta F = 1/(2 Delta T).
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import fig14_mismatch_vs_dT_zeroth_order as fig14
from _colours import HAIRLINE
from parameters import (DT, GC_PHI, GC_THETA, MONOCHROMATIC_PARAMS, N_SIGNAL,
                        NUM_INTEGRAL_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
KIND = "split_amp_sinc_first"
PAPER_KINDS = ("production", KIND)                   # the paper's two panels

F_EDGES = np.logspace(-4, -1.3, 14)                  # five cells per decade [Hz]
TAPS_SPLIT = 0.03                                    # [Hz]
F_CELLS = np.sqrt(F_EDGES[:-1] * F_EDGES[1:])
N_R = 6
R = (np.arange(N_R) + 0.5) / N_R - 0.5               # +-1/12, +-1/4, +-5/12

LOG_RANGE = (-12, -2)                                # colour scale, log10 mismatch
CONTOURS = [-10, -8, -6, -4]                         # log10 mismatch, the colour bar's ticks


def cell_frequencies(nfs, centres=F_CELLS, r=R):
    """Each cell's tones at every tiling, shape (cells, r, tilings); a cell is
    NaN throughout at tilings where any of its tones leaks into m = 0."""
    delta_f = 1 / (2 * np.asarray(nfs) * DT)
    m = np.round(np.asarray(centres)[:, None] / delta_f)
    f = (m[:, None, :] + np.asarray(r)[None, :, None]) * delta_f
    leaks = np.any(f < fig14.leak_edge() * delta_f, axis=1)
    f[np.broadcast_to(leaks[:, None, :], f.shape)] = np.nan
    return f


def suffix(kind):
    """File-name suffix: none for split_amp_sinc_first, the map's original variant."""
    return "" if kind == KIND else f"_{kind}"


def cache_path(kind=KIND):
    return HERE / "cache" / f"fig10_mismatch_map{suffix(kind)}.npz"


def compute(nfs, centres=F_CELLS, r=R, kind=KIND):
    """The mismatch of every cell's tones, shape (cells, r, tilings)."""
    import jax

    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms
    import investigation_fig8_delay_rate as inv

    orbits, start, signal_times, link = fig14.setup()
    freqs = cell_frequencies(nfs, centres, r)
    out = np.full(freqs.shape, np.nan)
    for j, Nf in enumerate(nfs):
        tick = time.perf_counter()
        info, wdm, response, tables, centre, cut = inv.tiling(orbits, start, link, Nf)
        done = {}                         # cells sharing a band share their tones
        for c, k in np.ndindex(freqs.shape[:2]):
            f = freqs[c, k, j]
            if np.isnan(f):
                continue
            if f not in done:
                params = dict(MONOCHROMATIC_PARAMS[0], f=f)
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
                _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                     info["pad_left"], info["pad_right"])
                ours = inv.responses(response, wdm, tables, centre, GC_THETA, GC_PHI,
                                     hp, hc, link, inv.tone_warp(params, start),
                                     (kind,))[kind]
                reference = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
                    response.one_arm_response_time_domain(
                        GC_THETA, GC_PHI, hp, hc, link=link,
                        num_integral_points=NUM_INTEGRAL_POINTS,
                        num_taps=12 if f < TAPS_SPLIT else 20)))), cut)
                done[f] = utils.wdm_mismatch(utils.apply_wdm_cut(ours, cut),
                                             reference)["mismatch"]
            out[c, k, j] = done[f]
        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf * DT:8.3g} s: {len(done)} tones, median "
              + " ".join(f"{x:.0e}" for x in np.nanmedian(out[:, :, j], axis=1)
                         if not np.isnan(x))
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
    return dict(Nf=np.asarray(nfs), f=freqs, mismatch=out, f_edges=F_EDGES, r=np.asarray(r))


def part_path(index, count, kind=KIND):
    return (HERE / "cache" / "parts"
            / f"fig10_mismatch_map{suffix(kind)}_part{index}of{count}.npz")


def merge(count, kind=KIND):
    parts = [dict(np.load(part_path(i, count, kind))) for i in range(count)]
    order = np.argsort(np.concatenate([p["Nf"] for p in parts]))
    merged = dict(parts[0])
    for key in ("Nf", "f", "mismatch"):
        merged[key] = np.concatenate([p[key] for p in parts], axis=-1)[..., order]
    assert np.array_equal(merged["Nf"], fig14.NF), merged["Nf"]
    np.savez(cache_path(kind), **merged)


def dt_edges(nfs):
    """Cell edges in Delta T: geometric midpoints between the tilings."""
    dt = np.asarray(nfs) * DT
    mid = np.sqrt(dt[:-1] * dt[1:])
    return np.concatenate([[dt[0] ** 2 / mid[0]], mid, [dt[-1] ** 2 / mid[-1]]])


def delta_f(delta_t):
    """Delta F [mHz] = 1 / (2 Delta T [s]), its own inverse."""
    with np.errstate(divide="ignore"):
        return 500 / np.asarray(delta_t, dtype=float)


def plot(data, statistic, path, label):
    """A smooth map of log10 `statistic` over each cell's tones; cells that leak
    into m = 0 are covered in grey."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN cells
        log_mm = np.log10(statistic(data["mismatch"], axis=1))  # (cells, tilings)
    leak = np.isnan(log_mm)
    filled = log_mm.copy()
    for j in range(filled.shape[1]):              # pad leaking cells for the shading
        first = np.flatnonzero(~leak[:, j])[0]
        filled[:first, j] = filled[first, j]

    f_edges = 1e3 * data["f_edges"]                    # [mHz]
    centres = np.sqrt(f_edges[:-1] * f_edges[1:])
    delta_t = data["Nf"] * DT
    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    mesh = ax.pcolormesh(centres, delta_t, filled.T, shading="gouraud", cmap="viridis",
                         vmin=LOG_RANGE[0], vmax=LOG_RANGE[1], rasterized=True)

    # the leaking corner, as the staircase of cell edges
    edges = dt_edges(data["Nf"])
    first_ok = np.array([np.flatnonzero(~leak[:, j])[0] for j in range(leak.shape[1])])
    f_stair = np.repeat(f_edges[first_ok], 2)
    dt_stair = np.column_stack([edges[:-1], edges[1:]]).ravel()
    ax.fill_betweenx(dt_stair, centres[0], np.maximum(f_stair, centres[0]),
                     color=HAIRLINE, lw=0)

    with np.errstate(all="ignore"):
        lines = ax.contour(centres, delta_t, log_mm.T, levels=CONTOURS, colors="white",
                           linewidths=0.6, linestyles="solid")
    ax.clabel(lines, fmt=lambda v: rf"$10^{{{int(v)}}}$", fontsize=6)

    ax.set_xlim(centres[0], centres[-1])
    ax.set_ylim(delta_t[0], delta_t[-1])
    ax.set_xlabel(r"$f$ [mHz]")
    ax.set_ylabel(r"$\Delta T$ [s]")
    right = ax.secondary_yaxis("right", functions=(delta_f, delta_f))
    right.set_ylabel(r"$\Delta F$ [mHz]")
    # outward ticks: inward ones vanish into the map
    for axis in (ax, right):
        axis.tick_params(which="both", direction="out")
    ticks = np.arange(LOG_RANGE[0], LOG_RANGE[1] + 1, 2)
    over = np.nanmax(log_mm) > LOG_RANGE[1]
    bar = fig.colorbar(mesh, ax=ax, ticks=ticks, location="top", pad=0.03, fraction=0.05,
                       aspect=30, extend="max" if over else "neither")
    bar.ax.set_xticklabels([rf"$10^{{{t}}}$" for t in ticks])
    bar.set_label(label)
    fig.savefig(path, dpi=300)
    print("wrote", path)


if __name__ == "__main__":
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else KIND
    if "--part" in sys.argv:
        index, count = map(int, sys.argv[sys.argv.index("--part") + 1].split("/"))
        path = part_path(index, count, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **compute(fig14.NF[index::count], kind=kind))
        sys.exit()
    if "--merge" in sys.argv:
        merge(int(sys.argv[sys.argv.index("--merge") + 1]), kind)
    plt.style.use(HERE / "paper.mplstyle")
    data = dict(np.load(cache_path(kind)))
    out = HERE / "figures" / "investigations"
    paper = HERE / "figures" if kind in PAPER_KINDS else out
    plot(data, np.nanmedian, paper / f"fig10_mismatch_map{suffix(kind)}.pdf", "median mismatch")
    plot(data, np.nanmax, out / f"fig10_mismatch_map{suffix(kind)}_max.pdf", "largest mismatch")
