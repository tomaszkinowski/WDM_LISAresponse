"""Figure 6: mismatch of the wavelet one-arm response against the time-domain
reference, as a function of the tiling Delta T = Nf dt, on link 1 -> 2: one panel
each for the three example Galactic binaries and the EMRI (parameters.py).

Delta T runs from 1e2 to 1e6 s, four tilings per decade. Each grid is compared
on the rows holding neither padding nor a tapered sample, without the m = 0
band, which WDM_transform gets wrong without calc_m0. A signal leaks into that
dropped band below (1 - A_frac) Delta F, the edge of the Meyer window, not just
below the band edge Delta F / 2. Hollow markers are tones that do, and the EMRI
where more than 1% of its WDM power lies in m = 0 (from investigation_emri_m0.py's
dense sweep of Delta T, which must be run first).

Each tone's mismatch depends on where it falls within its band (figure 7), so at
every tiling the same tone is also moved to its nearest point halfway between two
centres, (k + 1/2) Delta F (off-bin, r = 1/2, the peak of every curve in figure
6), and to its nearest band centre m Delta F and OFFSETS around it. The shading
runs from the least of those near-centre points and the source itself up to the
off-bin point; the points themselves are not drawn. The least is taken, not the
band centre alone, because near the floor the error is set by the lag truncation
at L = 25, which varies with the offset from the centre. Above Delta T ~ 1e5 s the
30 mHz on-bin point can lie above the off-bin one, so neither is a bound there.
Nothing is drawn where the tone itself leaks into m = 0, and the off-bin point is
kept to k >= 1 so it does not leak either. These come from a separate run, on
each tiling's grid directly, cached in fig6_bounds.npz.

The time-domain reference is computed once per waveform, on a grid with a few
hours of zeros at each end, and utils.apply_repadding moves it onto each
tiling: the response reaches at most ~500 s past the signal, so nothing is lost.

Ported from examples/accuracy_analysis.ipynb, at the paper's parameters.
Run with no arguments to compute (or reuse the caches) and plot. ``--only NF``
runs one tiling, prints it and caches nothing - for checking time and memory;
``--only-bounds NF`` does the same for the on- and off-bin points.
``--part K/COUNT`` computes tilings K, K + COUNT, ... into cache/parts/ and stops,
so COUNT processes can split the work; ``--merge COUNT`` assembles them. With
``--drop-edge-bands`` everything compares bands 2 to Nf - 2 only (m = 0, 1 and
Nf - 1 dropped), with caches of its own, and the plot goes to
figures/investigations/fig6_mismatch_vs_dT_edge_bands.pdf.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from _colours import DARK, HAIRLINE, INK, PALETTE
from parameters import (DT, EMRI_PARAMS, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "fig6_mismatch_vs_dT.npz"
BOUNDS_CACHE = HERE / "cache" / "fig6_bounds.npz"

# --drop-edge-bands: the investigation variant, comparing bands 2 to Nf - 2 only.
# The shift cannot get m = 1 right once a signal reaches m = 0 (it has no regular
# lower neighbour there), and wraps m = 0 into m = Nf - 1.
EDGE_FREQ_SLICE = slice(2, -1)
EDGE_CACHE = HERE / "cache" / "fig6_mismatch_vs_dT_edge_bands.npz"
EDGE_BOUNDS_CACHE = HERE / "cache" / "fig6_bounds_edge_bands.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

# 1e2 to 1e6 s, four tilings per decade; Nf must be even.
NF = np.unique(2 * np.round(np.logspace(2, 6, 17) / (2 * DT)).astype(int))

# The reference grid: 16 rows of 64 samples (68 min) of zeros at each end, which
# holds everything the response carries past the signal.
REF_NF = 64
REF_ROWS = 16

# A tone below (1 - A_frac) Delta F leaks into the dropped m = 0 band: the band
# edge is Delta F / 2, but the Meyer window reaches (1/2 - A_frac) Delta F past it. The
# taper study measured 1e-1 of the power there at 0.55 Delta F, 1e-5 at 0.70 and
# none above 0.75. WDM_transform's default A_frac.
A_FRAC = 0.25

DT_BAND = (3600.0, 36000.0)                     # 3600-36000 s, recommended in fig 5

# The EMRI is drawn hollow where more than this fraction of its WDM power (h_plus
# and h_cross) lies in m = 0, m0 in investigation_emri_m0.py's cache.
EMRI_SWEEP_CACHE = HERE / "cache" / "investigation_emri_m0.npz"
EMRI_M0_LIMIT = 0.01
# With the edge bands dropped, the EMRI is hollow where less than this fraction of
# its WDM power lies in the compared bands (emri_shade_edge).
EMRI_COMPARED_LIMIT = 0.5

TONES = tuple(f"{f * 1e3:g} mHz" for f in FREQUENCIES)
NAMES = TONES + ("EMRI",)
COLOURS = dict(zip(NAMES, PALETTE[:3] + [DARK]))


def lowest_band(freq_slice=None):
    """The lowest band compared: 1 by default (m = 0 dropped), or freq_slice's."""
    return 1 if freq_slice is None else freq_slice.start


def leak_edge(freq_slice=None):
    """Below this frequency, in units of Delta F, a signal leaks into a dropped low
    band: the lower edge of the lowest compared band's neighbour's Meyer window,
    (lowest - A_frac) Delta F - 0.75 Delta F by default, 1.75 with m = 1 dropped."""
    return lowest_band(freq_slice) - A_FRAC


def emri_shade_edge(freq_slice=None):
    """The Delta T below which the EMRI is drawn hollow. By default, where more than
    1% of its WDM power lies in m = 0. With edge bands dropped, the leak no longer
    shows in the mismatch (the dropped bands' error is not compared), so the hollow
    points mark coverage instead: where less than half its WDM power lies in the
    compared bands. Both are interpolated in log Delta T between the dense sweep's
    tilings, at the first one past the limit."""
    sweep = dict(np.load(EMRI_SWEEP_CACHE))
    if freq_slice is None:
        fraction, limit = -sweep["m0"], -EMRI_M0_LIMIT
    else:
        fraction, limit = sweep["compared"], EMRI_COMPARED_LIMIT
    delta_t = np.log(sweep["Nf"] * DT)
    j = np.flatnonzero(fraction >= limit)[0]
    return float(np.exp(np.interp(limit, fraction[[j - 1, j]], delta_t[[j - 1, j]])))


def bound_frequencies(nfs, freq_slice=None):
    """Each tone's nearest on-bin (r = 0) and off-bin (r = 1/2) frequencies at every
    tiling, NaN where the tone leaks into a dropped band. Arrays of shape
    (tones, tilings)."""
    delta_f = 1 / (2 * np.asarray(nfs) * DT)
    on, off = np.full((2, len(TONES), len(delta_f)), np.nan)
    for i, f in enumerate(FREQUENCIES):
        ok = f >= leak_edge(freq_slice) * delta_f
        on[i, ok] = np.round(f / delta_f[ok]) * delta_f[ok]
        off[i, ok] = (np.maximum(np.floor(f / delta_f[ok]), lowest_band(freq_slice))
                      + 0.5) * delta_f[ok]
    return on, off


# Offsets from the nearest band centre, in units of Delta F, at which each tone is
# also evaluated for the lower edge of its shading.
OFFSETS = (-0.25, -0.1, -0.05, 0.05, 0.1, 0.25)


def offset_frequencies(nfs, freq_slice=None):
    """Each tone at OFFSETS around its nearest band centre, NaN where that leaks
    into a dropped band. Array of shape (tones, offsets, tilings)."""
    on, _ = bound_frequencies(nfs, freq_slice)
    delta_f = 1 / (2 * np.asarray(nfs) * DT)
    shifted = on[:, None, :] + np.asarray(OFFSETS)[None, :, None] * delta_f
    shifted[shifted < leak_edge(freq_slice) * delta_f] = np.nan
    return shifted


def setup():
    """The orbits, the signal's start time and sample times, and the link."""
    from lisaorbits import InterpolatedOrbits
    from lisaorbits.file import read_orbit_file
    from WDM_LISAresponse.code.utils import utils

    data = read_orbit_file(str(ORBIT_FILE))
    orbit_times = data.t0 + data.dt * np.arange(data.size)
    orbits = InterpolatedOrbits(orbit_times, data.positions, data.velocities)
    # 200 d in: the widest tiling pads about 160 d ahead of the signal.
    start = orbit_times[0] + 200 * 86400.0
    return orbits, start, start + DT * np.arange(N_SIGNAL), utils.Link(1, 2)


def compute(nfs, freq_slice=None):
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = setup()

    # The tones sit at the Galactic centre, the EMRI towards M87.
    sky = {name: (GC_THETA, GC_PHI) for name in TONES}
    sky["EMRI"] = (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])

    signals = {}
    for name, params in zip(TONES, MONOCHROMATIC_PARAMS):
        _, h_plus, h_cross = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        signals[name] = (h_plus, h_cross)
    tick = time.perf_counter()
    td_gen = waveforms.emri_generator()
    _, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    signals["EMRI"] = (h_plus, h_cross)
    del td_gen
    print(f"EMRI generated ({time.perf_counter() - tick:.0f} s)", flush=True)

    def padded(info, h_plus, h_cross):
        _, h_plus, h_cross = utils.apply_zero_padding(
            signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])
        return h_plus, h_cross

    ref_info = utils.wdm_padding_info(N_SIGNAL, REF_NF, extra_rows_left=REF_ROWS,
                                      extra_rows_right=REF_ROWS)
    ref_response = LISAresponse(
        WDM.WDM.WDM_transform(dt=DT, Nf=REF_NF, N=ref_info["target_length"]), orbits,
        t0=start - ref_info["pad_left"] * DT, max_lag_L=MAX_LAG_L,
        num_interp_points=NUM_INTERP_POINTS, link_convention=(link,))
    references = {}
    for name, (h_plus, h_cross) in signals.items():
        tick = time.perf_counter()
        references[name] = np.asarray(ref_response.one_arm_response_time_domain(
            *sky[name], *padded(ref_info, h_plus, h_cross), link=link,
            num_integral_points=NUM_INTEGRAL_POINTS))
        print(f"reference {name:>8}: {time.perf_counter() - tick:.0f} s", flush=True)
    del ref_response
    jax.clear_caches()

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    out = dict(Nf=np.asarray(nfs), kept_rows=np.zeros(len(nfs), dtype=int),
               **{name: np.full(len(nfs), np.nan) for name in NAMES})
    for j, Nf in enumerate(nfs):
        tick = time.perf_counter()
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG_L,
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        cut = utils.wdm_cut_info(info)
        if freq_slice is not None:
            cut["freq_slice"] = freq_slice
        out["kept_rows"][j] = cut["time_slice"].stop - cut["time_slice"].start

        for name, (h_plus, h_cross) in signals.items():
            h_plus, h_cross = padded(info, h_plus, h_cross)
            ours = response.one_arm_response(*sky[name], wdm.dwt(h_plus), wdm.dwt(h_cross),
                                             link=link)
            reference = utils.apply_repadding(
                references[name], ref_info["pad_left"], ref_info["pad_right"],
                info["pad_left"], info["pad_right"])
            out[name][j] = utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut),
                utils.apply_wdm_cut(np.asarray(wdm.dwt(reference)), cut))["mismatch"]

        # Every tiling compiles for its own array shapes and none are reused, so let
        # the compiled code go rather than accumulate it.
        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf * DT:8.3g} s, {out['kept_rows'][j]:>6} rows:  "
              + "  ".join(f"{name} {out[name][j]:.1e}" for name in NAMES)
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)

    return out


def compute_bounds(nfs, freq_slice=None):
    """Mismatch of each tone moved to its on- and off-bin frequency, and to OFFSETS
    around its band centre, at every tiling. With a different frequency at every
    tiling, each reference is computed on that tiling's grid directly, as in
    figure 7."""
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = setup()
    freqs = dict(zip(("on", "off"), bound_frequencies(nfs, freq_slice)))
    freqs["offsets"] = offset_frequencies(nfs, freq_slice)
    out = dict(Nf=np.asarray(nfs), f_on=freqs["on"], f_off=freqs["off"],
               f_offsets=freqs["offsets"],
               on=np.full(freqs["on"].shape, np.nan),
               off=np.full(freqs["on"].shape, np.nan),
               offsets=np.full(freqs["offsets"].shape, np.nan))
    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    for j, Nf in enumerate(nfs):
        if np.all(np.isnan(freqs["on"][:, j])):
            continue
        tick = time.perf_counter()
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG_L,
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        cut = utils.wdm_cut_info(info)
        if freq_slice is not None:
            cut["freq_slice"] = freq_slice

        def mismatch(params, f):
            _, h_plus, h_cross = waveforms.generate_windowed_waveform(
                "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
                model_kwargs=dict(params, f=f), window_function=WINDOW_FUNCTION,
                window_kwargs=WINDOW_KWARGS)
            _, h_plus, h_cross = utils.apply_zero_padding(
                signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])
            ours = response.one_arm_response(GC_THETA, GC_PHI, wdm.dwt(h_plus),
                                             wdm.dwt(h_cross), link=link)
            reference = response.one_arm_response_time_domain(
                GC_THETA, GC_PHI, h_plus, h_cross, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)
            return utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut),
                utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(reference))), cut)
            )["mismatch"]

        for i, params in enumerate(MONOCHROMATIC_PARAMS):
            for kind in ("on", "off"):
                if not np.isnan(freqs[kind][i, j]):
                    out[kind][i, j] = mismatch(params, freqs[kind][i, j])
            for k, f in enumerate(freqs["offsets"][i, :, j]):
                if not np.isnan(f):
                    out["offsets"][i, k, j] = mismatch(params, f)

        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf * DT:8.3g} s:  "
              + "  ".join(f"{name} on {out['on'][i, j]:.1e} "
                          f"min {np.nanmin(np.append(out['offsets'][i, :, j], out['on'][i, j])):.1e} "
                          f"off {out['off'][i, j]:.1e}"
                          for i, name in enumerate(TONES) if not np.isnan(out["on"][i, j]))
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)

    return out


def plot(res, bounds, path, freq_slice=None):
    plt.style.use(HERE / "paper.mplstyle")
    delta_t = res["Nf"] * DT
    assert np.array_equal(bounds["Nf"], res["Nf"])
    values = np.concatenate([res[name] for name in NAMES]
                            + [bounds["on"].ravel(), bounds["off"].ravel(),
                               bounds["offsets"].ravel()])
    # One mismatch scale for all four panels, so they read against each other.
    bottom, top = np.nanmin(values) / 3, min(3.0, 3 * np.nanmax(values))

    fig, axes = plt.subplots(len(NAMES), 1, figsize=(3.375, 7.4), sharex=True,
                             sharey=True, layout="constrained")
    markers = dict(ms=2.5, mec="white", mew=0.4)
    hollow = dict(ls="none", mfc="white", mew=0.7, zorder=5)
    xlim = (delta_t[0] / 1.3, delta_t[-1] * 1.3)
    for ax, name in zip(axes, NAMES):
        ax.axvspan(*DT_BAND, color=PALETTE[3], alpha=0.3, lw=0, zorder=0.5)
        colour = COLOURS[name]
        if name == "EMRI":
            ax.plot(delta_t, res[name], color=colour, lw=1.2, marker="o", zorder=4,
                    **markers)
            leaks = delta_t < emri_shade_edge(freq_slice)
            ax.plot(delta_t[leaks], res[name][leaks], marker="o", ms=2.5, mec=colour,
                    **hollow)
        else:
            i = TONES.index(name)
            on, off = bounds["on"][i], bounds["off"][i]
            # the least of on-bin, the offsets around it and the source itself
            lower = np.nanmin(np.vstack([on, bounds["offsets"][i], res[name]]), axis=0)
            lower = np.where(np.isnan(on), np.nan, lower)
            ax.fill_between(delta_t, lower, off, where=~np.isnan(on), color=colour,
                            alpha=0.18, lw=0, zorder=1)
            ax.plot(delta_t, res[name], color=colour, lw=0.5, marker="o", zorder=3,
                    **markers)
            leaks = FREQUENCIES[i] < leak_edge(freq_slice) / (2 * delta_t)
            ax.plot(delta_t[leaks], res[name][leaks], marker="o", ms=2.5, mec=colour,
                    **hollow)
        # Bottom right is empty in every panel: fine tilings are where the error is.
        ax.text(0.97, 0.06, name, transform=ax.transAxes, color=colour, ha="right",
                va="bottom", fontsize="small", zorder=6)
        ax.set(xscale="log", yscale="log", ylim=(bottom, top), ylabel="mismatch")
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)

    ax = axes[-1]
    ax.set(xlim=xlim,
           xlabel=r"$\Delta T = N_f\,\delta t$ [s]")
    decades = np.arange(np.ceil(np.log10(bottom)), 1)
    ax.yaxis.set_major_locator(FixedLocator(10 ** decades[decades % 2 == 0]))
    ax.yaxis.set_minor_locator(FixedLocator(10 ** decades))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Delta F on top: the same axis read the other way, Delta F Delta T = 1/2.
    def reciprocal_mhz(x):
        """Delta F in mHz from Delta T in s, and back: the map is its own inverse."""
        with np.errstate(divide="ignore"):
            return 5e2 / np.asarray(x, dtype=float)

    top_axis = axes[0].secondary_xaxis("top", functions=(reciprocal_mhz, reciprocal_mhz))
    top_axis.set_xlabel(r"$\Delta F$ [mHz]")

    # One legend for the four panels, in grey: the colours are the panels'. The
    # on- to off-bin shading and the 3600-36000 s band are left to the caption.
    handles = [Line2D([], [], color=INK, lw=0.5, marker="o", **markers),
               Line2D([], [], marker="o", ms=2.5, mec=INK, **hollow)]
    dropped = "m = 0" if lowest_band(freq_slice) == 1 else rf"m \le {lowest_band(freq_slice) - 1}"
    labels = ["source", rf"power leaks into ${dropped}$"]
    fig.legend(handles, labels, loc="outside lower center", fontsize="small", ncols=2,
               handlelength=1.4, handletextpad=0.5, columnspacing=0.9, frameon=False)

    (HERE / "figures").mkdir(exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    edge = "--drop-edge-bands" in sys.argv
    freq_slice = EDGE_FREQ_SLICE if edge else None
    if "--only" in sys.argv:
        compute([int(sys.argv[sys.argv.index("--only") + 1])], freq_slice)
        sys.exit()
    if "--only-bounds" in sys.argv:
        compute_bounds([int(sys.argv[sys.argv.index("--only-bounds") + 1])], freq_slice)
        sys.exit()
    targets = ((EDGE_CACHE if edge else CACHE, compute),
               (EDGE_BOUNDS_CACHE if edge else BOUNDS_CACHE, compute_bounds))

    def part_path(path, index, count):
        return path.parent / "parts" / f"{path.stem}_part{index}of{count}.npz"

    if "--part" in sys.argv:
        index, count = map(int, sys.argv[sys.argv.index("--part") + 1].split("/"))
        for path, run in targets:
            part_path(path, index, count).parent.mkdir(parents=True, exist_ok=True)
            np.savez(part_path(path, index, count), **run(NF[index::count], freq_slice))
        sys.exit()
    if "--merge" in sys.argv:
        count = int(sys.argv[sys.argv.index("--merge") + 1])
        for path, _ in targets:
            parts = [dict(np.load(part_path(path, i, count))) for i in range(count)]
            order = np.argsort(np.concatenate([p["Nf"] for p in parts]))
            merged = {key: np.concatenate([p[key] for p in parts], axis=-1)[..., order]
                      for key in parts[0]}
            assert np.array_equal(merged["Nf"], NF), merged["Nf"]
            np.savez(path, **merged)

    caches = []
    for path, run in targets:
        if path.exists():
            caches.append(dict(np.load(path)))
        else:
            caches.append(run(NF, freq_slice))
            path.parent.mkdir(exist_ok=True)
            np.savez(path, **caches[-1])
    plot(*caches, HERE / "figures" / ("investigations/fig6_mismatch_vs_dT_edge_bands.pdf"
                                      if edge else "fig6_mismatch_vs_dT.pdf"), freq_slice)
