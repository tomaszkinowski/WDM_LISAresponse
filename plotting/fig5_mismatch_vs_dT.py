"""Figure 5: mismatch of the wavelet one-arm response against the time-domain
reference, as a function of the tiling Delta T = Nf dt, for the three example
Galactic binaries and the EMRI (parameters.py), on link 1 -> 2.

Delta T runs from 1e2 to 1e6 s, four tilings per decade. Each grid is compared
on the rows holding neither padding nor a tapered sample, without the m = 0
band, which WDM_transform gets wrong without calc_m0. Hollow markers are tones
that leak into that dropped band: below (1/2 + A_frac) Delta F, the edge of the
Meyer window, not just below the band edge Delta F / 2. The shading is where
the two responses are distinguishable at SNR 100.

The time-domain reference is computed once per waveform, on a grid with a few
hours of zeros at each end, and utils.apply_repadding moves it onto each
tiling: the response reaches at most ~500 s past the signal, so nothing is lost.

Ported from examples/accuracy_analysis.ipynb, at the paper's parameters.
Run with no arguments to compute (or reuse the cache) and plot. ``--only NF``
runs one tiling, prints it and caches nothing - for checking time and memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter

from _colours import DARK, HAIRLINE, INK, MUTED, PALETTE, SHADE
from parameters import (DT, EMRI_PARAMS, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "fig5_mismatch_vs_dT.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

# 1e2 to 1e6 s, four tilings per decade; Nf must be even.
NF = np.unique(2 * np.round(np.logspace(2, 6, 17) / (2 * DT)).astype(int))

# The reference grid: 16 rows of 64 samples (68 min) of zeros at each end, which
# holds everything the response carries past the signal.
REF_NF = 64
REF_ROWS = 16

# Two SNR-100 signals are distinguishable above a mismatch of 1/(2 SNR^2).
SNR = 100
THRESHOLD = 1 / (2 * SNR ** 2)

# A tone below (1/2 + A_frac) Delta F leaks into the dropped m = 0 band: the band
# edge is Delta F / 2, but the Meyer window reaches A_frac Delta F past it. The
# taper study measured 1e-1 of the power there at 0.55 Delta F, 1e-5 at 0.70 and
# none above 0.75. WDM_transform's default A_frac.
A_FRAC = 0.25

DT_BAND = (3600.0, 36000.0)                     # 1-10 h, recommended in fig 4
LEGEND_DECADES = 2.3                           # headroom above the data

TONES = tuple(f"{f * 1e3:g} mHz" for f in FREQUENCIES)
NAMES = TONES + ("EMRI",)
COLOURS = dict(zip(NAMES, PALETTE[:3] + [DARK]))


def compute(nfs):
    import jax
    from lisaorbits import InterpolatedOrbits
    from lisaorbits.file import read_orbit_file

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    data = read_orbit_file(str(ORBIT_FILE))
    orbit_times = data.t0 + data.dt * np.arange(data.size)
    orbits = InterpolatedOrbits(orbit_times, data.positions, data.velocities)
    # 200 d in: the widest tiling pads about 160 d ahead of the signal.
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

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


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    delta_t = res["Nf"] * DT
    values = np.concatenate([res[name] for name in NAMES])
    bottom = np.nanmin(values) / 3
    data_top = min(3.0, max(10 * THRESHOLD, 3 * np.nanmax(values)))
    # Empty headroom above the data holds the legend; every corner below has data.
    top = data_top * 10 ** LEGEND_DECADES

    fig, ax = plt.subplots(figsize=(3.375, 3.2), layout="constrained")
    ax.axhspan(THRESHOLD, top, color=SHADE, lw=0, zorder=0)
    ax.axvspan(*DT_BAND, color=PALETTE[3], alpha=0.3, lw=0, zorder=0.5)
    ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)

    markers = dict(ms=2.5, mec="white", mew=0.4)
    for name in TONES:
        f = FREQUENCIES[TONES.index(name)]
        ax.plot(delta_t, res[name], color=COLOURS[name], lw=0.5, marker="o", zorder=3,
                label=name, **markers)
        leaks = f < (0.5 + A_FRAC) / (2 * delta_t)      # f < (1/2 + A_frac) Delta F
        ax.plot(delta_t[leaks], res[name][leaks], ls="none", marker="o", ms=2.5,
                mfc="white", mec=COLOURS[name], mew=0.7, zorder=5)
    ax.plot(delta_t, res["EMRI"], color=COLOURS["EMRI"], lw=1.2, marker="D", zorder=4,
            label="EMRI", **dict(markers, ms=2.2))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", ms=2.5, mfc="white", mec=INK,
                          mew=0.7))
    labels.append(r"leaks into $m = 0$")
    # The swatch sits on the band itself, so a solid outline keeps it visible.
    handles.append(Patch(facecolor=to_rgba(PALETTE[3], 0.3), edgecolor=PALETTE[3], lw=0.8))
    labels.append(r"1--10\,h")
    ax.legend(handles, labels, loc="upper center", fontsize="small", ncols=3,
              handlelength=1.4, handletextpad=0.5, columnspacing=0.9, borderaxespad=0.4)

    ax.set(xscale="log", yscale="log", ylim=(bottom, top),
           xlim=(delta_t[0] / 1.3, delta_t[-1] * 1.3),
           xlabel=r"$\Delta T = N_f\,\delta t$ [s]", ylabel="mismatch")
    # No ticks in the legend's headroom: mismatch there means nothing.
    decades = np.arange(np.ceil(np.log10(bottom)), 1)
    ax.yaxis.set_major_locator(FixedLocator(10 ** decades[decades % 2 == 1]))
    ax.yaxis.set_minor_locator(FixedLocator(10 ** decades))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # The grid stops at the data, so none of it runs through the legend.
    ax.grid(axis="y", color=HAIRLINE, lw=0.4)
    ax.vlines(10.0 ** np.arange(2, 7), bottom, data_top, color=HAIRLINE, lw=0.4, zorder=0.6)
    ax.set_axisbelow(True)

    # Delta F on top: the same axis read the other way, Delta F Delta T = 1/2.
    def reciprocal_mhz(x):
        """Delta F in mHz from Delta T in s, and back: the map is its own inverse."""
        with np.errstate(divide="ignore"):
            return 5e2 / np.asarray(x, dtype=float)

    top_axis = ax.secondary_xaxis("top", functions=(reciprocal_mhz, reciprocal_mhz))
    top_axis.set_xlabel(r"$\Delta F$ [mHz]")

    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig5_mismatch_vs_dT.pdf")
    print(f"wrote {HERE / 'figures' / 'fig5_mismatch_vs_dT.pdf'}")


if __name__ == "__main__":
    if "--only" in sys.argv:
        compute([int(sys.argv[sys.argv.index("--only") + 1])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(NF)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
