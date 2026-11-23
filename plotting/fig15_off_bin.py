"""Figure 15: mismatch of the wavelet one-arm response against the time-domain
reference as a tone moves across a frequency band, on link 1 -> 2 at
Delta T = 1e4 s (Nf = 2500), the middle of the 3600-36000 s band figure 3
recommends.

Each of the three example Galactic binaries (parameters.py) is moved to
f = (m + r) Delta F, with m its nearest band at this tiling and r from 0 (on the
centre of band m) to 1 (on the centre of band m + 1), in steps of 0.05. Everything
else - sky, amplitudes, window, cut - is as in figure 14. With one tiling only,
the reference is computed on its grid directly. Each curve is labelled by its
frequency at r = 0, m Delta F.

Run with no arguments to compute (or reuse the cache) and plot. ``--only R`` runs
one r for every tone, prints it and caches nothing - for checking time and memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _colours import HAIRLINE, INK, MUTED, PALETTE
from _cache import load_or_compute
from parameters import (A_CROSS, A_PLUS, DT, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        N_SIGNAL, NUM_INTEGRAL_POINTS, NUM_INTERP_POINTS, PHI_0,
                        WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "fig15_off_bin.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = 2500                                       # Delta T = 1e4 s
DELTA_F = 1 / (2 * NF * DT)
R = np.linspace(0.0, 1.0, 21)

LEGEND_DECADES = 1.9                           # headroom above the data
# Each band's Meyer window spans (1 - A_frac) Delta F either side of its centre,
# (1/2 - A_frac) Delta F past its edges, so a tone lies in the windows of both
# bands m and m + 1 only for A_frac < r < 1 - A_frac. WDM_transform's default A_frac.
A_FRAC = 0.25

TONES = tuple(f"{f * 1e3:g} mHz" for f in FREQUENCIES)
COLOURS = dict(zip(TONES, PALETTE[:3]))
# Each tone's nearest band at this tiling; it moves from there towards m + 1.
M_BIN = {name: int(round(f / DELTA_F)) for name, f in zip(TONES, FREQUENCIES)}


def compute(rs):
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
    # 200 d in, as in figure 14, so the two figures see the same stretch of orbit.
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

    info = utils.wdm_padding_info(
        N_SIGNAL, NF, max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=info["target_length"])
    response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                            max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                            link_convention=(link,))
    cut = utils.wdm_cut_info(info)

    out = dict(r=np.asarray(rs), m=np.array([M_BIN[name] for name in TONES]),
               **{name: np.full(len(rs), np.nan) for name in TONES})
    for name in TONES:
        tick = time.perf_counter()
        for i, r in enumerate(rs):
            _, h_plus, h_cross = waveforms.generate_windowed_waveform(
                "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
                model_kwargs=dict(A_plus=A_PLUS, A_cross=A_CROSS,
                                  f=(M_BIN[name] + r) * DELTA_F, phi_0=PHI_0),
                window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
            _, h_plus, h_cross = utils.apply_zero_padding(
                signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])
            ours = response.one_arm_response(GC_THETA, GC_PHI, wdm.dwt(h_plus),
                                             wdm.dwt(h_cross), link=link)
            reference = response.one_arm_response_time_domain(
                GC_THETA, GC_PHI, h_plus, h_cross, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)
            out[name][i] = utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut),
                utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(reference))), cut)
            )["mismatch"]
            if len(rs) == 1:
                print(f"{name:>8}, r = {r:.2f}: {out[name][i]:.2e}", flush=True)
        print(f"{name:>8} (m = {M_BIN[name]}): max {np.nanmax(out[name]):.1e}   "
              f"({time.perf_counter() - tick:.0f} s)", flush=True)
    jax.clear_caches()
    return out


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    values = np.concatenate([res[name] for name in TONES])
    bottom = np.nanmin(values) / 3
    # Headroom above the peaks holds the legend, as in figure 14: the curves leave
    # no corner free.
    top = min(3.0, 3 * np.nanmax(values)) * 10 ** LEGEND_DECADES

    fig, ax = plt.subplots(figsize=(3.375, 2.8), layout="constrained")
    # r = 1/2 is halfway between the centres of bands m and m + 1, as far as a
    # tone can sit from any band centre: its power splits equally between the two.
    ax.axvline(0.5, color=MUTED, lw=0.6, zorder=1)
    overlap = dict(color=INK, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    for edge in (A_FRAC, 1 - A_FRAC):
        ax.axvline(edge, **overlap)

    markers = dict(ms=2.5, mec="white", mew=0.4)
    for name, m in zip(TONES, res["m"]):
        ax.plot(res["r"], res[name], color=COLOURS[name], lw=0.8, marker="o", zorder=3,
                label=rf"{m * DELTA_F * 1e3:g} mHz ($m = {m}$)", **markers)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **overlap))
    labels.append("window overlap")
    ax.legend(handles, labels, loc="upper left", fontsize="small", handlelength=1.4, handletextpad=0.5,
              borderaxespad=0.4, frameon=True, fancybox=False, framealpha=1,
              facecolor="white", edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", xlim=(0, 1), ylim=(bottom, top),
           xlabel=r"$r$, where $f = (m + r)\,\Delta F$", ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig15_off_bin.pdf")
    print(f"wrote {HERE / 'figures' / 'fig15_off_bin.pdf'}")


if __name__ == "__main__":
    if "--only" in sys.argv:
        compute([float(sys.argv[sys.argv.index("--only") + 1])])
        sys.exit()
    results = load_or_compute(CACHE, lambda: compute(R))
    plot(results)
