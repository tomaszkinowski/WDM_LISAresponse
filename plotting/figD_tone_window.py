"""Appendix figures D3 and D4: figure 6 for the 30 mHz Galactic binary alone, as
the Meyer window changes:
  D3  A_frac, the flat-top fraction (B_frac = 1 - 2 A_frac), at d = 4
  D4  d, the steepness of the transition, at A_frac = 0.25
on link 1 -> 2 at Delta T = 1000 s (Nf = 250). The tone moves to f = (m + r) Delta F
across its band, r from 0 to 1 in steps of 0.05, as in figure 6; everything else
is held at the paper's values. The reference is a time series, so it is computed
once per r and transformed with each window's own WDM transform. The vertical
lines in D3 mark, for each A_frac in its curve's line style, where the tone
enters both bands' windows: r = A_frac and 1 - A_frac.

Run with no arguments to compute (or reuse the cache) and plot both. ``--only R
A_FRAC D`` runs one point, prints it and caches nothing - for checking time and
memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import HAIRLINE, MUTED, PALETTE
from parameters import (A_CROSS, A_PLUS, DT, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        N_SIGNAL, NUM_INTEGRAL_POINTS, NUM_INTERP_POINTS, PHI_0,
                        WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "figD_tone_window.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = 250                                        # Delta T = 1000 s, as in figure 6
DELTA_F = 1 / (2 * NF * DT)
R = np.linspace(0.0, 1.0, 21)

F_GW = FREQUENCIES[2]                           # 30 mHz, PALETTE[2] in every figure
COLOUR = PALETTE[2]
M_BIN = int(round(F_GW / DELTA_F))

# WDM_transform's defaults, which every other figure uses.
A_FRAC = 0.25
D = 4
A_FRACS = (0.1, 0.25, 0.4)
# From d = 2: at d = 1 the transition is linear, so the window has kinks.
DS = (2, 4, 16)
# Every (A_frac, d) to run: the default sits in both sweeps but is run once.
CONFIGS = tuple(dict.fromkeys([(a, D) for a in A_FRACS] + [(A_FRAC, d) for d in DS]))

LEGEND_DECADES = 1.9                            # headroom above the data, as in figure 6

STYLES = ({"ls": "-", "marker": "o"}, {"ls": "--", "marker": "s"},
          {"ls": ":", "marker": "^"})


def compute(rs, configs):
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
    # 200 d in, as in figure 5, so the figures see the same stretch of orbit.
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

    info = utils.wdm_padding_info(
        N_SIGNAL, NF, max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    t0 = start - info["pad_left"] * DT
    cut = utils.wdm_cut_info(info)

    def build(A_frac, d):
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=info["target_length"], d=d,
                                    A_frac=A_frac)
        return wdm, LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                                 num_interp_points=NUM_INTERP_POINTS,
                                 link_convention=(link,))

    out = dict(r=np.asarray(rs), A_frac=np.array([a for a, _ in configs]),
               d=np.array([d for _, d in configs]),
               mismatch=np.full((len(configs), len(rs)), np.nan))
    for i, r in enumerate(rs):
        tick = time.perf_counter()
        _, h_plus, h_cross = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
            model_kwargs=dict(A_plus=A_PLUS, A_cross=A_CROSS,
                              f=(M_BIN + r) * DELTA_F, phi_0=PHI_0),
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        _, h_plus, h_cross = utils.apply_zero_padding(
            signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])

        # The reference is a time series: the window only enters its transform.
        _, response = build(A_FRAC, D)
        reference = np.asarray(response.one_arm_response_time_domain(
            GC_THETA, GC_PHI, h_plus, h_cross, link=link,
            num_integral_points=NUM_INTEGRAL_POINTS))
        del response

        for c, (A_frac, d) in enumerate(configs):
            wdm, response = build(A_frac, d)
            ours = response.one_arm_response(GC_THETA, GC_PHI, wdm.dwt(h_plus),
                                             wdm.dwt(h_cross), link=link)
            out["mismatch"][c, i] = utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut),
                utils.apply_wdm_cut(np.asarray(wdm.dwt(reference)), cut))["mismatch"]
            del wdm, response
            jax.clear_caches()
        print(f"r = {r:.2f}:  " + "  ".join(
            f"A={a:g},d={d:d} {m:.1e}" for (a, d), m in zip(configs, out["mismatch"][:, i]))
            + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)

    return out


def plot(res, sweep):
    if sweep == "A_frac":
        rows = [int(np.flatnonzero((res["A_frac"] == a) & (res["d"] == D))[0])
                for a in A_FRACS]
        labels = [rf"$A_\mathrm{{frac}} = {a:g}$" for a in A_FRACS]
        figure = "figD3_tone_A_frac"
    else:
        rows = [int(np.flatnonzero((res["A_frac"] == A_FRAC) & (res["d"] == d))[0])
                for d in DS]
        labels = [rf"$d = {d:d}$" for d in DS]
        figure = "figD4_tone_d"
    mismatch = res["mismatch"][rows]

    bottom = np.nanmin(mismatch) / 3
    # Headroom above the peaks holds the legend, as in figure 6.
    top = (min(3.0, 3 * np.nanmax(mismatch))
           * 10 ** LEGEND_DECADES)

    fig, ax = plt.subplots(figsize=(3.375, 2.8), layout="constrained")
    ax.axvline(0.5, color=MUTED, lw=0.6, zorder=1)
    if sweep == "A_frac":
        # Where each window starts to share the tone with the next band.
        for a, style in zip(A_FRACS, STYLES):
            for edge in (a, 1 - a):
                ax.axvline(edge, color=COLOUR, lw=0.9, ls=style["ls"], zorder=1)

    for row, label, style in zip(mismatch, labels, STYLES):
        ax.plot(res["r"], row, color=COLOUR, lw=0.8, ms=2.5, mec="white", mew=0.4,
                zorder=3, label=label, **style)

    ax.legend(loc="upper left", fontsize="small", handlelength=2.2, handletextpad=0.5,
              borderaxespad=0.4, frameon=True, fancybox=False, framealpha=1,
              facecolor="white", edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", xlim=(0, 1), ylim=(bottom, top),
           xlabel=rf"$r$, where $f = (m + r)\,\Delta F$, $m = {M_BIN}$",
           ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    (HERE / "figures" / "appendix").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figures" / "appendix" / f"{figure}.pdf")
    print(f"wrote {HERE / 'figures' / 'appendix' / figure}.pdf")


if __name__ == "__main__":
    if "--only" in sys.argv:
        k = sys.argv.index("--only")
        compute([float(sys.argv[k + 1])],
                [(float(sys.argv[k + 2]), int(sys.argv[k + 3]))])
        sys.exit()
    results = dict(np.load(CACHE)) if CACHE.exists() else None
    cached = set() if results is None else set(zip(results["A_frac"], results["d"]))
    # Run only the windows the cache lacks, and add them to it.
    missing = [config for config in CONFIGS if config not in cached]
    if missing:
        new = compute(R, missing)
        if results is not None:
            new = {key: np.concatenate([results[key], new[key]])
                   for key in ("A_frac", "d", "mismatch")} | {"r": R}
        results = new
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plt.style.use(HERE / "paper.mplstyle")
    for sweep in ("A_frac", "d"):
        plot(results, sweep)
