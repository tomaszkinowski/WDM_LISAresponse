"""Investigation (not a paper figure): how much lag L the time-shift kernel needs
as the Meyer window changes, for the 30 mHz Galactic binary of figures D3 and D4
on link 1 -> 2 at Delta T = 1000 s (Nf = 250, m = 60). Two figures, matching
investigation_filter_tails.py window for window and colour for colour:
  l_convergence_d       d = 2, 4, 8, 16 at A_frac = 0.25 (no d = 1, as in D4)
  l_convergence_A_frac  A_frac = 0.1, 0.25, 0.4, 0.45 at d = 4
Solid lines are the tone on a band centre (r = 0), where nothing but the kernel's
truncation lifts it off the floor; dashed lines halfway between band centres
(r = 1/2), where the off-bin error peaks.

A steeper (large d) or narrower (large A_frac) transition is longer in time, so
its filters reach further in lag and need more L before the mismatch settles.
Every point uses one grid padded for the largest L, so only the kernel changes
along a curve, and one time-domain reference per r, transformed with each
window's own WDM transform. Everything else is as in figure 6. The vertical line
marks L = 25. (An EMRI version of this sweep is cached as
investigation_l_convergence_emri.npz.)

Run with no arguments to compute (or reuse the cache) and plot. ``--only R A_FRAC
D L`` runs one point, prints it and caches nothing.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _colours import HAIRLINE, INK, MUTED, PALETTE, SHADE
from parameters import (A_CROSS, A_PLUS, DT, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        N_SIGNAL, NUM_INTEGRAL_POINTS, NUM_INTERP_POINTS, PHI_0,
                        WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "investigation_l_convergence.npz"
OUT = HERE / "figures" / "investigations"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = 250                                        # Delta T = 1000 s, as in figure 6
DELTA_F = 1 / (2 * NF * DT)
F_GW = FREQUENCIES[2]                           # 30 mHz
M_BIN = int(round(F_GW / DELTA_F))
R = np.array([0.0, 0.5])                        # on a band centre, and halfway
LAGS = np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
PAD_LAG = int(LAGS.max())

# The same colour for the same window as in investigation_filter_tails.py.
COLOURS = {(0.25, 1): PALETTE[0], (0.25, 2): PALETTE[1], (0.25, 4): PALETTE[2],
           (0.25, 8): PALETTE[3], (0.25, 16): PALETTE[4]}
A_COLOURS = dict(zip([(a, 4) for a in (0.1, 0.25, 0.4, 0.45)], PALETTE))
SWEEPS = {
    "d": dict(windows=[(0.25, d) for d in (2, 4, 8, 16)], colours=COLOURS,
              label=lambda a, d: rf"$d = {d}$", figure="l_convergence_d"),
    "A_frac": dict(windows=list(A_COLOURS), colours=A_COLOURS,
                   label=lambda a, d: rf"$A_\mathrm{{frac}} = {a:g}$",
                   figure="l_convergence_A_frac"),
}
# Every (A_frac, d) to run: the default sits in both sweeps but is run once.
WINDOWS = tuple(dict.fromkeys(w for spec in SWEEPS.values() for w in spec["windows"]))
STYLES = {0.0: "-", 0.5: "--"}

# Two SNR-100 signals are distinguishable above a mismatch of 1/(2 SNR^2).
SNR = 100
THRESHOLD = 1 / (2 * SNR ** 2)
LEGEND_DECADES = 1.3                            # headroom above the data from L = 20 on


def compute(rs, windows, lags):
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
        N_SIGNAL, NF, max_lag_L=PAD_LAG,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    t0 = start - info["pad_left"] * DT
    cut = utils.wdm_cut_info(info)

    def build(A_frac, d, L):
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=info["target_length"], d=d,
                                    A_frac=A_frac)
        return wdm, LISAresponse(wdm, orbits, t0=t0, max_lag_L=int(L),
                                 num_interp_points=NUM_INTERP_POINTS,
                                 link_convention=(link,))

    out = dict(r=np.asarray(rs), L=np.asarray(lags),
               A_frac=np.array([a for a, _ in windows]), d=np.array([d for _, d in windows]),
               mismatch=np.full((len(windows), len(rs), len(lags)), np.nan))
    for j, r in enumerate(rs):
        _, h_plus, h_cross = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
            model_kwargs=dict(A_plus=A_PLUS, A_cross=A_CROSS,
                              f=(M_BIN + r) * DELTA_F, phi_0=PHI_0),
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])
        # The reference is a time series: the window only enters its transform.
        _, response = build(0.25, 4, MAX_LAG_L)
        reference = np.asarray(response.one_arm_response_time_domain(
            GC_THETA, GC_PHI, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS))
        del response
        jax.clear_caches()

        for w, (A_frac, d) in enumerate(windows):
            for i, L in enumerate(lags):
                tick = time.perf_counter()
                wdm, response = build(A_frac, d, L)
                ours = response.one_arm_response(GC_THETA, GC_PHI, wdm.dwt(hp),
                                                 wdm.dwt(hc), link=link)
                out["mismatch"][w, j, i] = utils.wdm_mismatch(
                    utils.apply_wdm_cut(np.asarray(ours), cut),
                    utils.apply_wdm_cut(np.asarray(wdm.dwt(reference)), cut))["mismatch"]
                del wdm, response, ours
                jax.clear_caches()
                print(f"r = {r:.1f}, A_frac = {A_frac:.2f}, d = {d:2d}, L = {L:3d}: "
                      f"{out['mismatch'][w, j, i]:.3e}   ({time.perf_counter() - tick:.0f} s)",
                      flush=True)
    return out


def plot(res, sweep):
    spec = SWEEPS[sweep]
    rows = [int(np.flatnonzero((res["A_frac"] == a) & (res["d"] == d))[0])
            for a, d in spec["windows"]]
    mismatch = res["mismatch"][rows]
    bottom = np.nanmin(mismatch) / 3
    # Headroom above the off-bin curves holds the legend.
    top = 3 * max(10 * THRESHOLD, np.nanmax(mismatch[:, :, 2:])) * 10 ** LEGEND_DECADES

    fig, ax = plt.subplots(figsize=(3.375, 2.9), layout="constrained")
    ax.axhspan(THRESHOLD, top, color=SHADE, lw=0, zorder=0)
    ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(MAX_LAG_L, color=MUTED, lw=0.6, zorder=1)
    for window, per_r in zip(spec["windows"], mismatch):
        for r, row in zip(res["r"], per_r):
            ax.plot(res["L"], row, color=spec["colours"][window], lw=1.0,
                    ls=STYLES[float(r)], marker="o", ms=2.2, mec="white", mew=0.4,
                    zorder=3, label=spec["label"](*window) if r == 0 else None)

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=INK, lw=1.0, ls=STYLES[r]) for r in (0.0, 0.5)]
    labels += [r"$r = 0$", r"$r = 1/2$"]
    ax.legend(handles, labels, loc="upper right", fontsize="small", ncols=2,
              frameon=True, fancybox=False, framealpha=1, facecolor="white",
              edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", xlim=(0, res["L"].max() + 2), ylim=(bottom, top),
           xlabel=r"maximum lag $L$", ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{spec['figure']}.pdf")
    print(f"wrote {OUT / spec['figure']}.pdf")


if __name__ == "__main__":
    if "--only" in sys.argv:
        k = sys.argv.index("--only")
        compute([float(sys.argv[k + 1])],
                [(float(sys.argv[k + 2]), int(sys.argv[k + 3]))], [int(sys.argv[k + 4])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(R, WINDOWS, LAGS)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plt.style.use(HERE / "paper.mplstyle")
    for sweep in SWEEPS:
        plot(results, sweep)
