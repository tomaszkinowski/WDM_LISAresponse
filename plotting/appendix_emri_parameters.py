"""Appendix figures C1, D1 and D2: mismatch of the wavelet one-arm response
against the time-domain reference for the EMRI (parameters.py) on link 1 -> 2, at
Delta T = 1000 s, 3600 s and 36000 s (as in figure 7), as one setting changes:
  C1  P, the number of interpolation points in the time-delay filter tables
  D1  A_frac, the Meyer window's flat-top fraction (B_frac = 1 - 2 A_frac)
  D2  d, the steepness of the Meyer window's transition
One script for both appendices, so the three sweeps share each tiling's reference.
Everything else is held at the paper's values (P = 128, A_frac = 0.25, d = 4,
L = 25), with window and cut as in figure 5. The reference is a time series, so
it is computed once per tiling and transformed with each setting's own WDM
transform. The vertical line marks the value used everywhere else.

Run with no arguments to compute (or reuse the cache) and plot all three.
``--only NF SETTING VALUE`` (SETTING one of P, A_frac, d) runs one point, prints
it and caches nothing - for checking time and memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _colours import DARK, HAIRLINE, MUTED
from parameters import (DT, EMRI_PARAMS, MAX_LAG_L, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "appendix_emri_parameters.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = np.array([250, 900, 9000])                 # Delta T = 1000 s, 3600 s, 36000 s

# WDM_transform's defaults, which every other figure uses.
A_FRAC = 0.25
D = 4

SETTINGS = {
    "P": dict(values=np.array([2, 4, 8, 16, 32, 64, 128, 256, 512]),
              default=NUM_INTERP_POINTS, xlabel=r"interpolation points $P$",
              figure="figC1_interp_points"),
    "A_frac": dict(values=np.round(np.arange(0.05, 0.46, 0.05), 2), default=A_FRAC,
                   xlabel=r"flat-top fraction $A_\mathrm{frac}$", figure="figD1_A_frac"),
    "d": dict(values=np.array([1, 2, 3, 4, 6, 8, 12, 16]), default=D,
              xlabel=r"window steepness $d$", figure="figD2_d"),
}


# One source, so the tilings share the EMRI's colour and differ in line style.
STYLES = ({"ls": "-", "marker": "D"}, {"ls": "--", "marker": "s"},
          {"ls": ":", "marker": "^"})


LEGEND_DECADES = 0.9                            # headroom above the data


def tiling_label(Nf):
    """The tiling alone; the legend puts "Delta T =" once in front of the row."""
    return rf"{Nf * DT:.0f}\,s"


def configuration(setting, value):
    """P, A_frac and d for one point of one sweep, the others at their defaults."""
    config = dict(P=NUM_INTERP_POINTS, A_frac=A_FRAC, d=D)
    config[setting] = value
    return int(config["P"]), float(config["A_frac"]), int(config["d"])


def compute(nfs, points):
    """``points`` maps each setting to the values to run it at."""
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
    theta, phi = EMRI_PARAMS["theta"], EMRI_PARAMS["phi"]

    tick = time.perf_counter()
    td_gen = waveforms.emri_generator()
    _, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen
    print(f"EMRI generated ({time.perf_counter() - tick:.0f} s)", flush=True)

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    out = {"Nf": np.asarray(nfs)}
    for setting, values in points.items():
        out[f"{setting}_values"] = np.asarray(values)
        out[setting] = np.full((len(nfs), len(values)), np.nan)

    for j, Nf in enumerate(nfs):
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG_L,
                                      taper_samples=taper)
        t0 = start - info["pad_left"] * DT
        cut = utils.wdm_cut_info(info)
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])

        def build(P, A_frac, d):
            wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"],
                                        d=d, A_frac=A_frac)
            return wdm, LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                                     num_interp_points=P, link_convention=(link,))

        # The reference is a time series: the WDM settings only enter its transform.
        tick = time.perf_counter()
        _, response = build(*configuration("P", NUM_INTERP_POINTS))
        reference = np.asarray(response.one_arm_response_time_domain(
            theta, phi, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS))
        del response
        print(f"Nf = {Nf:>5}: reference ({time.perf_counter() - tick:.0f} s)", flush=True)

        for setting, values in points.items():
            for i, value in enumerate(values):
                tick = time.perf_counter()
                wdm, response = build(*configuration(setting, value))
                ours = response.one_arm_response(theta, phi, wdm.dwt(hp), wdm.dwt(hc),
                                                 link=link)
                out[setting][j, i] = utils.wdm_mismatch(
                    utils.apply_wdm_cut(np.asarray(ours), cut),
                    utils.apply_wdm_cut(np.asarray(wdm.dwt(reference)), cut))["mismatch"]
                del wdm, response
                jax.clear_caches()
                print(f"Nf = {Nf:>5}, dT = {Nf * DT:7.0f} s, {setting} = {value:g}: "
                      f"{out[setting][j, i]:.2e}   ({time.perf_counter() - tick:.0f} s)",
                      flush=True)

    return out


def plot(res, setting):
    spec = SETTINGS[setting]
    # Plot only the values SETTINGS asks for; the cache may hold more.
    keep = np.isin(res[f"{setting}_values"], spec["values"])
    values, mismatch = res[f"{setting}_values"][keep], res[setting][:, keep]
    bottom = np.nanmin(mismatch) / 3
    # Headroom above the data holds a one-row legend, the same in all three figures.
    top = min(3.0, 3 * np.nanmax(mismatch)) * 10 ** LEGEND_DECADES

    fig, ax = plt.subplots(figsize=(3.375, 2.7), layout="constrained")
    ax.axvline(spec["default"], color=MUTED, lw=0.6, zorder=1)

    for Nf, row, style in zip(res["Nf"], mismatch, STYLES):
        ax.plot(values, row, color=DARK, lw=1.0, ms=2.8, mec="white", mew=0.4,
                zorder=3, label=tiling_label(Nf), **style)

    # A blank-handle first entry puts "Delta T =" in front of the whole row.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([Line2D([], [], lw=0)] + handles, [r"$\Delta T =$"] + labels,
              loc="upper center", ncols=len(labels) + 1, fontsize="small",
              handlelength=2.2, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.4,
              frameon=True, fancybox=False, framealpha=1, facecolor="white",
              edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", ylim=(bottom, top), xlabel=spec["xlabel"], ylabel="mismatch")
    # P doubles and d is an integer that spans 1 to 16: both read best on log2.
    if setting in ("P", "d"):
        ax.set_xscale("log", base=2)
        ax.set_xticks(values, [f"{v:d}" for v in values])
        ax.minorticks_off()
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    (HERE / "figures" / "appendix").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figures" / "appendix" / f"{spec['figure']}.pdf")
    print(f"wrote {HERE / 'figures' / 'appendix' / spec['figure']}.pdf")


if __name__ == "__main__":
    if "--only" in sys.argv:
        k = sys.argv.index("--only")
        setting = sys.argv[k + 2]
        compute([int(sys.argv[k + 1])], {setting: [float(sys.argv[k + 3])]})
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(NF, {name: spec["values"] for name, spec in SETTINGS.items()})
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plt.style.use(HERE / "paper.mplstyle")
    for setting in SETTINGS:
        plot(results, setting)
