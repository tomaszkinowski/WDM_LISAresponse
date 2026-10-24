"""Figure 7: mismatch of the wavelet one-arm response against the time-domain
reference as the time-shift kernel's maximum lag L changes, for the EMRI
(parameters.py) on link 1 -> 2, at Delta T = 1000 s (the most accurate tiling of
figure 5), 3600 s and 36000 s (the ends of the band figure 4 recommends).

Every grid is padded for the largest L whatever L is tested, so the transform,
the compared rows and the time-domain reference are the same along each curve
and only the kernel's truncation changes; the reference is computed once per
tiling, on its grid directly. Window, cut and the rest are as in figure 5. The
shading is where the two responses are distinguishable at SNR 100, and the
vertical line marks the L used everywhere else.

Run with no arguments to compute (or reuse the cache) and plot. ``--only NF L``
runs one tiling at one L, prints it and caches nothing - for checking time and
memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import DARK, HAIRLINE, MUTED, SHADE
from parameters import (DT, EMRI_PARAMS, MAX_LAG_L, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "fig7_lag_truncation.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = np.array([250, 900, 9000])                 # Delta T = 1000 s, 3600 s, 36000 s
LAGS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40])
PAD_LAG = int(LAGS.max())                       # padding for the widest kernel

# Two SNR-100 signals are distinguishable above a mismatch of 1/(2 SNR^2).
SNR = 100
THRESHOLD = 1 / (2 * SNR ** 2)

# One source, so the tilings share the EMRI's colour and differ in line style.
STYLES = ({"ls": "-", "marker": "D"}, {"ls": "--", "marker": "s"},
          {"ls": ":", "marker": "^"})


def tiling_label(Nf):
    return rf"$\Delta T = {Nf * DT:.0f}$\,s"


def compute(nfs, lags):
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
    out = dict(Nf=np.asarray(nfs), L=np.asarray(lags),
               mismatch=np.full((len(nfs), len(lags)), np.nan))
    for j, Nf in enumerate(nfs):
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=PAD_LAG,
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        t0 = start - info["pad_left"] * DT
        cut = utils.wdm_cut_info(info)
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])
        hp_wdm, hc_wdm = wdm.dwt(hp), wdm.dwt(hc)

        tick = time.perf_counter()
        response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                                num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        reference = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
            response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)))), cut)
        del response
        print(f"Nf = {Nf:>5}: reference ({time.perf_counter() - tick:.0f} s)", flush=True)

        for i, L in enumerate(lags):
            tick = time.perf_counter()
            # L enters the filter tables, so every L needs a response of its own.
            response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=int(L),
                                    num_interp_points=NUM_INTERP_POINTS,
                                    link_convention=(link,))
            ours = response.one_arm_response(theta, phi, hp_wdm, hc_wdm, link=link)
            out["mismatch"][j, i] = utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut), reference)["mismatch"]
            del response
            jax.clear_caches()
            print(f"Nf = {Nf:>5}, dT = {Nf * DT:7.0f} s, L = {L:>2}: "
                  f"{out['mismatch'][j, i]:.2e}   ({time.perf_counter() - tick:.0f} s)",
                  flush=True)
        del wdm
        jax.clear_caches()

    return out


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    bottom = np.nanmin(res["mismatch"]) / 3
    top = min(3.0, max(10 * THRESHOLD, 3 * np.nanmax(res["mismatch"])))

    fig, ax = plt.subplots(figsize=(3.375, 2.6), layout="constrained")
    ax.axhspan(THRESHOLD, top, color=SHADE, lw=0, zorder=0)
    ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(MAX_LAG_L, color=MUTED, lw=0.6, zorder=1)

    for Nf, values, style in zip(res["Nf"], res["mismatch"], STYLES):
        ax.plot(res["L"], values, color=DARK, lw=1.0, ms=2.8, mec="white", mew=0.4,
                zorder=3, label=tiling_label(Nf), **style)

    ax.legend(loc="upper right", fontsize="small", handlelength=2.2, handletextpad=0.5,
              borderaxespad=0.4, frameon=True, fancybox=False, framealpha=1,
              facecolor="white", edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", xlim=(0, res["L"].max() + 1), ylim=(bottom, top),
           xlabel=r"maximum lag $L$", ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig7_lag_truncation.pdf")
    print(f"wrote {HERE / 'figures' / 'fig7_lag_truncation.pdf'}")


if __name__ == "__main__":
    if "--only" in sys.argv:
        k = sys.argv.index("--only")
        compute([int(sys.argv[k + 1])], [int(sys.argv[k + 2])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(NF, LAGS)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
