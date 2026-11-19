"""Figure 11: mismatch of the wavelet one-arm response against the time-domain
reference as the time-shift kernel's maximum lag L changes, on link 1 -> 2 at
Delta T = 1e4 s (Nf = 2500), the middle of the 3600-36000 s band figure 2
recommends, for the three example Galactic binaries and the EMRI (parameters.py).

At this tiling the three tones sit on band centres, so their curves follow the
lag truncation alone down to whatever lies beneath it; the EMRI shows where a
realistic source stops depending on L.

The grid is padded for the largest L whatever L is tested, so the transform, the
compared rows and the time-domain references are the same along each curve and
only the kernel's truncation changes; each reference is computed once, on the
grid directly. Window, cut and the rest are as in figure 8. The vertical line
marks the L used everywhere else.

Run with no arguments to compute (or reuse the cache) and plot. ``--only L``
runs one L for every signal, prints it and caches nothing - for checking time and
memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import DARK, HAIRLINE, MUTED, PALETTE
from parameters import (DT, EMRI_PARAMS, FREQUENCIES, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "fig11_lag_truncation.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

NF = 2500                                       # Delta T = 1e4 s
LAGS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80])
PAD_LAG = int(LAGS.max())                       # padding for the widest kernel

TONES = tuple(f"{f * 1e3:g} mHz" for f in FREQUENCIES)
NAMES = TONES + ("EMRI",)
COLOURS = dict(zip(NAMES, PALETTE[:3] + [DARK]))
MARKERS = dict(zip(NAMES, ("o", "s", "^", "D")))


def compute(lags):
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
    # 200 d in, as in figure 8, so the figures see the same stretch of orbit.
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

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    info = utils.wdm_padding_info(N_SIGNAL, NF, max_lag_L=PAD_LAG, taper_samples=taper)
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=info["target_length"])
    t0 = start - info["pad_left"] * DT
    cut = utils.wdm_cut_info(info)

    response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                            num_interp_points=NUM_INTERP_POINTS, link_convention=(link,))
    coefficients, references = {}, {}
    for name, (h_plus, h_cross) in signals.items():
        tick = time.perf_counter()
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])
        coefficients[name] = (wdm.dwt(hp), wdm.dwt(hc))
        references[name] = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
            response.one_arm_response_time_domain(
                *sky[name], hp, hc, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)))), cut)
        print(f"reference {name:>8}: {time.perf_counter() - tick:.0f} s", flush=True)
    del response, signals

    out = dict(Nf=np.asarray(NF), L=np.asarray(lags),
               **{name: np.full(len(lags), np.nan) for name in NAMES})
    for i, L in enumerate(lags):
        tick = time.perf_counter()
        # L enters the filter tables, so every L needs a response of its own.
        response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=int(L),
                                num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        for name in NAMES:
            ours = response.one_arm_response(*sky[name], *coefficients[name], link=link)
            out[name][i] = utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut), references[name])["mismatch"]
        del response
        jax.clear_caches()
        print(f"L = {L:>2}: " + "  ".join(f"{name} {out[name][i]:.2e}" for name in NAMES)
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)

    return out


def plot(res, path=HERE / "figures" / "fig11_lag_truncation.pdf"):
    plt.style.use(HERE / "paper.mplstyle")
    values = np.concatenate([res[name] for name in NAMES])
    bottom = np.nanmin(values) / 3
    top = min(3.0, 3 * np.nanmax(values))

    fig, ax = plt.subplots(figsize=(3.375, 2.6), layout="constrained")
    ax.axvline(MAX_LAG_L, color=MUTED, lw=0.6, zorder=1)

    for name in NAMES:
        ax.plot(res["L"], res[name], color=COLOURS[name], lw=1.2 if name == "EMRI" else 0.8,
                marker=MARKERS[name], ms=2.8, mec="white", mew=0.4, zorder=3, label=name)

    ax.legend(loc="center right", fontsize="small", handlelength=2.2, handletextpad=0.5,
              borderaxespad=0.4, frameon=True, fancybox=False, framealpha=1,
              facecolor="white", edgecolor=HAIRLINE).get_frame().set_linewidth(0.6)
    ax.set(yscale="log", xlim=(0, res["L"].max() + 1), ylim=(bottom, top),
           xlabel=r"maximum lag $L$", ylabel="mismatch")
    ax.grid(color=HAIRLINE, lw=0.4)
    ax.set_axisbelow(True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    if "--only" in sys.argv:
        compute([int(sys.argv[sys.argv.index("--only") + 1])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(LAGS)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
