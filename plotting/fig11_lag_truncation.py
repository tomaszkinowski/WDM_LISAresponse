"""Figure 11 with the corrected wavelet response: mismatch against the time-domain
reference on link 1 -> 2 at Delta T = 1e4 s as the maximum lag L changes, with
split_amp_sinc_first (_corrected_response.py) in place of production. Every
correction table takes the same L as the shift. Grid, signals, references and plot
are figure 11's own.

Writes figures/fig11_lag_truncation_corrected.pdf; cache
cache/investigation_fig11_corrected.npz. ``--only L`` runs one L, prints it and caches
nothing. ../prototypes/delay_rate_correction must be on PYTHONPATH.
"""

import sys
import time

import numpy as np

import fig11_lag_truncation as fig11
from parameters import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

CACHE = fig11.HERE / "cache" / "investigation_fig11_corrected.npz"
FIGURE = fig11.HERE / "figures" / "fig11_lag_truncation_corrected.pdf"


def compute(lags):
    import jax
    from lisaorbits import InterpolatedOrbits
    from lisaorbits.file import read_orbit_file

    import WDM
    import _corrected_response as corrected
    import _emri_comparison as emri
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    data = read_orbit_file(str(fig11.ORBIT_FILE))
    orbit_times = data.t0 + data.dt * np.arange(data.size)
    orbits = InterpolatedOrbits(orbit_times, data.positions, data.velocities)
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

    sky = {name: (GC_THETA, GC_PHI) for name in fig11.TONES}
    sky["EMRI"] = (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])
    signals = {}
    for name, params in zip(fig11.TONES, MONOCHROMATIC_PARAMS):
        _, h_plus, h_cross = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        signals[name] = (h_plus, h_cross)
    # the same EMRI as figure 11 generates, from the figure 12/13 cache
    signals["EMRI"] = emri.waveform(N_SIGNAL)

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    info = utils.wdm_padding_info(N_SIGNAL, fig11.NF, max_lag_L=fig11.PAD_LAG,
                                  taper_samples=taper)
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=fig11.NF, N=info["target_length"])
    t0 = start - info["pad_left"] * DT
    cut = utils.wdm_cut_info(info)
    times = np.asarray(t0 + wdm.times)

    # The warp needs only the constellation centre, the same at every L.
    response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=MAX_LAG_L,
                            num_interp_points=NUM_INTERP_POINTS, link_convention=(link,))
    consts = corrected.constants(response, wdm)
    warped, references = {}, {}
    for name, (h_plus, h_cross) in signals.items():
        tick = time.perf_counter()
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])
        warp = (corrected.tone_warp(MONOCHROMATIC_PARAMS[fig11.TONES.index(name)], start)
                if name in fig11.TONES else corrected.spline_warp(times, hp, hc))
        warped[name] = corrected.warped_transforms(response, wdm, consts, *sky[name], warp)
        references[name] = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
            response.one_arm_response_time_domain(
                *sky[name], hp, hc, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)))), cut)
        print(f"warp and reference {name:>8}: {time.perf_counter() - tick:.0f} s",
              flush=True)
    del response, consts, signals

    out = dict(Nf=np.asarray(fig11.NF), L=np.asarray(lags),
               **{name: np.full(len(lags), np.nan) for name in fig11.NAMES})
    for i, L in enumerate(lags):
        tick = time.perf_counter()
        response = LISAresponse(wdm, orbits, t0=t0, max_lag_L=int(L),
                                num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        consts = corrected.constants(response, wdm)
        for name in fig11.NAMES:
            ours = corrected.links(wdm, consts, *sky[name], *warped[name])[0]
            out[name][i] = utils.wdm_mismatch(
                utils.apply_wdm_cut(ours, cut), references[name])["mismatch"]
        del response, consts
        jax.clear_caches()
        print(f"L = {L:>2}: " + "  ".join(f"{name} {out[name][i]:.2e}"
                                          for name in fig11.NAMES)
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
    return out


if __name__ == "__main__":
    if "--only" in sys.argv:
        compute([int(sys.argv[sys.argv.index("--only") + 1])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(fig11.LAGS)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    fig11.plot(results, FIGURE, legend_loc="upper right")
