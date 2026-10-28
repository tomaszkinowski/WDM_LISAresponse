"""Does L = 40 put figure 5's 0.3 mHz points back inside the on/off-bin band?

With the amplitude pre-warp (provisional_prewarp.py) the near-on-bin floor is the
L = 25 lag truncation, which is not monotonic in the offset r from a band centre,
so the 0.3 mHz source can fall below the on-bin curve. At the tilings where it
does, this compares the source, its on-bin (r = 0, nearest band) and off-bin
(r = 1/2) bounds, and r = 0.05, 0.1, 0.25 above the band below, at L = 25 and
L = 40. One grid per tiling, padded for L = 40, so only the kernel's truncation
changes; each time-domain reference is computed once, on that grid.

    python investigation_prewarp_lag_bounds.py [Nf ...] [--short]
"""
import sys
import time

import numpy as np

import fig5_mismatch_vs_dT as fig5
import provisional_prewarp
from parameters import (DT, GC_PHI, GC_THETA, MONOCHROMATIC_PARAMS, N_SIGNAL,
                        NUM_INTEGRAL_POINTS, NUM_INTERP_POINTS, WINDOW_FUNCTION,
                        WINDOW_KWARGS)

NFS = (444, 790, 7906, 25000)
LAGS = (25, 40)
OFFSETS = (0.05, 0.1, 0.25)
TONE = 0                                            # 0.3 mHz


def main(nfs, n_signal=N_SIGNAL):
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    provisional_prewarp.patch()
    orbits, start, _, link = fig5.setup()
    signal_times = start + DT*np.arange(n_signal)
    params = MONOCHROMATIC_PARAMS[TONE]
    f0 = params["f"]
    taper = waveforms.tukey_taper_samples(n_signal, WINDOW_KWARGS["alpha"])

    for Nf in nfs:
        tick = time.perf_counter()
        delta_f = 1/(2*Nf*DT)
        below = np.floor(f0/delta_f)
        freqs = {"source": f0,
                 "on": np.round(f0/delta_f)*delta_f,
                 "off": (max(below, 1) + 0.5)*delta_f,
                 **{f"r={r}": (below + r)*delta_f for r in OFFSETS}}

        info = utils.wdm_padding_info(n_signal, Nf, max_lag_L=max(LAGS),
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        t0 = start - info["pad_left"]*DT
        cut = utils.wdm_cut_info(info)
        responses = {L: LISAresponse(wdm, orbits, t0=t0, max_lag_L=L,
                                     num_interp_points=NUM_INTERP_POINTS,
                                     link_convention=(link,)) for L in LAGS}

        rows = {}
        for name, f in freqs.items():
            _, hp, hc = waveforms.generate_windowed_waveform(
                "monochromatic", t0=start, dt=DT, N=n_signal,
                model_kwargs=dict(params, f=f), window_function=WINDOW_FUNCTION,
                window_kwargs=WINDOW_KWARGS)
            _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                 info["pad_left"], info["pad_right"])
            reference = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
                responses[max(LAGS)].one_arm_response_time_domain(
                    GC_THETA, GC_PHI, hp, hc, link=link,
                    num_integral_points=NUM_INTEGRAL_POINTS)))), cut)
            wp, wc = wdm.dwt(hp), wdm.dwt(hc)
            rows[name] = {L: utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(
                responses[L].one_arm_response(GC_THETA, GC_PHI, wp, wc, link=link)),
                cut), reference)["mismatch"] for L in LAGS}

        print(f"--- Nf = {Nf}, dT = {Nf*DT:.0f} s, source f/dF = {f0/delta_f:.3f}", flush=True)
        for L in LAGS:
            lo, hi = sorted((rows["on"][L], rows["off"][L]))
            src = rows["source"][L]
            where = "below" if src < lo else ("above" if src > hi else "inside")
            print(f"  L = {L}: source {src:.1e} ({where})  on {rows['on'][L]:.1e}  "
                  f"off {rows['off'][L]:.1e}  | "
                  + "  ".join(f"{k} {rows[k][L]:.1e}" for k in rows
                              if k.startswith("r=")), flush=True)
        del responses, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_signal = 2**19 if "--short" in sys.argv else N_SIGNAL
    main([int(a) for a in args] or list(NFS), n_signal)
