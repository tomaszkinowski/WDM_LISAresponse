r"""
The fully corrected response against the time-domain reference.

The `breakdown_offbin.py` tilings and cases (figure-5 tones at their on- and off-bin
frequencies, and the EMRI, link 1 -> 2), with the variants of
`response_corrections.responses`.

    python prototypes/delay_rate_correction/response_full.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from breakdown_offbin import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                            MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                            NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS,
                            LISAresponse, WDM, fig6, utils, waveforms)
from response_corrections import build_tables, responses


def main(nfs):
    orbits, start, signal_times, link = fig6.setup()
    freqs = dict(zip(("on", "off"), fig6.bound_frequencies(fig6.NF)))
    nf_index = {int(nf): j for j, nf in enumerate(fig6.NF)}
    td_gen = waveforms.emri_generator()
    _, emri_p, emri_c = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen

    for Nf in nfs:
        tick = time.perf_counter()
        info = utils.wdm_padding_info(
            N_SIGNAL, Nf, max_lag_L=MAX_LAG_L,
            taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        tables = build_tables(wdm, MAX_LAG_L, NUM_INTERP_POINTS)
        cut = utils.wdm_cut_info(info)

        def mm(a, b, c=cut):
            return utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(a), c),
                                      utils.apply_wdm_cut(np.asarray(b), c))["mismatch"]

        j = nf_index[Nf]
        cases = []
        for i, (name, params) in enumerate(zip(fig6.TONES, MONOCHROMATIC_PARAMS)):
            for which in ("on", "off"):
                f = freqs[which][i, j]
                if not np.isnan(f):
                    cases.append((f"{name} {which}", dict(params, f=f), (GC_THETA, GC_PHI)))
        cases.append(("EMRI", None, (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])))

        print(f"--- Nf = {Nf}, dT = {wdm.dT:.4g} s", flush=True)
        for label, params, (theta, phi) in cases:
            if params is None:
                hp, hc = emri_p, emri_c
            else:
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
            _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                 info["pad_left"], info["pad_right"])
            ours = responses(response, wdm, tables, theta, phi,
                             wdm.dwt(hp), wdm.dwt(hc))
            ref = np.asarray(wdm.dwt(np.asarray(response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS))))
            print(f"{label:>14} | prod {mm(ours['production'], ref):.1e}  "
                  f"first {mm(ours['first'], ref):.1e}  full {mm(ours['full'], ref):.1e}  "
                  f"full_sinc {mm(ours['full_sinc'], ref):.1e}", flush=True)
        del response, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [250, 444, 1406, 2500, 7906, 14058, 25000])
