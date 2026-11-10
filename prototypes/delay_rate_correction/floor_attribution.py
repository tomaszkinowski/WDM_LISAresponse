r"""
What sets the ~1.5e-10 floor of the shift once the amplitude is pre-warped.

A bare signal (no amplitude, no sinc) is shifted by the real per-row delay of
link 1->2, scaled by ``s``, and compared with the transform of its exact warp
h(t + s delta(t)). The filter table is rebuilt for each (L, P): L the lag
range, P the number of delay nodes on [-dT/2, dT/2] (linear interpolation).
``prod`` is the production shift, ``rate`` the first-order delay-rate
corrected one with tables of the same (L, P).

    python prototypes/delay_rate_correction/floor_attribution.py [Nf ...] [--short]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

jax.config.update("jax_enable_x64", True)

from breakdown_offbin import (DT, EMRI_PARAMS, GC_PHI, GC_THETA,
                              MONOCHROMATIC_PARAMS, N_SIGNAL, WINDOW_FUNCTION,
                              WINDOW_KWARGS, LISAresponse, WDM, fig8, rs,
                              tukey_continuous, utils, waveforms)
from rate_kernel import apply_rate_corrected_shift, build_rate_tables

L_PAD = 60
TABLES = ((25, 128), (25, 1024), (40, 128), (40, 1024), (60, 2048))
SCALES = (1.0, 0.1, 0.01)


def main(nfs, n_signal=N_SIGNAL):
    orbits, start, _, link = fig8.setup()
    signal_times = start + DT*np.arange(n_signal)
    freqs = dict(zip(("on", "off"), fig8.bound_frequencies(fig8.NF)))
    nf_index = {int(nf): j for j, nf in enumerate(fig8.NF)}
    alpha = WINDOW_KWARGS["alpha"]
    td_gen = waveforms.emri_generator()
    _, emri_p, _ = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=n_signal,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen

    for Nf in nfs:
        tick = time.perf_counter()
        info = utils.wdm_padding_info(n_signal, Nf, max_lag_L=L_PAD,
                                      taper_samples=waveforms.tukey_taper_samples(n_signal, alpha))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=25, num_interp_points=128,
                                link_convention=(link,))
        cut = utils.wdm_cut_info(info)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        dwt = lambda x: np.asarray(wdm.dwt(jnp.asarray(x)))

        def mm(a, b):
            return utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(a), cut),
                                      utils.apply_wdm_cut(np.asarray(b), cut))["mismatch"]

        tables = {}
        for L, P in TABLES:
            ft = wdm.build_time_delay_filter_interpolants(L, P)
            tables[L, P] = (ft, build_rate_tables(wdm, L, P))

        cases = []
        j = nf_index[Nf]
        for i, (name, params) in enumerate(zip(fig8.TONES, MONOCHROMATIC_PARAMS)):
            if name in ("3 mHz", "30 mHz"):
                cases.append((f"{name} on", dict(params, f=freqs["on"][i, j])))
        cases.append(("EMRI", None))

        idx = jnp.asarray([0])
        print(f"--- Nf = {Nf}, dT = {wdm.dT:.4g} s; columns (L, P)", flush=True)
        for label, params in cases:
            theta, phi = ((GC_THETA, GC_PHI) if params is not None
                          else (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"]))
            _, _, tau, n_hat = rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi),
                response.light_travel_time[idx], response.link_direction[idx])
            delta = np.asarray(rs.compute_time_shifts(
                n_hat, response.receiver_position[idx], tau)[0, 0])
            if params is None:
                h = emri_p
                h_spline = None
            else:
                _, h, _ = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=n_signal, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
            _, h, _ = utils.apply_zero_padding(signal_times, h, h,
                                               info["pad_left"], info["pad_right"])
            h = np.asarray(h)
            if params is None:
                h_spline = make_interp_spline(times, h, k=5)
            w = jnp.asarray(dwt(h))

            for s in SCALES:
                d = s*delta
                warped = times + make_interp_spline(t_rows, d, k=5)(times)
                if params is None:
                    h_w = h_spline(warped)
                else:
                    h_w = tukey_continuous(warped, start, n_signal, alpha)*\
                        waveforms.generate_waveform_monochromatic(
                            warped, params["A_plus"], params["A_cross"], params["f"],
                            params["phi_0"])[0]
                exact = dwt(h_w)
                dj, rate = jnp.asarray(d), jnp.gradient(jnp.asarray(d), wdm.dT)
                out = []
                for key, (ft, rt) in tables.items():
                    prod = wdm.apply_variable_time_shift(ft, w, dj)
                    corr = apply_rate_corrected_shift(wdm, ft, rt, w, dj, rate)
                    out.append(f"{key}: {mm(prod, exact):.1e}/{mm(corr, exact):.1e}")
                print(f"{label:>10} s={s:<5} |d|max {np.abs(d).max():6.1f} s | "
                      + "  ".join(out), flush=True)
        del response, wdm, tables
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_signal = 2**19 if "--short" in sys.argv else N_SIGNAL
    main([int(a) for a in args] or [250, 2500], n_signal)
