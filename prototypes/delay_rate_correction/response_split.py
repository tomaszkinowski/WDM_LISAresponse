r"""
Split the delay: the common Roemer delay on the waveform, only the
per-link remainder in the wavelet shift.

Almost all of the per-row delay :math:`\delta_n=\hat n\cdot x_r/c-\tau_n/2`
is the annual Roemer delay of the constellation as a whole. Here

.. math::

    \delta_R(t) = \hat n\cdot x_c(t)/c ,

with :math:`x_c` the mean of the three spacecraft positions, is applied to the
waveform itself - the tones evaluated analytically at :math:`t+\delta_R(t)`,
the EMRI splined (k = 5) - and the wavelet response then shifts by
:math:`\delta_n-\delta_R(t_n)` only (`response_corrections.responses` with
``delay_offset``). Everything is compared, as in `response_full.py`, with the unchanged
time-domain reference of the unwarped waveform.

Columns: ``prod`` / ``full_sinc`` without the split, then ``split`` (the
production shift on the remainder), ``split+1st`` and ``split+full_sinc``.

    python prototypes/delay_rate_correction/response_split.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

jax.config.update("jax_enable_x64", True)

from breakdown_offbin import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                            MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                            NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS,
                            LISAresponse, WDM, fig6, rs, tukey_continuous, utils,
                            waveforms)
from response_corrections import build_tables, responses


def main(nfs, n_signal=N_SIGNAL):
    orbits, start, _, link = fig6.setup()
    signal_times = start + DT*np.arange(n_signal)
    freqs = dict(zip(("on", "off"), fig6.bound_frequencies(fig6.NF)))
    nf_index = {int(nf): j for j, nf in enumerate(fig6.NF)}
    alpha = WINDOW_KWARGS["alpha"]
    td_gen = waveforms.emri_generator()
    _, emri_p, emri_c = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=n_signal,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen

    for Nf in nfs:
        tick = time.perf_counter()
        info = utils.wdm_padding_info(n_signal, Nf, max_lag_L=MAX_LAG_L,
                                      taper_samples=waveforms.tukey_taper_samples(n_signal, alpha))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        tables = build_tables(wdm, MAX_LAG_L, NUM_INTERP_POINTS)
        cut = utils.wdm_cut_info(info)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        x_c = np.mean(orbits.compute_position(t_rows, sc=[1, 2, 3]), axis=1)

        def mm(a, b):
            return utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(a), cut),
                                      utils.apply_wdm_cut(np.asarray(b), cut))["mismatch"]

        j = nf_index.get(Nf)
        cases = []
        for i, (name, params) in enumerate(zip(fig6.TONES, MONOCHROMATIC_PARAMS)):
            if name == "0.3 mHz":
                continue
            for which in ("on", "off"):
                f = freqs[which][i, j] if j is not None else np.nan
                if not np.isnan(f):
                    cases.append((f"{name} {which}", dict(params, f=f), (GC_THETA, GC_PHI)))
        cases.append(("EMRI", None, (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])))

        print(f"--- Nf = {Nf}, dT = {wdm.dT:.4g} s", flush=True)
        for label, params, (theta, phi) in cases:
            if params is None:
                hp, hc = emri_p, emri_c
            else:
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=n_signal, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
            _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                 info["pad_left"], info["pad_right"])
            hp, hc = np.asarray(hp), np.asarray(hc)

            n_hat = np.asarray(rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi),
                response.light_travel_time[jnp.asarray([0])],
                response.link_direction[jnp.asarray([0])])[3])[0]
            roemer_rows = x_c @ n_hat/rs.c
            roemer = make_interp_spline(t_rows, roemer_rows, k=5)(times)
            warped = times + roemer
            if params is None:
                hp_w = make_interp_spline(times, hp, k=5)(warped)
                hc_w = make_interp_spline(times, hc, k=5)(warped)
            else:
                win = tukey_continuous(warped, start, n_signal, alpha)
                hp_w, hc_w = waveforms.generate_waveform_monochromatic(
                    warped, params["A_plus"], params["A_cross"], params["f"],
                    params["phi_0"])
                hp_w, hc_w = win*hp_w, win*hc_w

            plain = responses(response, wdm, tables, theta, phi,
                              wdm.dwt(hp), wdm.dwt(hc))
            split = responses(response, wdm, tables, theta, phi,
                              wdm.dwt(hp_w), wdm.dwt(hc_w), delay_offset=roemer_rows)
            ref = np.asarray(wdm.dwt(np.asarray(response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS))))
            delta = np.asarray(rs.compute_time_shifts(
                jnp.asarray(n_hat)[None], response.receiver_position[jnp.asarray([0])],
                rs.compute_angular_geometry_wdm(
                    jnp.atleast_1d(theta), jnp.atleast_1d(phi),
                    response.light_travel_time[jnp.asarray([0])],
                    response.link_direction[jnp.asarray([0])])[2]))[0, 0]
            rest = delta - roemer_rows
            print(f"{label:>11} | prod {mm(plain['production'], ref):.1e} "
                  f"full_sinc {mm(plain['full_sinc'], ref):.1e} | split {mm(split['production'], ref):.1e} "
                  f"split+1st {mm(split['first'], ref):.1e} "
                  f"split+full_sinc {mm(split['full_sinc'], ref):.1e} "
                  f"(rest {rest.min():.1f}..{rest.max():.1f} s)", flush=True)
        del response, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_signal = 2**19 if "--short" in sys.argv else N_SIGNAL
    main([int(a) for a in args] or [250, 1406, 2500, 7906, 25000], n_signal)
