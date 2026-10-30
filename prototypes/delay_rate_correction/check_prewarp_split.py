r"""
The amplitude pre-warp on top of the split delay.

The waveform carries the common Roemer delay :math:`\delta_R` (as in
`response_split.py`) and the shift only the rest,
:math:`\delta_n-\delta_R(t_n)`, a few seconds. The pre-warp then evaluates the
row geometry at :math:`t_n-(\delta_n-\delta_R(t_n))`. Expected: the timing
term scales as the shifted delay squared, so about :math:`10^{-12}`.

Columns, each without and with the pre-warp (``_pw``), against the
time-domain reference: ``split`` (production shift on the rest) and
``split_all`` (amplitude-rate, sinc and delay-rate corrections,
`check_prewarp.variants`). ``_fix`` replaces the rest by the self-consistent
:math:`r_n=\delta_n-\delta_R(t_n+r_n)`: the shift moves the already-warped
waveform, which picks up :math:`\delta_R` at :math:`t+r`, not at :math:`t`.

    python prototypes/delay_rate_correction/check_prewarp_split.py [Nf ...] [--short] [--lag=L]
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
from check_prewarp import variants
from response_corrections import build_tables


def main(nfs, n_signal=N_SIGNAL, max_lag_L=MAX_LAG_L):
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
        info = utils.wdm_padding_info(n_signal, Nf, max_lag_L=max_lag_L,
                                      taper_samples=waveforms.tukey_taper_samples(n_signal, alpha))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=max_lag_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        tables = build_tables(wdm, max_lag_L, NUM_INTERP_POINTS)
        cut = utils.wdm_cut_info(info)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        x_c = np.mean(orbits.compute_position(t_rows, sc=[1, 2, 3]), axis=1)
        spline_rows = lambda y: make_interp_spline(t_rows, np.asarray(y), k=5)

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
        idx = jnp.asarray([0])
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

            ltt = response.light_travel_time[idx]
            pp, pc, tau, n_hat = rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt, response.link_direction[idx])
            delta = np.asarray(rs.compute_time_shifts(
                n_hat, response.receiver_position[idx], tau)[0, 0])
            ltt, pp, pc, tau = ltt[0], pp[0, 0], pc[0, 0], tau[0, 0]

            roemer_rows = x_c @ np.asarray(n_hat)[0]/rs.c
            warped = times + make_interp_spline(t_rows, roemer_rows, k=5)(times)
            if params is None:
                hp_w = make_interp_spline(times, hp, k=5)(warped)
                hc_w = make_interp_spline(times, hc, k=5)(warped)
            else:
                win = tukey_continuous(warped, start, n_signal, alpha)
                hp_w, hc_w = waveforms.generate_waveform_monochromatic(
                    warped, params["A_plus"], params["A_cross"], params["f"],
                    params["phi_0"])
                hp_w, hc_w = win*hp_w, win*hc_w
            wp, wc = wdm.dwt(hp_w), wdm.dwt(hc_w)

            rest = delta - roemer_rows
            # shifting h(t + delta_R(t)) by the rest gives h(t + rest + delta_R(t + rest)),
            # so the rest that lands on delta solves rest = delta - delta_R(t + rest)
            roemer_of = make_interp_spline(t_rows, roemer_rows, k=5)
            rest_fix = rest
            for _ in range(3):
                rest_fix = delta - roemer_of(t_rows + rest_fix)
            landing = t_rows - rest
            geometry_pw = [jnp.asarray(spline_rows(y)(landing)) for y in (ltt, pp, pc, tau)]
            rest_j = jnp.asarray(rest)

            ref = np.asarray(wdm.dwt(np.asarray(response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS))))
            split, split_all = variants(response, wdm, tables, wp, wc,
                                        ltt, pp, pc, tau, rest_j)
            split_pw, split_all_pw = variants(response, wdm, tables, wp, wc,
                                              *geometry_pw, rest_j)
            split_fix, split_all_fix = variants(response, wdm, tables, wp, wc,
                                                ltt, pp, pc, tau, jnp.asarray(rest_fix))
            print(f"{label:>11} | split {mm(split, ref):.1e} split_pw {mm(split_pw, ref):.1e} "
                  f"split_fix {mm(split_fix, ref):.1e} | "
                  f"split_all {mm(split_all, ref):.1e} split_all_pw {mm(split_all_pw, ref):.1e} "
                  f"split_all_fix {mm(split_all_fix, ref):.1e} "
                  f"(rest {rest.min():.1f}..{rest.max():.1f} s, "
                  f"fix moves it by up to {np.abs(rest_fix - rest).max():.1e} s)", flush=True)
        del response, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_signal = 2**19 if "--short" in sys.argv else N_SIGNAL
    lag = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--lag=")]
    main([int(a) for a in args] or [250, 2500], n_signal, *lag)
