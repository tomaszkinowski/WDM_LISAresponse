r"""
Pre-warp the amplitude: evaluate the detector factors at the time each source
row lands at, so the shift carries them to the reception time.

The model is :math:`A(t)\,h(t+\delta(t))`, with :math:`A` (the arm length,
antenna pattern and window-average sinc) a detector quantity at reception.
Production multiplies :math:`A_{n'}` onto the unshifted grid and shifts the
product, which gives :math:`A(t+\delta)\,h(t+\delta)`: the amplitude timing
term. The shift maps source time :math:`t'` to :math:`t'-\delta`, so
multiplying instead by

.. math::

    \tilde A(t') = A\bigl(t' - \delta(t')\bigr)

gives :math:`A\bigl(t+\delta(t)-\delta(t+\delta(t))\bigr)\approx
A(t-\dot\delta\,\delta)` after the shift, with one shift per link as before.
The row geometry (:math:`\ell`, :math:`F^{+,\times}`, :math:`\tau`) is splined
(k = 5) and evaluated at :math:`t_n-\delta_n`.

Columns:

- ``timing`` / ``timing_pw``  (A h)(t + delta), and the same with A pre-warped,
  against A(t) h(t + delta), all continuous (as in `breakdown_offbin.py`);
- ``shift_pw`` / ``shiftc_pw``  the production and the corrected shift alone,
  applied to W(A(t - delta(t)) h), against W of its exact warp;
- ``prod`` / ``prod_pw``, ``full_sinc`` / ``full_sinc_pw``  the wavelet
  responses against the time-domain reference, without and with the pre-warp.

    python prototypes/delay_rate_correction/check_prewarp.py [Nf ...] [--short] [--lag=L]
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
from response_corrections import build_tables
from amp_sinc_kernel import apply_amplitude_rate, apply_sinc_rate, sinc_slope
from rate_kernel import apply_corrected_shift_2


def variants(response, wdm, tables, wp, wc, ltt, proj_p, proj_c, tau, delta):
    """Production and full_sinc from the given row geometry, shape (Nt,)."""
    freqs = response.wavelet_freqs
    base = 0.5*ltt[:, None]
    sinc = jnp.sinc(tau[:, None]*freqs)
    amp_p, amp_c = base*proj_p[:, None]*sinc, base*proj_c[:, None]*sinc
    unshifted = amp_p*wp + amp_c*wc
    amp_term = (jnp.gradient(amp_p, wdm.dT, axis=0)*apply_amplitude_rate(wdm, tables["amp"], wp)
                + jnp.gradient(amp_c, wdm.dT, axis=0)*apply_amplitude_rate(wdm, tables["amp"], wc))
    slope = sinc_slope(tau, freqs, wdm.dF)
    sinc_term = base*(proj_p[:, None]*apply_sinc_rate(wdm, tables["sinc"], wp, slope)
                      + proj_c[:, None]*apply_sinc_rate(wdm, tables["sinc"], wc, slope))
    rate = jnp.gradient(delta, wdm.dT)
    curvature = jnp.gradient(rate, wdm.dT)
    prod = wdm.apply_variable_time_shift(response.filter_tables, unshifted, delta)
    full_sinc = apply_corrected_shift_2(wdm, response.filter_tables, tables["second"],
                                        unshifted + amp_term + sinc_term,
                                        delta, rate, curvature)
    return np.asarray(prod), np.asarray(full_sinc)


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
        dwt = lambda x: np.asarray(wdm.dwt(jnp.asarray(x)))
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

        print(f"--- Nf = {Nf}, dT = {wdm.dT:.4g} s, Nt = {wdm.Nt}", flush=True)
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
            wp, wc = wdm.dwt(hp), wdm.dwt(hc)

            ltt = response.light_travel_time[idx]
            pp, pc, tau, n_hat = rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt, response.link_direction[idx])
            delta = rs.compute_time_shifts(n_hat, response.receiver_position[idx], tau)[0, 0]
            ltt, pp, pc, tau = ltt[0], pp[0, 0], pc[0, 0], tau[0, 0]
            d = np.asarray(delta)

            # the row geometry at the time each source row lands at
            landing = t_rows - d
            ltt_w, pp_w, pc_w, tau_w = (jnp.asarray(spline_rows(y)(landing))
                                        for y in (ltt, pp, pc, tau))

            ref = dwt(np.asarray(response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link, num_integral_points=NUM_INTEGRAL_POINTS)))
            prod, full_sinc = variants(response, wdm, tables, wp, wc, ltt, pp, pc, tau, delta)
            prod_pw, full_sinc_pw = variants(response, wdm, tables, wp, wc,
                                             ltt_w, pp_w, pc_w, tau_w, delta)

            # continuous timing piece, as breakdown_offbin
            f = None if params is None else params["f"]
            base_p = 0.5*np.asarray(ltt)*np.asarray(pp)
            base_c = 0.5*np.asarray(ltt)*np.asarray(pc)
            sinc = 1.0 if f is None else np.sinc(np.asarray(tau)*f)
            Ap, Ac = spline_rows(base_p*sinc), spline_rows(base_c*sinc)
            d_of = spline_rows(d)
            warped = times + d_of(times)
            if params is None:
                hp_w = make_interp_spline(times, hp, k=5)(warped)
                hc_w = make_interp_spline(times, hc, k=5)(warped)
            else:
                win = tukey_continuous(warped, start, n_signal, alpha)
                hp_w, hc_w = waveforms.generate_waveform_monochromatic(
                    warped, params["A_plus"], params["A_cross"], params["f"], params["phi_0"])
                hp_w, hc_w = win*hp_w, win*hc_w
            at_reception = dwt(Ap(times)*hp_w + Ac(times)*hc_w)
            moved = dwt(Ap(warped)*hp_w + Ac(warped)*hc_w)
            landed = warped - d_of(warped)
            moved_pw = dwt(Ap(landed)*hp_w + Ac(landed)*hc_w)

            # the shifts alone, on the transform of the continuous pre-warped A h
            source = times - d_of(times)
            unshifted_pw = jnp.asarray(dwt(Ap(source)*hp + Ac(source)*hc))
            rate = jnp.gradient(delta, wdm.dT)
            shift_pw = wdm.apply_variable_time_shift(response.filter_tables,
                                                     unshifted_pw, delta)
            shiftc_pw = apply_corrected_shift_2(wdm, response.filter_tables,
                                                tables["second"], unshifted_pw, delta,
                                                rate, jnp.gradient(rate, wdm.dT))

            print(f"{label:>11} | timing {mm(moved, at_reception):.1e} "
                  f"timing_pw {mm(moved_pw, at_reception):.1e} | "
                  f"shift_pw {mm(shift_pw, moved_pw):.1e} "
                  f"shiftc_pw {mm(shiftc_pw, moved_pw):.1e} | "
                  f"prod {mm(prod, ref):.1e} prod_pw {mm(prod_pw, ref):.1e} | "
                  f"full_sinc {mm(full_sinc, ref):.1e} full_sinc_pw {mm(full_sinc_pw, ref):.1e}",
                  flush=True)
        del response, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_signal = 2**19 if "--short" in sys.argv else N_SIGNAL
    lag = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--lag=")]
    main([int(a) for a in args] or [250, 444, 1406, 2500], n_signal, *lag)
