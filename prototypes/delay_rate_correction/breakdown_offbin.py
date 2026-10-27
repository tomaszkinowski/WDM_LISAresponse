r"""
What limits the corrected response off-bin at coarse Delta T.

At each tiling, for the figure-5 tones moved to their on-bin and off-bin
frequencies (fig5.bound_frequencies) and for the EMRI, the corrected wavelet
response's error against the time-domain reference is split into pieces, each
measured on its own against an exact counterpart:

- ``shift``  the corrected shift alone: `apply_rate_corrected_shift` applied to
  W(A h), the transform of the continuous unshifted model, against
  W((A h)(t + delta(t))), its exact warp. Tones are evaluated analytically at
  the warped times; the EMRI is splined (k = 5, <= 3 mHz at dt = 4 s).
- ``amp``    the per-pixel amplitude product: A_n (.) W(h), the production
  unshifted grid with the sinc at the tone's own frequency, against W(A(t) h),
  A(t) splined from the row values. The EMRI drops the sinc from both.
- ``sinc``   the sinc at the band centre f_m against the tone's frequency, both
  per pixel: the local-delay approximation.
- ``timing`` (A h)(t + delta) against A(t) h(t + delta): the amplitude moving
  with the shift.
- ``corr`` / ``prod``  the corrected and production responses against the
  time-domain reference, as in figure 5; ``corr+20`` with 20 more rows cut at
  each end, for the record edges.

delta(t) is splined (k = 5) from the response's own row delays; the rate is
their central difference, as in the corrected response.

    python prototypes/delay_rate_correction/breakdown_offbin.py [Nf ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

jax.config.update("jax_enable_x64", True)

RESPONSE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESPONSE_ROOT / "plotting"))

import fig5_mismatch_vs_dT as fig5
from parameters import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

import WDM
from WDM_LISAresponse import LISAresponse
from WDM_LISAresponse.code.lisa_response import response_sub as rs
from WDM_LISAresponse.code.utils import utils
from WDM_LISAresponse.code.waveforms import waveforms

from rate_kernel import apply_rate_corrected_shift, build_rate_tables

NFS = (444, 1406, 7906, 14058, 25000)
EXTRA_ROWS = 20


def tukey_continuous(t, t_start, n_samples, alpha):
    """scipy's Tukey window on t_start + k dt, at arbitrary times; zero outside."""
    u = (t - t_start)/((n_samples - 1)*DT)
    ramp = 0.5*alpha
    w = np.ones_like(u)
    lo, hi = u < ramp, u > 1 - ramp
    w[lo] = 0.5*(1 + np.cos(np.pi*(u[lo]/ramp - 1)))
    w[hi] = 0.5*(1 + np.cos(np.pi*((u[hi] - 1)/ramp + 1)))
    w[(u < 0) | (u > 1)] = 0.0
    return w


def main(nfs):
    orbits, start, signal_times, link = fig5.setup()
    freqs = dict(zip(("on", "off"), fig5.bound_frequencies(fig5.NF)))
    nf_index = {int(nf): j for j, nf in enumerate(fig5.NF)}

    td_gen = waveforms.emri_generator()
    _, emri_p, emri_c = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen

    alpha = WINDOW_KWARGS["alpha"]
    for Nf in nfs:
        tick = time.perf_counter()
        info = utils.wdm_padding_info(
            N_SIGNAL, Nf, max_lag_L=MAX_LAG_L,
            taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, alpha))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=MAX_LAG_L,
                                num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        rate_tables = build_rate_tables(wdm, MAX_LAG_L, NUM_INTERP_POINTS)
        cut = utils.wdm_cut_info(info)
        cut_more = utils.wdm_cut_info(info, extra_rows=EXTRA_ROWS)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        t_start = start
        dwt = lambda x: np.asarray(wdm.dwt(jnp.asarray(x)))
        spline_rows = lambda y: make_interp_spline(t_rows, np.asarray(y), k=5)

        def mm(a, b, c=cut):
            return utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(a), c),
                                      utils.apply_wdm_cut(np.asarray(b), c))["mismatch"]

        cases = []
        j = nf_index[Nf]
        for i, (name, params) in enumerate(zip(fig5.TONES, MONOCHROMATIC_PARAMS)):
            if name == "0.3 mHz":
                continue
            for which in ("on", "off"):
                f = freqs[which][i, j]
                if not np.isnan(f):
                    cases.append((f"{name} {which}", dict(params, f=f),
                                  (GC_THETA, GC_PHI)))
        cases.append(("EMRI", None, (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])))

        print(f"--- Nf = {Nf}, dT = {wdm.dT:.4g} s, Nt = {wdm.Nt}, rows kept "
              f"{cut['time_slice'].start}..{cut['time_slice'].stop}", flush=True)
        idx = jnp.asarray([0])
        for label, params, (theta, phi) in cases:
            if params is None:
                hp, hc = emri_p, emri_c
            else:
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
                    model_kwargs=params, window_function=WINDOW_FUNCTION,
                    window_kwargs=WINDOW_KWARGS)
            _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                 info["pad_left"], info["pad_right"])
            hp, hc = np.asarray(hp), np.asarray(hc)
            wp, wc = wdm.dwt(hp), wdm.dwt(hc)

            ltt = response.light_travel_time[idx]
            pp, pc, tau, n_hat = rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt,
                response.link_direction[idx])
            delta = rs.compute_time_shifts(n_hat, response.receiver_position[idx], tau)
            amp_p, amp_c = rs.compute_detector_amplitudes(ltt, pp, pc, tau,
                                                          response.wavelet_freqs)
            unshifted = np.asarray(amp_p*wp[None, None] + amp_c*wc[None, None])[0, 0]
            d = np.asarray(delta[0, 0])
            rate = np.gradient(d, float(wdm.dT))
            prod = np.asarray(wdm.apply_variable_time_shift(
                response.filter_tables, jnp.asarray(unshifted), jnp.asarray(d)))
            shift = lambda w: np.asarray(apply_rate_corrected_shift(
                wdm, response.filter_tables, rate_tables, jnp.asarray(w),
                jnp.asarray(d), jnp.asarray(rate)))
            corr = shift(unshifted)
            ref = dwt(np.asarray(response.one_arm_response_time_domain(
                theta, phi, hp, hc, link=link,
                num_integral_points=NUM_INTEGRAL_POINTS)))

            # continuous amplitudes, splined from the rows
            f = None if params is None else params["f"]
            base_p = 0.5*np.asarray(ltt[0])*np.asarray(pp[0, 0])
            base_c = 0.5*np.asarray(ltt[0])*np.asarray(pc[0, 0])
            sinc = 1.0 if f is None else np.sinc(np.asarray(tau[0, 0])*f)
            Ap, Ac = spline_rows(base_p*sinc), spline_rows(base_c*sinc)
            d_t = spline_rows(d)(times)
            warped = times + d_t

            # per-pixel products, with the sinc at the true frequency (tones)
            # or without it (EMRI), so only the time variation differs
            per_pixel = (base_p*sinc)[:, None]*np.asarray(wp) \
                      + (base_c*sinc)[:, None]*np.asarray(wc)
            continuous = dwt(Ap(times)*hp + Ac(times)*hc)

            if params is None:
                hp_w = make_interp_spline(times, hp, k=5)(warped)
                hc_w = make_interp_spline(times, hc, k=5)(warped)
            else:
                win = tukey_continuous(warped, t_start, N_SIGNAL, alpha)
                hp_w, hc_w = waveforms.generate_waveform_monochromatic(
                    warped, params["A_plus"], params["A_cross"], params["f"],
                    params["phi_0"])
                hp_w, hc_w = win*hp_w, win*hc_w
            moved = dwt(Ap(warped)*hp_w + Ac(warped)*hc_w)        # (A h)(t+delta)
            at_reception = dwt(Ap(times)*hp_w + Ac(times)*hc_w)   # A(t) h(t+delta)

            row = dict(
                prod=mm(prod, ref), corr=mm(corr, ref),
                corr_more=mm(corr, ref, cut_more),
                shift=mm(shift(continuous), moved),
                amp=mm(per_pixel, continuous),
                sinc=float("nan") if f is None else mm(unshifted, per_pixel),
                timing=mm(moved, at_reception),
                model=mm(at_reception, ref))
            print(f"{label:>14} | prod {row['prod']:.1e} corr {row['corr']:.1e} "
                  f"corr+{EXTRA_ROWS} {row['corr_more']:.1e} | shift {row['shift']:.1e} "
                  f"amp {row['amp']:.1e} sinc {row['sinc']:.1e} "
                  f"timing {row['timing']:.1e} model {row['model']:.1e}", flush=True)
        del response, wdm
        jax.clear_caches()
        print(f"    ({time.perf_counter() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or list(NFS))
