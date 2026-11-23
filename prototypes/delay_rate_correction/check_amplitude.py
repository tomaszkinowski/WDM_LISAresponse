r"""
Check the amplitude-rate term.

1. `apply_amplitude_rate` against the explicit sum over `elements_amplitude`
   closed-form elements, on the small test grid.
2. On the `breakdown_offbin.py` tilings, the per-pixel product with and without the term,
   against the transform of the continuous product W(A(t) h): the ``amp``
   piece of `breakdown_offbin.py`. No time-domain references are needed.

    python prototypes/delay_rate_correction/check_amplitude.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

jax.config.update("jax_enable_x64", True)

import breakdown_offbin as d
from breakdown_offbin import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                            MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTERP_POINTS,
                            WINDOW_FUNCTION, WINDOW_KWARGS, LISAresponse, WDM,
                            fig14, rs, utils, waveforms)
from amp_sinc_kernel import amp_weights, apply_amplitude_rate, build_amp_table
from elements_amplitude import closed_Z


def check_small():
    wdm = WDM.WDM.WDM_transform(dt=0.5, Nf=8, N=8*32)
    L = 4
    w = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (wdm.Nt, wdm.Nf)))
    fast = np.asarray(apply_amplitude_rate(wdm, amp_weights(build_amp_table(wdm, L)),
                                           jnp.asarray(w)))
    worst = 0.0
    for n in range(8, 24):
        for m in range(2, wdm.Nf - 2):
            ref = sum(w[n - l, m + sigma]*closed_Z(wdm, n, m, l, sigma, -1j/(2*np.pi))
                      for l in range(-L, L + 1) for sigma in (-1, 0, 1))
            worst = max(worst, abs(fast[n, m] - ref))
    print(f"kernel vs explicit closed-form sum: max abs {worst:.1e} "
          f"(|sum| ~ {np.max(np.abs(fast[8:24, 2:-2])):.1f})", flush=True)


def main(nfs):
    check_small()
    orbits, start, signal_times, link = fig14.setup()
    freqs = fig14.bound_frequencies(fig14.NF)[1]
    nf_index = {int(nf): j for j, nf in enumerate(fig14.NF)}
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
        weights = amp_weights(build_amp_table(wdm, MAX_LAG_L))
        cut = utils.wdm_cut_info(info)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        dwt = lambda x: np.asarray(wdm.dwt(jnp.asarray(x)))
        mm = lambda a, b: utils.wdm_mismatch(utils.apply_wdm_cut(np.asarray(a), cut),
                                             utils.apply_wdm_cut(np.asarray(b), cut))["mismatch"]

        j = nf_index[Nf]
        cases = [(f"{name} off", dict(p, f=freqs[i, j]), (GC_THETA, GC_PHI))
                 for i, (name, p) in enumerate(zip(fig14.TONES, MONOCHROMATIC_PARAMS))
                 if name != "0.3 mHz" and not np.isnan(freqs[i, j])]
        cases.append(("EMRI", None, (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])))
        idx = jnp.asarray([0])
        line = []
        for label, params, (theta, phi) in cases:
            if params is None:
                hp, hc = emri_p, emri_c
            else:
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
            _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                 info["pad_left"], info["pad_right"])
            hp, hc = np.asarray(hp), np.asarray(hc)
            wp, wc = np.asarray(wdm.dwt(hp)), np.asarray(wdm.dwt(hc))
            ltt = response.light_travel_time[idx]
            pp, pc, tau, _ = rs.compute_angular_geometry_wdm(
                jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt,
                response.link_direction[idx])
            sinc = 1.0 if params is None else np.sinc(np.asarray(tau[0, 0])*params["f"])
            Ap_n = 0.5*np.asarray(ltt[0])*np.asarray(pp[0, 0])*sinc
            Ac_n = 0.5*np.asarray(ltt[0])*np.asarray(pc[0, 0])*sinc
            Ap = make_interp_spline(t_rows, Ap_n, k=5)(times)
            Ac = make_interp_spline(t_rows, Ac_n, k=5)(times)

            continuous = dwt(Ap*hp + Ac*hc)
            per_pixel = Ap_n[:, None]*wp + Ac_n[:, None]*wc
            rate_p = np.gradient(Ap_n, float(wdm.dT))[:, None]
            rate_c = np.gradient(Ac_n, float(wdm.dT))[:, None]
            corrected = per_pixel \
                + rate_p*np.asarray(apply_amplitude_rate(wdm, weights, jnp.asarray(wp))) \
                + rate_c*np.asarray(apply_amplitude_rate(wdm, weights, jnp.asarray(wc)))
            line.append(f"{label} {mm(per_pixel, continuous):.1e}->"
                        f"{mm(corrected, continuous):.1e}")
        print(f"Nf={Nf:6d} dT={wdm.dT:8.4g} s | " + "  ".join(line)
              + f"  ({time.perf_counter() - tick:.0f} s)", flush=True)
        del response, wdm
        jax.clear_caches()


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [444, 1406, 2500, 7906, 14058, 25000])
