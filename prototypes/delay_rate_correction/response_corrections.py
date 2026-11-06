r"""
The one-arm wavelet response with the prototype corrections, for one link.

Three variants from the same geometry and coefficients:

- ``production``  the response as `one_arm_response_wdm` forms it;
- ``first``       the delay-rate term to first order (`rate_kernel.
  apply_rate_corrected_shift`);
- ``full``        the amplitude-rate term on the unshifted grid
  (`amp_sinc_kernel`), then the shift to second order in the delay's rate and first
  in its curvature (`rate_kernel.apply_corrected_shift_2`);
- ``full_sinc``   ``full`` plus the window-average sinc to first order about
  each band centre (`amp_sinc_kernel.apply_sinc_rate`);
- ``amp_sinc_first``  the amplitude-rate and sinc terms with the first-order
  delay-rate shift: ``split_amp_sinc_first`` without the split.

`split_responses` gives the split-delay variants: the waveform already carries
the common Roemer delay, the plain production shift only the per-link rest,
with ``split`` (nothing else), ``split_amp`` (plus the amplitude-rate term),
``split_amp_sinc`` (plus the sinc term too) and ``split_amp_sinc_first`` (with
the first-order delay-rate kernel for the rest's own, small, rate).

The rates are central differences across rows of the response's own
:math:`\delta_n` and :math:`A_{nm}`, the curvature the difference of the rate.
:math:`A_{nm}` is taken at :math:`t_n - \delta_n` with the delay the shift
carries, the per-link rest in the split variants.
"""
import numpy as np
import jax.numpy as jnp

from amp_sinc_kernel import (amp_weights, apply_amplitude_rate, apply_sinc_rate,
                        build_amp_table, build_sinc_table, sinc_slope)
from rate_kernel import (apply_corrected_shift_2, apply_rate_corrected_shift,
                         build_rate_tables, build_rate_tables_2)

VARIANTS = ("production", "first", "full", "full_sinc", "amp_sinc_first")
SPLIT_VARIANTS = ("split", "split_amp", "split_amp_sinc", "split_amp_sinc_first")

# Lag range of the amplitude-rate sum. Z falls to ~0.1 of its peak by l = 10;
# truncating there adds <= 1e-9 mismatch (check_amplitude.py at L_Z = 10, 15,
# 25), below the 6.8e-9 on-bin floor, for 21 lags instead of 51.
AMP_LAG_L = 10

# Lag range of the first-order sinc sum: T1 falls to 3e-2 of its peak by l = 10.
SINC_LAG_L = 10


def build_tables(wdm, max_lag_L, num_interp_points):
    return dict(first=build_rate_tables(wdm, max_lag_L, num_interp_points),
                second=build_rate_tables_2(wdm, max_lag_L, num_interp_points),
                amp=amp_weights(build_amp_table(wdm, AMP_LAG_L)),
                sinc=amp_weights(build_sinc_table(wdm, SINC_LAG_L)))


def _grids(response, wdm, tables, theta, phi, wp, wc, link_index, delay_offset):
    """The per-row delay, the plain unshifted grid and its two corrections."""
    from WDM_LISAresponse.code.lisa_response import response_sub as rs

    idx = jnp.asarray([link_index])
    wp, wc = jnp.asarray(wp), jnp.asarray(wc)
    ltt = response.light_travel_time[idx]
    proj_p, proj_c, tau, n_hat = rs.compute_angular_geometry_wdm(
        jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt, response.link_direction[idx])
    delta = rs.compute_time_shifts(n_hat, response.receiver_position[idx], tau)[0, 0]
    if delay_offset is not None:
        delta = delta - jnp.asarray(delay_offset)
    # the detector factors where each row lands, as in `one_arm_response_wdm`
    ltt, proj_p, proj_c, tau = rs.evaluate_at_landing_times(
        jnp.stack(jnp.broadcast_arrays(ltt, proj_p, proj_c, tau)), delta, wdm.dT)
    amp_p, amp_c = rs.compute_detector_amplitudes(ltt, proj_p, proj_c, tau,
                                                  response.wavelet_freqs)
    amp_p, amp_c = amp_p[0, 0], amp_c[0, 0]
    unshifted = amp_p*wp + amp_c*wc
    amp_term = (jnp.gradient(amp_p, wdm.dT, axis=0)
                * apply_amplitude_rate(wdm, tables["amp"], wp)
                + jnp.gradient(amp_c, wdm.dT, axis=0)
                * apply_amplitude_rate(wdm, tables["amp"], wc))
    # the window-average sinc to first order about each band centre, with the
    # antenna amplitude 0.5 ltt F (the sinc divided back out) applied after
    base = 0.5*ltt[0, 0][:, jnp.newaxis]
    slope = sinc_slope(tau[0, 0], response.wavelet_freqs, wdm.dF)
    sinc_term = base*(proj_p[0, 0][:, jnp.newaxis]
                      * apply_sinc_rate(wdm, tables["sinc"], wp, slope)
                      + proj_c[0, 0][:, jnp.newaxis]
                      * apply_sinc_rate(wdm, tables["sinc"], wc, slope))
    return delta, unshifted, amp_term, sinc_term


def split_responses(response, wdm, tables, theta, phi, wp_warped, wc_warped,
                    roemer_rows, link_index=0):
    r"""
    The split-delay variants, each an (Nt, Nf) numpy array.

    ``wp_warped``, ``wc_warped`` are the transforms of the waveform evaluated at
    :math:`t+\delta_R(t)`; ``roemer_rows`` is :math:`\delta_R(t_n)`, taken off
    the per-row delay so the plain shift carries only the per-link rest.
    """
    delta, unshifted, amp_term, sinc_term = _grids(
        response, wdm, tables, theta, phi, wp_warped, wc_warped, link_index,
        roemer_rows)
    shift = lambda grid: np.asarray(
        wdm.apply_variable_time_shift(response.filter_tables, grid, delta))
    full = unshifted + amp_term + sinc_term
    first = np.asarray(apply_rate_corrected_shift(
        wdm, response.filter_tables, tables["first"], full, delta,
        jnp.gradient(delta, wdm.dT)))
    return {"split": shift(unshifted),
            "split_amp": shift(unshifted + amp_term),
            "split_amp_sinc": shift(full),
            "split_amp_sinc_first": first}


def responses(response, wdm, tables, theta, phi, wp, wc, link_index=0,
              delay_offset=None, kinds=VARIANTS):
    """
    The variants named in ``kinds``, each an (Nt, Nf) numpy array.

    ``delay_offset`` (shape (Nt,), seconds) is subtracted from the per-row
    delay before any shift: pass the part of the delay already applied to the
    waveform (`response_split.py`), so the shifts carry only the rest.
    """
    delta, unshifted, amp_term, sinc_term = _grids(
        response, wdm, tables, theta, phi, wp, wc, link_index, delay_offset)
    rate = jnp.gradient(delta, wdm.dT)
    curvature = jnp.gradient(rate, wdm.dT)
    filters = response.filter_tables
    build = {
        "production": lambda: wdm.apply_variable_time_shift(filters, unshifted, delta),
        "first": lambda: apply_rate_corrected_shift(
            wdm, filters, tables["first"], unshifted, delta, rate),
        "full": lambda: apply_corrected_shift_2(
            wdm, filters, tables["second"], unshifted + amp_term, delta, rate,
            curvature),
        "full_sinc": lambda: apply_corrected_shift_2(
            wdm, filters, tables["second"], unshifted + amp_term + sinc_term, delta,
            rate, curvature),
        "amp_sinc_first": lambda: apply_rate_corrected_shift(
            wdm, filters, tables["first"], unshifted + amp_term + sinc_term, delta,
            rate),
    }
    return {k: np.asarray(build[k]()) for k in kinds}
