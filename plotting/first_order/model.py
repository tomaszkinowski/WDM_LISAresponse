"""The first-order wavelet response as the figure scripts call it: fast_split's jitted
links, with the waveform evaluated at t + delta_R(t), delta_R the Roemer delay of the
constellation centre - tones analytically (tone_warp), the EMRI by quintic spline
(spline_warp). Every correction table takes the lag range and interpolation points
of the response's own filter tables.
"""

import numpy as np
import jax.numpy as jnp

from parameters import DT, WINDOW_KWARGS

from . import fast_split as fs
from .tables import amp_weights, build_amp_table, build_rate_tables, build_sinc_table


def tukey_continuous(t, t_start, n_samples, alpha):
    """scipy's Tukey window on t_start + k dt, at arbitrary times; zero outside."""
    u = (t - t_start) / ((n_samples - 1) * DT)
    ramp = 0.5 * alpha
    w = np.ones_like(u)
    lo, hi = u < ramp, u > 1 - ramp
    w[lo] = 0.5 * (1 + np.cos(np.pi * (u[lo] / ramp - 1)))
    w[hi] = 0.5 * (1 + np.cos(np.pi * ((u[hi] - 1) / ramp + 1)))
    w[(u < 0) | (u > 1)] = 0.0
    return w


def tone_warp(params, start, n_samples):
    """The windowed tone of `n_samples` from `start` at arbitrary times, analytically."""
    from WDM_LISAresponse.code.waveforms import waveforms

    def warp(times):
        window = tukey_continuous(times, start, n_samples, WINDOW_KWARGS["alpha"])
        hp, hc = waveforms.generate_waveform_monochromatic(
            times, params["A_plus"], params["A_cross"], params["f"], params["phi_0"])
        return window * hp, window * hc
    return warp


def spline_warp(times, h_plus, h_cross):
    """A sampled waveform at arbitrary times, by quintic spline (EMRI, <= mHz)."""
    from scipy.interpolate import make_interp_spline

    sp = make_interp_spline(times, h_plus, k=5)
    sc = make_interp_spline(times, h_cross, k=5)
    return lambda t: (sp(t), sc(t))


def constants(response, wdm):
    """fast_split's source-independent arrays for this response."""
    max_lag_L = (response.filter_tables.shape[1] - 1) // 2
    num_points = response.filter_tables.shape[-1]
    tables = dict(first=build_rate_tables(wdm, max_lag_L, num_points),
                  amp=amp_weights(build_amp_table(wdm, max_lag_L)),
                  sinc=amp_weights(build_sinc_table(wdm, max_lag_L)))
    centre = np.mean(response.orbits_class.compute_position(
        np.asarray(response.wavelet_times), sc=[1, 2, 3]), axis=1)
    return fs.setup(response, tables, centre)


def warped_transforms(response, wdm, consts, theta, phi, warp):
    """The transforms of the waveform at t + delta_R(t) on the response's grid;
    ``warp(times)`` returns h+ and hx at those times."""
    from scipy.interpolate import make_interp_spline

    roemer = np.asarray(fs.roemer_rows(theta, phi, consts["centre"]))
    times = np.asarray(response.t0 + wdm.times)
    h_plus, h_cross = warp(times + make_interp_spline(
        np.asarray(response.wavelet_times), roemer, k=5)(times))
    return wdm.dwt(h_plus), wdm.dwt(h_cross)


def links(wdm, consts, theta, phi, wp, wc, dtype=jnp.float64):
    """Every link of the response, (num_links, Nt, Nf), from warped_transforms.
    dtype is the working precision of the shared sums, grids, weights and lag
    loop (the figures use float64; float32 is the faster one timed in
    numbers_runtime.py); geometry, delays and the carrier phase stay float64, and
    the result is float64."""
    if dtype == jnp.float64:
        rows, sums = fs.stage_common(wdm, consts, theta, phi, wp, wc)
    else:
        rows, sums = fs.stage_common_typed(wdm, consts, theta, phi, dtype, wp, wc)
    # the shift's weights read only the table nodes a rest delay within REST_BOUND touches
    worst = float(np.max(np.abs(np.asarray(rows[0]))))
    assert worst < fs.REST_BOUND, f"rest delay {worst:.1f} s beyond REST_BOUND"
    if dtype == jnp.float64:
        grids = fs.stage_grids(wdm, consts, rows, wp, wc, sums)
    else:
        grids = fs.stage_grids_typed(wdm, consts, rows, wp, wc, sums, dtype)
    weights = fs.stage_weights_vec(wdm, consts, rows,
                                   fs.table_nodes(wdm, consts["filters"].shape[-1]))
    return np.asarray(fs.stage_wshifts(wdm, rows, grids, weights), dtype=np.float64)


def one_link(response, wdm, link, start, n_signal):
    """The sweeps' model interface: respond(theta, phi, h_plus, h_cross, tone=None),
    the first-order response on `link` to a padded waveform of `n_signal` samples
    from `start`. `tone`, a tone's parameters, warps it analytically; otherwise by
    spline."""
    consts = constants(response, wdm)
    index = response.link_convention.index(link)
    times = np.asarray(response.t0 + wdm.times)

    def respond(theta, phi, h_plus, h_cross, tone=None):
        warp = (tone_warp(tone, start, n_signal) if tone is not None
                else spline_warp(times, np.asarray(h_plus), np.asarray(h_cross)))
        wp, wc = warped_transforms(response, wdm, consts, theta, phi, warp)
        return links(wdm, consts, theta, phi, wp, wc)[index]
    return respond
