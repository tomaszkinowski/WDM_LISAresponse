"""The corrected wavelet response (split_amp_sinc_first) for the investigation scripts,
in the jitted every-link form of prototypes/delay_rate_correction/fast_split.py
(float64, interpolated tables).

The waveform goes in already evaluated at t + delta_R(t), delta_R the Roemer delay of
the constellation centre, as figure 9 does it: tones analytically (tone_warp), the
EMRI by quintic spline (spline_warp). The lag range and interpolation points of every
correction table are those of the response's own filter tables.
../prototypes/delay_rate_correction must be on PYTHONPATH.
"""

import numpy as np

import fast_split as fs
from amp_sinc_kernel import amp_weights, build_amp_table, build_sinc_table
from investigation_fig8_delay_rate import spline_warp, tone_warp  # noqa: F401
from rate_kernel import build_rate_tables


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


def links(wdm, consts, theta, phi, wp, wc):
    """Every link of the response, (num_links, Nt, Nf), from warped_transforms."""
    rows, sums = fs.stage_common(wdm, consts, theta, phi, wp, wc)
    # the shift's weights read only the table nodes a rest delay within REST_BOUND touches
    worst = float(np.max(np.abs(np.asarray(rows[0]))))
    assert worst < fs.REST_BOUND, f"rest delay {worst:.1f} s beyond REST_BOUND"
    grids = fs.stage_grids(wdm, consts, rows, wp, wc, sums)
    weights = fs.stage_weights_vec(wdm, consts, rows,
                                   fs.table_nodes(wdm, consts["filters"].shape[-1]))
    return np.asarray(fs.stage_wshifts(wdm, rows, grids, weights))
