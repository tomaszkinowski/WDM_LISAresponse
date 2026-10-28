"""The one-arm wavelet response with the amplitude pre-warp, as a stand-in for
LISAresponse.one_arm_response, for provisional figures only.

Production multiplies the per-row detector factors (arm length, antenna pattern,
window-average sinc) onto the unshifted grid and shifts the product, so each row's
factors arrive at t_n + delta_n instead of the reception time t_n. Here the row
geometry is splined (k = 5) to t_n - delta_n first, so the shift carries it back
to t_n; everything else, the one shift per link included, is production's.

``patch()`` swaps it into LISAresponse for the rest of the process. Single link,
scalar sky, wavelet coefficients in and out - what figures 5, 6 and 7 call.
"""
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

from WDM_LISAresponse import LISAresponse
from WDM_LISAresponse.code.lisa_response import response_sub as rs
from WDM_LISAresponse.code.utils.utils import resolve_link_selection

PRODUCTION = LISAresponse.one_arm_response


def one_arm_response_prewarped(self, theta, phi, h_plus, h_cross, in_domain="wdm",
                               out_domain="wdm", link=None, prewarp=True):
    assert in_domain == "wdm" and out_domain == "wdm", (in_domain, out_domain)
    link_indices, drop_link_axis = resolve_link_selection(link, self.link_convention)
    assert len(link_indices) == 1 and drop_link_axis, link
    idx = jnp.asarray(link_indices)

    ltt = self.light_travel_time[idx]
    proj_p, proj_c, tau, n_hat = rs.compute_angular_geometry_wdm(
        jnp.atleast_1d(theta), jnp.atleast_1d(phi), ltt, self.link_direction[idx])
    delta = rs.compute_time_shifts(n_hat, self.receiver_position[idx], tau)[0, 0]

    rows = np.asarray(self.wavelet_times)
    landing = rows - np.asarray(delta) if prewarp else rows
    def at_landing(y):
        return jnp.asarray(make_interp_spline(rows, np.asarray(y), k=5)(landing))

    amp_p, amp_c = rs.compute_detector_amplitudes(
        at_landing(ltt[0])[None], at_landing(proj_p[0, 0])[None, None],
        at_landing(proj_c[0, 0])[None, None], at_landing(tau[0, 0])[None, None],
        self.wavelet_freqs)
    unshifted = amp_p[0, 0]*jnp.asarray(h_plus) + amp_c[0, 0]*jnp.asarray(h_cross)
    return self.wdm_class.apply_variable_time_shift(self.filter_tables, unshifted, delta)


def patch():
    LISAresponse.one_arm_response = one_arm_response_prewarped
