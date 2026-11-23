"""Figure 12: the EMRI (parameters.py) on link 1 -> 2 at Delta T = 1e4 s, from four
codes - the wavelet one_arm_response, the time-domain reference
one_arm_response_time_domain, lisagwresponse and fastlisaresponse: the whole
record, and a quarter-day zoom with the wavelet-minus-lisagwresponse residual
overlaid (magnified). In figures/investigations/: the same with the source's h+ and
hx over the zoom (fig12_emri_link_signals_waveform.pdf), and the residuals of every
pair over the zoom (fig12_emri_link_residuals.pdf).

The mismatches over the whole cut record, for the caption, are printed and written
to figures/fig12_emri_link_mismatches.txt. The grid, window and cut are figure 8's
at this tiling, so the wavelet-vs-reference mismatch here is figure 8's EMRI point
at 1e4 s. What the four share, and how the external codes are converted, is in
_emri_comparison.py.

Run with no arguments to compute (or reuse the cache) and plot. ``--short`` runs
2**19 samples (24 d), prints the mismatches and draws into figures/investigations/
without caching - for checking the pipeline on the laptop. ``--nf NF`` runs the
whole record at another tiling, cached and drawn as an investigation.
"""

import time

import numpy as np

import _emri_comparison as emri
from parameters import NUM_INTEGRAL_POINTS

LABELS = dict(WDM="WDM response", TD="time-domain reference", GWR="lisagwresponse",
              FLR="fastlisaresponse")


def compute(n_signal, nf):
    from fastlisaresponse import pyResponseTDI
    from lisagwresponse import ReadStrain

    from WDM_LISAresponse.code.comparison import common, conventions_flr, conventions_gwr
    from WDM_LISAresponse.code.utils import utils

    conventions_flr.check_environment()
    link = utils.Link(1, 2)
    s = emri.prepare(n_signal, nf, link_convention=(link,))
    row = common.LINK_LABELS.index(10 * link.receiver + link.sender)

    grids = {}
    tick = time.perf_counter()
    grids["WDM"] = np.asarray(s.response.one_arm_response(
        s.theta, s.phi, s.wdm.dwt(s.h_plus), s.wdm.dwt(s.h_cross), link=link))
    print(f"WDM response ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    grids["TD"] = np.asarray(s.wdm.dwt(np.asarray(s.response.one_arm_response_time_domain(
        s.theta, s.phi, s.h_plus, s.h_cross, link=link,
        num_integral_points=NUM_INTEGRAL_POINTS))))
    print(f"time-domain reference ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    gwr = conventions_gwr.links(ReadStrain, s.grid, s.h_plus, s.h_cross, s.theta, s.phi)
    print(f"lisagwresponse ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    flr, _ = conventions_flr.run(pyResponseTDI, emri.flr_orbits(), s.grid, s.h_plus,
                                 s.h_cross, s.theta, s.phi)
    print(f"fastlisaresponse ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, grids, dict(GWR=gwr[row], FLR=flr[row]))


if __name__ == "__main__":
    emri.main(compute, "fig12_emri_link",
              ("fig12_emri_link_signals", "fig12_emri_link_residuals"),
              LABELS, "link 1 -> 2", r"\delta\ell_{1\to2}", residual=("WDM", "GWR"))
