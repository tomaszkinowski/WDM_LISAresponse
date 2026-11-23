"""Figure 12 with the zeroth-order wavelet response: the EMRI on link 1 -> 2 at
Delta T = 1e4 s, with production one_arm_response in place of the first-order
response; the time-domain reference, lisagwresponse and fastlisaresponse are
computed alongside. The grid, window and cut are figure 14's at this tiling, so the
wavelet-vs-reference mismatch here is figure 14's EMRI point at 1e4 s.

Writes figures/fig12_emri_link_signals_zeroth_order.pdf and
fig12_emri_link_mismatches_zeroth_order.txt, and
figures/investigations/fig12_emri_link_{signals_waveform,residuals}_zeroth_order.pdf;
cache cache/investigation_fig12_zeroth_order.npz. Options as figure 12's
(``--short``, ``--nf NF``).
"""

import time

import numpy as np

import _emri_comparison as emri
from fig12_emri_link_comparison import LABELS
from parameters import NUM_INTEGRAL_POINTS


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
              LABELS, "link 1 -> 2", r"\delta\ell_{1\to2}", residual=("WDM", "GWR"),
              variant="zeroth_order")
