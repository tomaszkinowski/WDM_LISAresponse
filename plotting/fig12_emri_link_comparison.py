"""Figure 12: the EMRI (parameters.py) on link 1 -> 2 at Delta T = 1e4 s, from four
codes - the first-order wavelet response split_amp_sinc_first
(first_order/model.py), the time-domain reference one_arm_response_time_domain,
lisagwresponse and fastlisaresponse: the whole record, and a quarter-day zoom with
the wavelet-minus-lisagwresponse residual overlaid (x 1e3), in
figures/fig12_emri_link_signals.pdf.

The mismatches over the whole cut record, for the caption, are printed and written
to figures/fig12_emri_link_mismatches.txt. Cache cache/fig12_emri_link_comparison.npz;
the three other codes' grids are cached apart
(cache/fig12_references_nfNF.npz), so a rerun after a change to the
wavelet response recomputes only that. What the four share, and how the external
codes are converted, is in _emri_comparison.py.

Run with no arguments to compute (or reuse the cache) and plot. ``--short`` runs
2**19 samples (24 d) and prints the mismatches, caching and drawing nothing - a
quick check of the pipeline.
"""

import time

import numpy as np

import _emri_comparison as emri
from _cache import load_or_compute
from parameters import N_SIGNAL, NUM_INTEGRAL_POINTS

LABELS = dict(WDM="WDM response", TD="time-domain reference", GWR="lisagwresponse",
              FLR="fastlisaresponse")


def compute(n_signal, nf):
    from first_order import model
    from WDM_LISAresponse.code.comparison import common
    from WDM_LISAresponse.code.utils import utils

    link = utils.Link(1, 2)
    s = emri.prepare(n_signal, nf, link_convention=(link,))
    row = common.LINK_LABELS.index(10 * link.receiver + link.sender)

    def references():
        from fastlisaresponse import pyResponseTDI
        from lisagwresponse import ReadStrain
        from WDM_LISAresponse.code.comparison import conventions_flr, conventions_gwr

        conventions_flr.check_environment()
        tick = time.perf_counter()
        td = np.asarray(s.wdm.dwt(np.asarray(s.response.one_arm_response_time_domain(
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
        return dict(TD=td, GWR=np.asarray(gwr[row]), FLR=np.asarray(flr[row]))

    full = n_signal == N_SIGNAL
    ref = load_or_compute(emri.HERE / "cache" / f"fig12_references_nf{nf}.npz"
                          if full else None, references)

    tick = time.perf_counter()
    times = np.asarray(s.response.t0 + s.wdm.times)
    consts = model.constants(s.response, s.wdm)
    wp, wc = model.warped_transforms(s.response, s.wdm, consts, s.theta, s.phi,
                                     model.spline_warp(times, s.h_plus, s.h_cross))
    wdm_grid = model.links(s.wdm, consts, s.theta, s.phi, wp, wc)[0]
    print(f"WDM response ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, dict(WDM=wdm_grid, TD=ref["TD"]),
                       dict(GWR=ref["GWR"], FLR=ref["FLR"]))


if __name__ == "__main__":
    emri.main(compute, "fig12_emri_link", "fig12_emri_link_signals", LABELS,
              "link 1 -> 2", r"\delta\ell_{1\to2}", residual=("WDM", "GWR"), gain=1e3)
