"""Figure 12 with the corrected wavelet response: the EMRI on link 1 -> 2 at
Delta T = 1e4 s, with split_amp_sinc_first (_corrected_response.py) in place of
production one_arm_response. The time-domain reference, lisagwresponse and
fastlisaresponse are figure 12's own.

Writes figures/fig12_emri_link_signals_corrected.pdf and fig12_emri_link_mismatches_corrected.txt,
and figures/investigations/fig12_emri_link_{signals_waveform,residuals}_corrected.pdf; cache
cache/investigation_fig12_corrected.npz. The three other codes' grids are cached
apart (cache/investigation_fig12_references_nfNF.npz), so a rerun after a change to
the correction recomputes only the wavelet response. Options as figure 12's
(``--short``, ``--nf NF``). ../prototypes/delay_rate_correction must be on PYTHONPATH.
"""

import time

import numpy as np

import _emri_comparison as emri
from fig12_emri_link_comparison import LABELS as PAPER_LABELS
from parameters import N_SIGNAL, NUM_INTEGRAL_POINTS

LABELS = dict(PAPER_LABELS, WDM="corrected WDM response")


def cached(path, compute):
    """compute(), a dict of arrays, cached at ``path`` (None: not cached)."""
    if path is not None and path.exists():
        return dict(np.load(path))
    out = compute()
    if path is not None:
        np.savez(path, **out)
    return out


def compute(n_signal, nf):
    import _corrected_response as corrected
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
    ref = cached(emri.HERE / "cache" / f"investigation_fig12_references_nf{nf}.npz"
                 if full else None, references)

    tick = time.perf_counter()
    times = np.asarray(s.response.t0 + s.wdm.times)
    consts = corrected.constants(s.response, s.wdm)
    wp, wc = corrected.warped_transforms(s.response, s.wdm, consts, s.theta, s.phi,
                                         corrected.spline_warp(times, s.h_plus, s.h_cross))
    wdm_grid = corrected.links(s.wdm, consts, s.theta, s.phi, wp, wc)[0]
    print(f"corrected WDM response ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, dict(WDM=wdm_grid, TD=ref["TD"]),
                       dict(GWR=ref["GWR"], FLR=ref["FLR"]))


if __name__ == "__main__":
    emri.main(compute, "fig12_emri_link",
              ("fig12_emri_link_signals", "fig12_emri_link_residuals"),
              LABELS, "link 1 -> 2", r"\delta\ell_{1\to2}", residual=("WDM", "GWR"),
              variant="corrected")
