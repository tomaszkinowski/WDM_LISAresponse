"""Figure 13 with the corrected wavelet links: the EMRI in second-generation
Michelson X at Delta T = 1e4 s, with every link from split_amp_sinc_first
(_corrected_response.py) in place of production, combined by the same
comparison.tdi.TDIgen2. The time-domain reference, lisagwresponse + pytdi and
fastlisaresponse are figure 13's own.

Writes figures/fig13_emri_tdi_signals_corrected.pdf and fig13_emri_tdi_mismatches_corrected.txt,
and figures/investigations/fig13_emri_tdi_{signals_waveform,residuals}_corrected.pdf; cache
cache/investigation_fig13_corrected.npz. The three other codes' X are cached apart
(cache/investigation_fig13_references_nfNF.npz), so a rerun after a change to the
correction recomputes only the wavelet side. Options as figure 13's (``--short``,
``--nf NF``). ../prototypes/delay_rate_correction must be on PYTHONPATH.
"""

import time

import numpy as np

import _emri_comparison as emri
from fig13_emri_tdi_comparison import LABELS as PAPER_LABELS
from investigation_fig12_corrected import cached
from parameters import MAX_LAG_L, N_SIGNAL, NUM_INTEGRAL_POINTS

LABELS = PAPER_LABELS


def compute(n_signal, nf):
    import _corrected_response as corrected
    from WDM_LISAresponse.code.comparison.tdi import TDIgen2

    s = emri.prepare(n_signal, nf, extra_rows=-(-MAX_LAG_L // 2))

    def references():
        from fastlisaresponse import pyResponseTDI
        from lisagwresponse import ReadStrain
        from pytdi import Data
        from pytdi.michelson import X2
        from WDM_LISAresponse.code.comparison import conventions_flr, conventions_gwr
        from WDM_LISAresponse.code.comparison.tdi_reference import gen2_xyz_time_domain

        conventions_flr.check_environment()
        tick = time.perf_counter()
        links = np.asarray(s.response.one_arm_response_time_domain(
            s.theta, s.phi, s.h_plus, s.h_cross, num_integral_points=NUM_INTEGRAL_POINTS))
        td = np.asarray(s.wdm.dwt(gen2_xyz_time_domain(s.response, links,
                                                       channels=("X",))[0]))
        print(f"time-domain reference and TDI ({time.perf_counter() - tick:.0f} s)",
              flush=True)
        tick = time.perf_counter()
        links = conventions_gwr.links(ReadStrain, s.grid, s.h_plus, s.h_cross, s.theta,
                                      s.phi)
        gwr = conventions_gwr.channels(Data, (X2,), s.grid, links)[0]
        print(f"lisagwresponse + pytdi ({time.perf_counter() - tick:.0f} s)", flush=True)
        tick = time.perf_counter()
        _, flr = conventions_flr.run(pyResponseTDI, emri.flr_orbits(), s.grid, s.h_plus,
                                     s.h_cross, s.theta, s.phi)
        print(f"fastlisaresponse ({time.perf_counter() - tick:.0f} s)", flush=True)
        return dict(TD=td, GWR=np.asarray(gwr), FLR=np.asarray(flr[0]))

    full = n_signal == N_SIGNAL
    ref = cached(emri.HERE / "cache" / f"investigation_fig13_references_nf{nf}.npz"
                 if full else None, references)

    tick = time.perf_counter()
    times = np.asarray(s.response.t0 + s.wdm.times)
    consts = corrected.constants(s.response, s.wdm)
    wp, wc = corrected.warped_transforms(s.response, s.wdm, consts, s.theta, s.phi,
                                         corrected.spline_warp(times, s.h_plus, s.h_cross))
    links = corrected.links(s.wdm, consts, s.theta, s.phi, wp, wc)
    x = np.asarray(TDIgen2(s.response, channels=("X",))(links)[0])
    print(f"corrected WDM links and TDI ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, dict(WDM=x, TD=ref["TD"]), dict(GWR=ref["GWR"], FLR=ref["FLR"]))


if __name__ == "__main__":
    emri.main(compute, "fig13_emri_tdi",
              ("fig13_emri_tdi_signals", "fig13_emri_tdi_residuals"),
              LABELS, "second-generation X", "X_2", variant="corrected")
