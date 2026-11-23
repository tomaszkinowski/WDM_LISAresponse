"""Figure 13 with the zeroth-order wavelet links: the EMRI in second-generation
Michelson X at Delta T = 1e4 s, with every link from production one_arm_response,
combined by comparison.tdi.TDIgen2; the time-domain reference, lisagwresponse +
pytdi and fastlisaresponse are computed alongside.

Writes figures/fig13_emri_tdi_signals_zeroth_order.pdf and
fig13_emri_tdi_mismatches_zeroth_order.txt, and
figures/investigations/fig13_emri_tdi_{signals_waveform,residuals}_zeroth_order.pdf;
cache cache/investigation_fig13_zeroth_order.npz. Options as figure 13's
(``--short``, ``--nf NF``).
"""

import time

import numpy as np

import _emri_comparison as emri
from fig13_emri_tdi_comparison import LABELS
from parameters import MAX_LAG_L, NUM_INTEGRAL_POINTS


def compute(n_signal, nf):
    from fastlisaresponse import pyResponseTDI
    from lisagwresponse import ReadStrain
    from pytdi import Data
    from pytdi.michelson import X2

    from WDM_LISAresponse.code.comparison import conventions_flr, conventions_gwr
    from WDM_LISAresponse.code.comparison.tdi import TDIgen2
    from WDM_LISAresponse.code.comparison.tdi_reference import gen2_xyz_time_domain

    conventions_flr.check_environment()
    s = emri.prepare(n_signal, nf, extra_rows=-(-MAX_LAG_L // 2))

    grids = {}
    tick = time.perf_counter()
    links = s.response(s.theta, s.phi, s.wdm.dwt(s.h_plus), s.wdm.dwt(s.h_cross))
    grids["WDM"] = np.asarray(TDIgen2(s.response, channels=("X",))(links)[0])
    print(f"WDM response and TDI ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    links = np.asarray(s.response.one_arm_response_time_domain(
        s.theta, s.phi, s.h_plus, s.h_cross, num_integral_points=NUM_INTEGRAL_POINTS))
    x = gen2_xyz_time_domain(s.response, links, channels=("X",))[0]
    grids["TD"] = np.asarray(s.wdm.dwt(x))
    print(f"time-domain reference and TDI ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    links = conventions_gwr.links(ReadStrain, s.grid, s.h_plus, s.h_cross, s.theta, s.phi)
    gwr = conventions_gwr.channels(Data, (X2,), s.grid, links)[0]
    print(f"lisagwresponse + pytdi ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    _, flr = conventions_flr.run(pyResponseTDI, emri.flr_orbits(), s.grid, s.h_plus,
                                 s.h_cross, s.theta, s.phi)
    print(f"fastlisaresponse ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, grids, dict(GWR=gwr, FLR=flr[0]))


if __name__ == "__main__":
    emri.main(compute, "fig13_emri_tdi",
              ("fig13_emri_tdi_signals", "fig13_emri_tdi_residuals"),
              LABELS, "second-generation X", "X_2", variant="zeroth_order")
