"""Figure 10: the EMRI (parameters.py) in second-generation Michelson TDI X
at Delta T = 1e4 s, from four codes:
  - WDM: our wavelet links, combined by comparison.tdi.TDIgen2
  - TD: our time-domain links, combined by comparison.tdi_reference
  - lisagwresponse's links, combined by pytdi
  - fastlisaresponse, links and TDI both its own

X over the whole record and a third-of-a-day zoom. In figures/investigations/: the same
with the source's h+ and hx over the zoom (fig10_emri_tdi_signals_waveform.pdf),
and the residuals of every pair over the zoom (fig10_emri_tdi_residuals.pdf).

As figure 9 (_emri_comparison.py has what they share), except that the
guard band is twice as wide: TDI shifts the signal a second time. The mismatches
over the whole cut record, for the caption, are printed and written to
figures/fig10_emri_tdi_mismatches.txt.

Run with no arguments to compute (or reuse the cache) and plot. ``--short`` runs
2**19 samples (24 d), prints the mismatches and draws into figures/investigations/
without caching - for checking the pipeline on the laptop. ``--nf NF`` runs the
whole record at another tiling, cached and drawn as an investigation.
"""

import time

import numpy as np

import _emri_comparison as emri
from parameters import MAX_LAG_L, NUM_INTEGRAL_POINTS

LABELS = dict(WDM="WDM response", TD="time-domain reference",
              GWR="lisagwresponse + pytdi", FLR="fastlisaresponse")


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
    emri.main(compute, "fig10_emri_tdi",
              ("fig10_emri_tdi_signals", "fig10_emri_tdi_residuals"),
              LABELS, "second-generation X", "X_2")
