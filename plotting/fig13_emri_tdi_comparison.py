"""Figure 13: the EMRI (parameters.py) in second-generation Michelson TDI X
at Delta T = 1e4 s, from four codes:
  - WDM: first-order wavelet links split_amp_sinc_first (_corrected_response.py),
    combined by comparison.tdi.TDIgen2
  - TD: our time-domain links, combined by comparison.tdi_reference
  - lisagwresponse's links, combined by pytdi
  - fastlisaresponse, links and TDI both its own

X over the whole record and a quarter-day zoom. In figures/investigations/: the same
with the source's h+ and hx over the zoom (fig13_emri_tdi_signals_waveform.pdf),
and the residuals of every pair over the zoom (fig13_emri_tdi_residuals.pdf).

As figure 12, except that the guard band is twice as wide: TDI shifts the signal a
second time. The mismatches over the whole cut record, for the caption, are printed
and written to figures/fig13_emri_tdi_mismatches.txt. Cache
cache/fig13_emri_tdi_comparison.npz; the three other codes' X are cached apart
(cache/investigation_fig13_references_nfNF.npz). The zeroth-order version is
investigation_fig13_zeroth_order.py. Options as figure 12's (``--short``,
``--nf NF``).
"""

import time

import numpy as np

import _emri_comparison as emri
from fig12_emri_link_comparison import cached
from parameters import MAX_LAG_L, N_SIGNAL, NUM_INTEGRAL_POINTS

LABELS = dict(WDM="WDM response", TD="time-domain reference",
              GWR="lisagwresponse + pytdi", FLR="fastlisaresponse")


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
    print(f"WDM links and TDI ({time.perf_counter() - tick:.0f} s)", flush=True)

    return emri.finish(s, dict(WDM=x, TD=ref["TD"]), dict(GWR=ref["GWR"], FLR=ref["FLR"]))


if __name__ == "__main__":
    emri.main(compute, "fig13_emri_tdi",
              ("fig13_emri_tdi_signals", "fig13_emri_tdi_residuals"),
              LABELS, "second-generation X", "X_2")
