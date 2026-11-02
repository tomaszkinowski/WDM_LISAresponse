"""Investigation beside figure 10: the EMRI (parameters.py) in first-generation
Michelson TDI X, from the two external codes - lisagwresponse's links combined by
pytdi, and fastlisaresponse with its own TDI - in figure 10's layout with the
source's h+ and hx over the zoom
(figures/investigations/fig10_emri_tdi_gen1_signals_waveform.pdf).

Our wavelet and time-domain TDI form second generation only, so neither appears.
The grid, guard band, conversion and m = 0 treatment are figure 10's
(_emri_comparison.py); the two codes' mismatch over the whole cut record is printed
and written to figures/investigations/fig10_emri_tdi_gen1_mismatches.txt.

Run with no arguments to compute (or reuse the cache) and plot.
"""

import time

import numpy as np

import _emri_comparison as emri
from parameters import MAX_LAG_L, N_SIGNAL

LABELS = dict(GWR="lisagwresponse + pytdi", FLR="fastlisaresponse")
METHODS = tuple(LABELS)
CACHE = emri.HERE / "cache" / "investigation_emri_tdi_gen1.npz"
INVESTIGATIONS = emri.HERE / "figures" / "investigations"


def compute(n_signal, nf):
    from fastlisaresponse import pyResponseTDI
    from lisagwresponse import ReadStrain
    from pytdi import Data
    from pytdi.michelson import X1

    from WDM_LISAresponse.code.comparison import common, conventions_flr, conventions_gwr
    from WDM_LISAresponse.code.utils import utils

    conventions_flr.check_environment()
    s = emri.prepare(n_signal, nf, extra_rows=-(-MAX_LAG_L // 2))

    tick = time.perf_counter()
    links = conventions_gwr.links(ReadStrain, s.grid, s.h_plus, s.h_cross, s.theta, s.phi)
    gwr = conventions_gwr.channels(Data, (X1,), s.grid, links)[0]
    print(f"lisagwresponse + pytdi ({time.perf_counter() - tick:.0f} s)", flush=True)

    tick = time.perf_counter()
    _, flr = conventions_flr.run(pyResponseTDI, emri.flr_orbits(), s.grid, s.h_plus,
                                 s.h_cross, s.theta, s.phi, tdi="1st generation")
    print(f"fastlisaresponse ({time.perf_counter() - tick:.0f} s)", flush=True)

    grids = {name: common.convert(series[None], s.grid, s.wdm)[0]
             for name, series in dict(GWR=gwr, FLR=flr[0]).items()}
    result = utils.wdm_mismatch(utils.apply_wdm_cut(grids["FLR"], s.cut),
                                utils.apply_wdm_cut(grids["GWR"], s.cut))
    return dict(Nf=s.nf, mismatch=result["mismatch"], rel_L2=result["rel_L2"],
                **emri.to_time(s, grids, METHODS))


if __name__ == "__main__":
    if CACHE.exists():
        res = dict(np.load(CACHE))
    else:
        res = compute(N_SIGNAL, emri.NF)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **res)

    text = (f"Nf = {int(res['Nf'])}, Delta T = {int(res['Nf']) * emri.DT:g} s, "
            f"first-generation X, whole cut record\n"
            f"{LABELS['FLR']} against {LABELS['GWR']}: mismatch "
            f"{float(res['mismatch']):.2e}, rel L2 {float(res['rel_L2']):.2e}")
    print(text)
    INVESTIGATIONS.mkdir(parents=True, exist_ok=True)
    (INVESTIGATIONS / "fig10_emri_tdi_gen1_mismatches.txt").write_text(text + "\n")
    emri.plot_signals(res, INVESTIGATIONS / "fig10_emri_tdi_gen1_signals_waveform.pdf",
                      LABELS, "X_1", emri.waveform(N_SIGNAL), methods=METHODS)
