"""Investigation (not a paper figure): figure 8 and the h+ time-frequency map
(fig2_emri_tf_map.py) for the EMRI of parameters.py with p0
moved so that it plunges closer to the end of the record, to see how much visible
evolution each choice gives. P0 maps each plunge time after the end of the record
(years, FEW's KerrEccEqFlux trajectory) to the p0, rounded to 0.01, that gives it;
everything else is EMRI_PARAMS.

Each waveform is cached as cache/investigation_emri_plunge_p0<p0>.npz; the figures
go to figures/investigations/, as fig8_emri_polarisations_plunge<years>yr.pdf and
fig2_emri_tf_map_plunge<years>yr.pdf.
"""

import time
from pathlib import Path

import numpy as np

import _emri_comparison as emri
import fig8_emri_polarisations as fig8
import fig2_emri_tf_map as tf_map
from parameters import DT, EMRI_PARAMS, N_SIGNAL, WINDOW_FUNCTION, WINDOW_KWARGS

HERE = Path(__file__).parent
P0 = {1.3: 8.89, 1.0: 8.59, 0.75: 8.31, 0.5: 8.00, 0.25: 7.65}


def waveform(p0):
    """h+ and hx for EMRI_PARAMS with this p0, tapered and unpadded, cached."""
    cache = HERE / "cache" / f"investigation_emri_plunge_p0{p0:.2f}.npz"
    if not cache.exists():
        from WDM_LISAresponse.code.waveforms import waveforms

        tick = time.perf_counter()
        _, h_plus, h_cross = waveforms.generate_windowed_waveform(
            "emri", t0=emri.signal_start(), dt=DT, N=N_SIGNAL,
            model_kwargs=dict(td_gen=waveforms.emri_generator(), **dict(EMRI_PARAMS, p0=p0)),
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        print(f"p0 = {p0}: EMRI generated ({time.perf_counter() - tick:.0f} s)", flush=True)
        np.savez(cache, h_plus=h_plus, h_cross=h_cross)
    cached = np.load(cache)
    return cached["h_plus"], cached["h_cross"]


if __name__ == "__main__":
    out = HERE / "figures" / "investigations"
    out.mkdir(parents=True, exist_ok=True)
    for years, p0 in P0.items():
        h_plus, h_cross = waveform(p0)
        fig8.draw(h_plus, h_cross, out / f"fig8_emri_polarisations_plunge{years:g}yr.pdf")
        tf_map.draw(h_plus, h_cross, out / f"fig2_emri_tf_map_plunge{years:g}yr.pdf")
