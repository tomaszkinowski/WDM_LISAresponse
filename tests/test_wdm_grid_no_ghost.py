"""Regression tests for waveform-independent WDM grid construction."""

import numpy as np
import WDM

from WDM_LISAresponse.code.lisa_response.response_with_plan import (
    _build_wdm_dT_context_from_wdm,
    _build_wdm_dT_grid,
)


def test_wdm_grid_builder_does_not_call_dwt(monkeypatch):
    """A0 grid construction must not project a dummy waveform."""

    def fail_dwt(self, values):  # pragma: no cover - called only on regression
        raise AssertionError("_build_wdm_dT_grid called wdm.dwt")

    monkeypatch.setattr(WDM.WDM.WDM_transform, "dwt", fail_dwt)

    times = np.arange(1024, dtype=float) * 10.0
    grid = _build_wdm_dT_grid(
        times,
        Nf=32,
        q=4,
        d=4,
        A_frac=0.25,
        calc_m0=True,
    )

    assert "w_t_plus" not in grid
    assert "w_t_cross" not in grid
    assert grid["wdm"].N % (2 * grid["wdm"].Nf) == 0
    assert grid["freq_vals"].shape == (grid["wdm"].Nf,)
    assert grid["int_times_vals"].shape == (grid["wdm"].Nt,)


def test_waveform_context_uses_prebuilt_grid():
    """A1 should perform the real plus/cross projections on the A0 grid."""

    times = np.arange(1024, dtype=float) * 10.0
    plus = np.cos(2.0 * np.pi * 1.0e-3 * times)
    cross = np.sin(2.0 * np.pi * 1.0e-3 * times)

    grid = _build_wdm_dT_grid(
        times,
        Nf=32,
        q=4,
        d=4,
        A_frac=0.25,
        calc_m0=True,
    )

    context = _build_wdm_dT_context_from_wdm(
        times,
        plus,
        cross,
        grid["wdm"],
    )

    assert context["wdm"] is grid["wdm"]
    assert context["w_t_plus"].shape == (grid["wdm"].Nt, grid["wdm"].Nf)
    assert context["w_t_cross"].shape == (grid["wdm"].Nt, grid["wdm"].Nf)
    np.testing.assert_allclose(context["freq_vals"], grid["freq_vals"])
    np.testing.assert_allclose(context["int_times_vals"], grid["int_times_vals"])