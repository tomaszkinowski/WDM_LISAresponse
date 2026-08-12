"""Unit tests for interpolation helper used during response prototyping.

The production path is expected to rely primarily on coefficient-space shifting,
but these tests preserve a validated baseline for early-method comparisons.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.response_sub_old import gnm_at_times_grid_np_interp


class DummyWDM:
    def __init__(self, dt=0.5, N=16):
        self.dt = float(dt)
        self.N = int(N)

    def gnm(self, n, m):
        base = np.arange(self.N, dtype=float)
        return base + 100.0 * n + 10.0 * m


def test_gnm_interp_nearest_matches_samples():
    """Nearest mode should reproduce exact sample values at on-grid query times."""
    wdm = DummyWDM(dt=0.5, N=16)
    n_grid = np.array([[0, 1], [2, 3]])
    m_grid = np.array([[0, 1], [1, 0]])
    t_query = np.array([0.0, 0.5, 1.0, 1.5])

    vals = gnm_at_times_grid_np_interp(
        wdm, n_grid, m_grid, t_query, method="nearest", wrap=False, time_first=True
    )

    assert vals.shape == (4, 2, 2)
    expected_first_mode = wdm.gnm(0, 0)[[0, 1, 2, 3]]
    assert np.allclose(vals[:, 0, 0], expected_first_mode)


def test_gnm_interp_linear_half_step_on_linear_signal():
    """Linear mode should recover midpoint values exactly for linear-in-index data."""
    wdm = DummyWDM(dt=1.0, N=32)
    n_grid = np.array([[1]])
    m_grid = np.array([[2]])
    t_query = np.array([2.5, 7.5, 10.5])

    vals = gnm_at_times_grid_np_interp(
        wdm, n_grid, m_grid, t_query, method="linear", wrap=False, time_first=True
    )

    expected = np.array([122.5, 127.5, 130.5])
    assert np.allclose(vals[:, 0, 0], expected, atol=1e-12)


def test_gnm_interp_wrap_vs_no_wrap_at_boundaries_legacy_wrap_check():
    """Document legacy wrap behavior and preferred non-wrap boundary behavior.

    For physical response calculations, zero-extension (`wrap=False`) is the
    safer default to avoid injecting DFT periodicity artifacts at boundaries.
    This test keeps wrap behavior explicit only as a regression reference.
    """
    wdm = DummyWDM(dt=1.0, N=8)
    n_grid = np.array([[0]])
    m_grid = np.array([[0]])
    t_query = np.array([-1.0, 8.0])

    vals_wrap = gnm_at_times_grid_np_interp(
        wdm, n_grid, m_grid, t_query, method="nearest", wrap=True, time_first=True
    )
    vals_no_wrap = gnm_at_times_grid_np_interp(
        wdm, n_grid, m_grid, t_query, method="nearest", wrap=False, time_first=True
    )

    assert np.allclose(vals_wrap[:, 0, 0], np.array([7.0, 0.0]))
    assert np.allclose(vals_no_wrap[:, 0, 0], np.array([0.0, 0.0]))
