"""Unit tests for discrete time-shift construction helpers.

These tests use a simple analytic orbit model to validate the implemented shift
formula in a case where the expected answer is known exactly.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from lisaconstants import c

from WDM_LISAresponse.code.lisa_response.response_sub import (
    basis_vectors,
    generate_time_shifts,
)


class SimpleOrbits:
    def __init__(self, arm_length=2.5e9):
        self.arm_length = float(arm_length)

    def compute_position(self, times, sc):
        times = np.atleast_1d(times)
        spacecraft = int(sc[0])
        x = (spacecraft - 1) * self.arm_length
        position = np.zeros((times.size, 1, 3), dtype=float)
        position[:, 0, 0] = x
        return position


def test_generate_time_shifts_constant_geometry_matches_formula():
    """Check generated shifts against closed-form constant-geometry expression."""
    Nt = 32
    t0 = 100.0
    dT = 2.0
    theta = 0.7
    phi = 1.2
    s = 1
    r = 2

    orbits = SimpleOrbits(arm_length=2.5e9)
    shifts = generate_time_shifts(Nt, t0, dT, theta, phi, s, r, orbits)

    _, _, n = basis_vectors(theta, phi)
    rhat_x = 1.0
    l_sr = 2.5e9
    tau = 0.5 * l_sr * (1.0 + rhat_x * n[0])

    x_s = 0.0
    expected = np.full(Nt, x_s * n[0] / c + tau)

    assert shifts.shape == (Nt,)
    assert np.allclose(shifts, expected, rtol=0.0, atol=1e-9)
