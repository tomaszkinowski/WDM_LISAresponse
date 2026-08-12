"""Unit tests for geometry helpers used by the LISA response implementation.

These tests validate core geometric identities and a simple analytic light-travel
time case to guard against regressions in low-level helper routines.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.response_sub_old import (
    _compute_ltt_iterative,
    antenna_pattern_factors,
    basis_tensors,
    basis_vectors,
    build_sky_basis,
    generate_time_shifts,
    rhat,
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


def test_basis_vectors_are_orthonormal():
    """Check that (u, v, n) form an orthonormal triad for a generic sky location."""
    theta = 1.0
    phi = 1.0
    u, v, n = basis_vectors(theta, phi)

    assert np.isclose(np.dot(u, v), 0.0, atol=1e-12)
    assert np.isclose(np.dot(u, n), 0.0, atol=1e-12)
    assert np.isclose(np.dot(v, n), 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-12)


def test_basis_tensors_are_trace_free_and_orthogonal():
    """Verify GW polarization tensors are trace-free and mutually orthogonal."""
    e_plus, e_cross = basis_tensors(theta=0.9, phi=1.7)

    assert np.isclose(np.trace(e_plus), 0.0, atol=1e-12)
    assert np.isclose(np.trace(e_cross), 0.0, atol=1e-12)
    assert np.isclose(np.sum(e_plus * e_cross), 0.0, atol=1e-12)


def test_antenna_pattern_factors_match_tensor_projection():
    """Direct antenna factors should match explicit tensor contractions."""
    sky_basis = build_sky_basis(theta=0.9, phi=1.7)
    rhat_vec = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    f_plus, f_cross = antenna_pattern_factors(rhat_vec, sky_basis)
    expected_plus = np.einsum("ti,ij,tj->t", rhat_vec, sky_basis.e_plus, rhat_vec)
    expected_cross = np.einsum("ti,ij,tj->t", rhat_vec, sky_basis.e_cross, rhat_vec)

    assert np.allclose(f_plus, expected_plus, rtol=0.0, atol=1e-12)
    assert np.allclose(f_cross, expected_cross, rtol=0.0, atol=1e-12)


def test_compute_ltt_iterative_constant_separation():
    """Validate iterative LTT against analytic constant-separation travel time."""
    from lisaconstants import c

    orbits = SimpleOrbits(arm_length=2.5e9)
    times = np.linspace(0.0, 1000.0, 128)

    ltt = _compute_ltt_iterative(times, s=1, r=2, orbits=orbits)
    expected = np.full_like(times, 2.5e9 / c)

    assert np.allclose(ltt, expected, rtol=0.0, atol=1e-12)


def test_compute_ltt_iterative_reuses_supplied_sender_positions_and_receiver_solution():
    """Optional reuse outputs should match the default constant-separation solution."""
    from lisaconstants import c

    orbits = SimpleOrbits(arm_length=2.5e9)
    times = np.linspace(0.0, 1000.0, 128)
    x_s = orbits.compute_position(times, sc=[1])[:, 0, :]

    ltt, x_r_rec = _compute_ltt_iterative(
        times,
        s=1,
        r=2,
        orbits=orbits,
        x_s=x_s,
        return_receiver_position=True,
    )

    expected_ltt = np.full_like(times, 2.5e9 / c)
    expected_x_r = orbits.compute_position(times, sc=[2])[:, 0, :]

    assert np.allclose(ltt, expected_ltt, rtol=0.0, atol=1e-12)
    assert np.allclose(x_r_rec, expected_x_r, rtol=0.0, atol=1e-12)


def test_compute_ltt_iterative_reception_time_geometry_identities():
    """Reception-time geometry should satisfy the retarded-time identities exactly."""
    from lisaconstants import c

    orbits = SimpleOrbits(arm_length=2.5e9)
    times = np.linspace(0.0, 1000.0, 128)

    ltt, x_r_rec = _compute_ltt_iterative(
        times,
        s=1,
        r=2,
        orbits=orbits,
        return_receiver_position=True,
    )
    t_emit = times - ltt
    x_s = orbits.compute_position(t_emit, sc=[1])[:, 0, :]
    rhat_vec = (x_r_rec - x_s) / np.linalg.norm(x_r_rec - x_s, axis=1)[:, None]

    assert np.allclose(t_emit + ltt, times, rtol=0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(x_r_rec - x_s, axis=1) / c, ltt, rtol=0.0, atol=1e-12)

    _, _, n = basis_vectors(1.0, 0.4)
    tau = 0.5 * ltt * (1.0 + np.einsum("i,ti->t", n, rhat_vec))
    xi_i_plus_tau = t_emit + np.einsum("i,ti->t", n, x_s / c) + tau
    xi_j_minus_tau = times + np.einsum("i,ti->t", n, x_r_rec / c) - tau

    assert np.allclose(xi_i_plus_tau, xi_j_minus_tau, rtol=0.0, atol=1e-12)


def test_rhat_accepts_reused_link_geometry():
    """Reused sender/receiver positions should produce the same unit vector."""
    orbits = SimpleOrbits(arm_length=2.5e9)
    times = np.linspace(0.0, 1000.0, 128)
    ltt, x_r_rec = _compute_ltt_iterative(
        times,
        s=1,
        r=2,
        orbits=orbits,
        return_receiver_position=True,
    )
    x_s = orbits.compute_position(times - ltt, sc=[1])[:, 0, :]

    r_hat = rhat(times, s=1, r=2, orbits=orbits, tt=ltt, x_r_rec=x_r_rec)
    expected = np.zeros((times.size, 3), dtype=float)
    expected[:, 0] = 1.0

    assert np.allclose(r_hat, expected, rtol=0.0, atol=1e-12)


def test_generate_time_shifts_accepts_precomputed_sky_basis():
    """Supplying a precomputed sky basis should not change generated shifts."""
    orbits = SimpleOrbits(arm_length=2.5e9)
    sky_basis = build_sky_basis(theta=1.0, phi=0.4)

    shifts_default = generate_time_shifts(
        Nt=32,
        times_0=0.0,
        dT=10.0,
        theta=1.0,
        phi=0.4,
        s=1,
        r=2,
        orbits=orbits,
    )
    shifts_reused = generate_time_shifts(
        Nt=32,
        times_0=0.0,
        dT=10.0,
        theta=1.0,
        phi=0.4,
        s=1,
        r=2,
        orbits=orbits,
        sky_basis=sky_basis,
    )

    assert np.allclose(shifts_default, shifts_reused, rtol=0.0, atol=1e-12)
