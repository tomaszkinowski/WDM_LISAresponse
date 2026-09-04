"""The per-pixel quantities of the wavelet response.

Three small functions between the geometry and the shift: the effective light
travel time, the delay that carries a pixel to its window centre, and the
per-pixel amplitudes.

They need no fixture - the geometry arrays are built here by hand, which is
what makes the limiting cases reachable. A real orbit never puts an arm
exactly along the line of sight, and that is precisely the case where the
physics is easiest to state and a sign error is impossible to hide.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from lisaconstants import c

from WDM_LISAresponse.code.lisa_response.response_sub import (
    build_sky_basis,
    compute_angular_geometry_wdm,
    compute_detector_amplitudes,
    compute_time_shifts,
)

NUM_TIMES = 8
LIGHT_TRAVEL_TIME = 8.3

THETA = jnp.asarray([0.7])
PHI = jnp.asarray([1.3])


def _geometry(direction):
    """Build a one-link geometry pointing a given way at every time."""
    light_travel_time = jnp.full((1, NUM_TIMES), LIGHT_TRAVEL_TIME)
    link_direction = jnp.broadcast_to(
        jnp.asarray(direction), (1, NUM_TIMES, 3))
    return light_travel_time, link_direction


def _source_direction():
    """The unit vector towards the source, for the fixed sky position."""
    _, _, n = build_sky_basis(THETA, PHI)
    return n[0]


# ------------------------------------------------ the effective travel time

def test_tau_vanishes_for_a_wave_travelling_with_the_photon():
    """A wave surfing the arm accumulates no phase, so tau is zero.

    tau = ell (1 + n.r) and the photon travels along -n here, so the factor
    closes exactly. This is the degenerate direction of the whole model and
    the cleanest check that the sign of the projection is right.
    """
    light_travel_time, link_direction = _geometry(-_source_direction())

    _, _, tau, _ = compute_angular_geometry_wdm(
        THETA, PHI, light_travel_time, link_direction)

    assert np.allclose(tau, 0.0, atol=1e-13)


def test_tau_is_the_light_travel_time_across_the_arm():
    """A wave crossing the arm sees the plain light travel time.

    n.r = 0, so the closing-rate factor is one.
    """
    u, _, _ = build_sky_basis(THETA, PHI)
    light_travel_time, link_direction = _geometry(u[0])

    _, _, tau, _ = compute_angular_geometry_wdm(
        THETA, PHI, light_travel_time, link_direction)

    assert np.allclose(tau, LIGHT_TRAVEL_TIME, rtol=1e-13)


def test_tau_doubles_for_a_wave_travelling_against_the_photon():
    """Head-on, the closing rate is twice the light travel time."""
    light_travel_time, link_direction = _geometry(_source_direction())

    _, _, tau, _ = compute_angular_geometry_wdm(
        THETA, PHI, light_travel_time, link_direction)

    assert np.allclose(tau, 2.0 * LIGHT_TRAVEL_TIME, rtol=1e-13)


def test_tau_is_linear_in_the_light_travel_time(rng):
    """Doubling the arm doubles tau at fixed geometry."""
    _, link_direction = _geometry(rng.normal(size=3) / np.sqrt(3.0))
    short = jnp.full((1, NUM_TIMES), 4.0)

    _, _, tau_short, _ = compute_angular_geometry_wdm(
        THETA, PHI, short, link_direction)
    _, _, tau_long, _ = compute_angular_geometry_wdm(
        THETA, PHI, 2.0 * short, link_direction)

    assert np.allclose(tau_long, 2.0 * tau_short, rtol=1e-13)


def test_the_source_direction_comes_back_unchanged():
    """n is returned so the caller can form the plane-wave phase.

    It must be the same vector ``build_sky_basis`` produced, not a
    recomputed or negated one.
    """
    light_travel_time, link_direction = _geometry([1.0, 0.0, 0.0])

    _, _, _, returned = compute_angular_geometry_wdm(
        THETA, PHI, light_travel_time, link_direction)

    _, _, expected = build_sky_basis(THETA, PHI)
    assert np.allclose(returned, expected, atol=1e-15)


# ------------------------------------------------------------- the time shift

def test_the_time_shift_is_the_plane_wave_phase_less_half_the_window(rng):
    """delta = n.x_r/c - tau/2, against a direct computation.

    The first term is the plane-wave phase at the receiver; the second steps
    back to the middle of the retarded-phase window.
    """
    receiver = jnp.asarray(rng.normal(size=(1, NUM_TIMES, 3)) * 1.5e11)
    tau = jnp.asarray(rng.uniform(1.0, 16.0, size=(1, 1, NUM_TIMES)))
    n = jnp.asarray(_source_direction())[jnp.newaxis]

    got = compute_time_shifts(n, receiver, tau)

    expected = np.einsum("bi,lti->blt", np.asarray(n) / c,
                         np.asarray(receiver)) - 0.5 * np.asarray(tau)
    assert np.allclose(got, expected, rtol=1e-13)


def test_a_receiver_at_the_barycentre_leaves_only_the_window_term():
    """With no plane-wave phase the shift is exactly -tau/2."""
    receiver = jnp.zeros((1, NUM_TIMES, 3))
    tau = jnp.full((1, 1, NUM_TIMES), 8.3)
    n = jnp.asarray(_source_direction())[jnp.newaxis]

    got = compute_time_shifts(n, receiver, tau)

    assert np.allclose(got, -0.5 * 8.3, rtol=1e-14)


# -------------------------------------------------------------- the amplitudes

def test_the_amplitudes_reduce_to_half_ell_f_when_tau_vanishes():
    """With no window to average over the sinc is one.

    The amplitude is then half the light travel time times the antenna
    factor, which is the whole transfer function stripped of its averaging.
    """
    light_travel_time = jnp.full((1, NUM_TIMES), LIGHT_TRAVEL_TIME)
    plus = jnp.full((1, 1, NUM_TIMES), 0.4)
    cross = jnp.full((1, 1, NUM_TIMES), -0.25)
    frequencies = jnp.linspace(0.0, 0.05, 6)

    amplitude_plus, amplitude_cross = compute_detector_amplitudes(
        light_travel_time, plus, cross, jnp.zeros((1, 1, NUM_TIMES)),
        frequencies)

    assert np.allclose(amplitude_plus, 0.5 * LIGHT_TRAVEL_TIME * 0.4)
    assert np.allclose(amplitude_cross, 0.5 * LIGHT_TRAVEL_TIME * -0.25)


def test_the_transfer_function_is_a_sinc_of_the_window(rng):
    """The averaging factor is sinc(f tau), evaluated per pixel.

    Averaging a wave over a window of width tau gives exactly this, and it
    is the closed form the time-domain path reaches by quadrature instead.
    """
    light_travel_time = jnp.full((1, NUM_TIMES), 2.0)
    plus = jnp.ones((1, 1, NUM_TIMES))
    tau = jnp.asarray(rng.uniform(1.0, 16.0, size=(1, 1, NUM_TIMES)))
    frequencies = jnp.linspace(0.0, 0.05, 6)

    amplitude_plus, _ = compute_detector_amplitudes(
        light_travel_time, plus, plus, tau, frequencies)

    expected = np.sinc(np.asarray(tau)[..., None] * np.asarray(frequencies))
    assert np.allclose(amplitude_plus, expected, rtol=1e-13)


def test_the_amplitudes_are_linear_in_the_antenna_factors():
    """Scaling F scales the amplitude, with nothing else moving."""
    light_travel_time = jnp.full((1, NUM_TIMES), LIGHT_TRAVEL_TIME)
    tau = jnp.full((1, 1, NUM_TIMES), 4.0)
    frequencies = jnp.linspace(0.0, 0.05, 6)
    unit = jnp.ones((1, 1, NUM_TIMES))

    single, _ = compute_detector_amplitudes(
        light_travel_time, unit, unit, tau, frequencies)
    tripled, _ = compute_detector_amplitudes(
        light_travel_time, 3.0 * unit, unit, tau, frequencies)

    assert np.allclose(tripled, 3.0 * single, rtol=1e-14)


def test_the_two_polarisations_do_not_leak_into_each_other():
    """Each amplitude carries only its own antenna factor."""
    light_travel_time = jnp.full((1, NUM_TIMES), LIGHT_TRAVEL_TIME)
    tau = jnp.full((1, 1, NUM_TIMES), 4.0)
    frequencies = jnp.linspace(0.0, 0.05, 6)

    amplitude_plus, amplitude_cross = compute_detector_amplitudes(
        light_travel_time,
        jnp.ones((1, 1, NUM_TIMES)),
        jnp.zeros((1, 1, NUM_TIMES)),
        tau, frequencies)

    assert np.all(np.abs(np.asarray(amplitude_plus)) > 0.0)
    assert np.allclose(amplitude_cross, 0.0, atol=1e-15)


# -------------------------------------------------------------------- shapes

def test_the_frequency_axis_appears_only_at_the_amplitudes():
    """Everything before the amplitudes stays (..., Nt).

    This is what keeps the large arrays to one stage. Growing the frequency
    axis earlier would multiply every intermediate by Nf for no gain, and
    the shape check is the cheapest way to notice it happening.
    """
    light_travel_time, link_direction = _geometry([1.0, 0.0, 0.0])
    frequencies = jnp.linspace(0.0, 0.05, 6)

    plus, cross, tau, n = compute_angular_geometry_wdm(
        THETA, PHI, light_travel_time, link_direction)
    shift = compute_time_shifts(n, jnp.zeros((1, NUM_TIMES, 3)), tau)
    amplitude_plus, _ = compute_detector_amplitudes(
        light_travel_time, plus, cross, tau, frequencies)

    assert plus.shape == cross.shape == tau.shape == (1, 1, NUM_TIMES)
    assert shift.shape == (1, 1, NUM_TIMES)
    assert amplitude_plus.shape == (1, 1, NUM_TIMES, 6)


def test_a_batch_of_sources_broadcasts_against_the_links(rng):
    """Source axes lead, link and time axes follow."""
    theta = jnp.asarray(rng.uniform(0.0, np.pi, size=5))
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, size=5))
    light_travel_time = jnp.full((6, NUM_TIMES), LIGHT_TRAVEL_TIME)
    raw = rng.normal(size=(6, NUM_TIMES, 3))
    link_direction = jnp.asarray(
        raw / np.linalg.norm(raw, axis=-1, keepdims=True))

    plus, _, tau, n = compute_angular_geometry_wdm(
        theta, phi, light_travel_time, link_direction)

    assert plus.shape == (5, 6, NUM_TIMES)
    assert tau.shape == (5, 6, NUM_TIMES)
    assert n.shape == (5, 3)


def test_each_source_is_independent(rng):
    """Row i of a batched result equals the standalone single-source call."""
    theta = jnp.asarray(rng.uniform(0.0, np.pi, size=4))
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, size=4))
    light_travel_time, link_direction = _geometry([0.6, -0.8, 0.0])

    _, _, together, _ = compute_angular_geometry_wdm(
        theta, phi, light_travel_time, link_direction)

    for source in range(4):
        _, _, alone, _ = compute_angular_geometry_wdm(
            theta[source:source + 1], phi[source:source + 1],
            light_travel_time, link_direction)
        assert np.allclose(together[source], alone[0], rtol=1e-14)
