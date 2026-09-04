"""The time-domain response path.

The reference implementation: the retarded-phase integral evaluated by
quadrature, against which the wavelet path's local-delay approximation is
judged. Its own accuracy is a research question and is not tested here.

What is tested is the machinery around the integral - that blocking the
record does not change the answer, that the two routes to the window width
agree, and that the response is linear in the waveform.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response import response_sub
from WDM_LISAresponse.code.lisa_response.response import LISAresponse
from WDM_LISAresponse.code.utils.utils import Link

from conftest import MAX_LAG_L, T0

pytestmark = pytest.mark.slow

LINK = Link(2, 1)
THETA, PHI = 0.7, 1.3


# ---------------------------------------------------------------- blocking

@pytest.mark.parametrize("chunk_size", [7, 100, 512, 1023])
def test_the_block_length_does_not_change_the_answer(
        chunk_size, response, waveform):
    """Blocking is a memory strategy, never a change of result.

    The record is swept in blocks whose geometry is rebuilt per block, and
    the final block is padded to a fixed length so the kernel compiles once.
    A block size that does not divide the record therefore exercises the
    pad-and-discard path, which nothing else in the suite touches.
    """
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)

    whole = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK,
        time_chunk_size=response.N)
    blocked = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK,
        time_chunk_size=chunk_size)

    assert np.array_equal(np.asarray(whole), np.asarray(blocked))


def test_a_block_length_that_does_not_divide_the_record_is_handled(
        response, waveform):
    """The padded tail of the final block is discarded, not returned."""
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)

    got = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK, time_chunk_size=300)

    assert got.shape == (response.N,)


# ------------------------------------------------------- the two routes to tau

@pytest.mark.parametrize("epoch", [0.0, 1.0e5, 1.0e7, 1.0e9])
def test_the_window_width_agrees_between_the_two_paths(epoch, orbits):
    """tau from the endpoints matches tau from the identity, at any epoch.

    The time-domain path forms the retarded phase at each end of the link;
    the wavelet path reaches the same width through ell (1 + n.r). They are
    algebraically identical, so a disagreement means a geometry term has
    been applied to one of ell or r_hat and not the other - which is what a
    Shapiro delay on ell alone would do.

    The epoch sweep is the point. Formed as a difference of absolute times
    this quantity used to lose precision as eps*t0, reaching 6e-8 s at 3e8;
    cancelling the reception time analytically made it flat. Running at both
    ends of a wide epoch range is what keeps that property honest.
    """
    times = epoch + np.arange(32) * 80.0
    angles = (jnp.asarray([THETA]), jnp.asarray([PHI]))

    ltt, direction, receiver, sender = response_sub.build_link_geometry(
        times, orbits, (LINK,), return_sender_position=True)

    _, _, from_identity, _ = response_sub.compute_angular_geometry_wdm(
        *angles, jnp.asarray(ltt), jnp.asarray(direction))
    _, _, _, from_endpoints, _ = response_sub.compute_angular_geometry_td(
        *angles, jnp.asarray(ltt), jnp.asarray(direction),
        jnp.asarray(receiver), jnp.asarray(sender), jnp.asarray(times))

    assert np.allclose(from_identity, from_endpoints, rtol=0, atol=1e-12)


def test_the_agreement_does_not_degrade_with_the_epoch(orbits):
    """A late epoch is no worse than an early one.

    Stated as a comparison rather than a bound, because it is the *scaling*
    that the relative-time change removed. A fixed tolerance would pass on a
    regression that reintroduced the cancellation, so long as it were small
    enough at the epoch the test happened to use.
    """
    def disagreement(epoch):
        times = epoch + np.arange(32) * 80.0
        angles = (jnp.asarray([THETA]), jnp.asarray([PHI]))
        ltt, direction, receiver, sender = response_sub.build_link_geometry(
            times, orbits, (LINK,), return_sender_position=True)
        _, _, identity, _ = response_sub.compute_angular_geometry_wdm(
            *angles, jnp.asarray(ltt), jnp.asarray(direction))
        _, _, _, endpoints, _ = response_sub.compute_angular_geometry_td(
            *angles, jnp.asarray(ltt), jnp.asarray(direction),
            jnp.asarray(receiver), jnp.asarray(sender), jnp.asarray(times))
        return float(jnp.abs(identity - endpoints).max())

    early = disagreement(0.0)
    late = disagreement(1.0e9)

    assert late < 100.0 * max(early, 1e-15)


# ------------------------------------------------------------------ linearity

def test_a_silent_waveform_gives_a_silent_response(response):
    """No wave in, no response out - exactly, not approximately."""
    silence = jnp.zeros(response.N)

    got = response.one_arm_response_time_domain(
        THETA, PHI, silence, silence, link=LINK)

    assert np.array_equal(np.asarray(got), np.zeros(response.N))


def test_the_response_is_linear_in_the_waveform(response, waveform):
    """Scaling the wave scales the response, with nothing else moving.

    The whole path is linear in h - the quadrature, the antenna factors and
    the prefactor all are - so this holds exactly rather than to a tolerance.
    """
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)

    single = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK)
    scaled = response.one_arm_response_time_domain(
        THETA, PHI, 3.0 * h_plus, 3.0 * h_cross, link=LINK)

    assert np.allclose(scaled, 3.0 * single, rtol=1e-13)


def test_the_polarisations_superpose(response, waveform):
    """Plus and cross contribute independently.

    Feeding each alone and adding the results must reproduce feeding both,
    which fails if the two are ever mixed by the antenna projection.
    """
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)
    silence = jnp.zeros_like(h_plus)

    together = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK)
    plus_only = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, silence, link=LINK)
    cross_only = response.one_arm_response_time_domain(
        THETA, PHI, silence, h_cross, link=LINK)

    assert np.allclose(together, plus_only + cross_only, rtol=1e-13)


# --------------------------------------------------------------------- units

def test_the_response_is_a_time_of_the_expected_order(response, waveform):
    """The result is a delay in seconds, not an optical path length.

    A factor of c between the two would leave everything else looking
    entirely reasonable, so the order of magnitude is worth an assertion:
    the delay fluctuation is of order the light travel time times the strain.
    """
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)

    got = response.one_arm_response_time_domain(
        THETA, PHI, h_plus, h_cross, link=LINK)

    light_travel_time = float(jnp.abs(response.light_travel_time).max())
    peak = float(jnp.abs(got).max())

    assert 1e-3 * light_travel_time < peak < light_travel_time
