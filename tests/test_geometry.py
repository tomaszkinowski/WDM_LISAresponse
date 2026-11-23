"""The sky-independent link geometry.

Everything here is built from the orbits alone - the light-travel time, the
link unit vector and the two spacecraft positions - and the interesting
properties are relations *between* those outputs rather than their values,
which depend on whichever provider is in use.

The analytic provider is used throughout. Its positions are closed form, so
the fixed-point solve converges exactly and the tolerances can be tight; on an
interpolated provider the same quantities are limited by the spline, which is
that package's accuracy and not this one's.
"""

import numpy as np
import pytest
from lisaconstants import c

from WDM_LISAresponse.code.lisa_response.response_sub import (
    build_link_geometry,
    link_direction,
    solve_light_travel_time,
)
from WDM_LISAresponse.code.utils.utils import Link

from conftest import T0

LINKS = (Link(2, 1), Link(3, 2), Link(1, 3))
TIMES = T0 + np.arange(32) * 80.0


# ------------------------------------------------------------- the fixed point

def test_the_solved_delay_matches_the_separation_it_implies(orbits):
    """|x_r - x_s| / c equals the light travel time, to the bit.

    This is what the fixed point is solving for, so it is the definition of
    a converged answer rather than a tolerance to be tuned. The receiver sits
    at reception time and the sender at emission time, and the two are
    consistent only if the iteration has actually closed.
    """
    x_s, x_r, ltt = solve_light_travel_time(TIMES, 2, 1, orbits)

    separation = np.linalg.norm(x_r - x_s, axis=1) / c

    assert np.allclose(separation, ltt, rtol=0, atol=1e-15)


def test_three_sweeps_are_as_converged_as_twelve(orbits):
    """The default iteration count is enough on analytic orbits.

    ``LTT_NUM_ITERATIONS`` is a choice this package makes, so whether it is
    adequate is its own contract. A spacecraft moves by about 1e-4 of an arm
    length in one light-crossing time, so the first sweep gains four digits
    and the second closes it.
    """
    _, _, three = solve_light_travel_time(TIMES, 2, 1, orbits,
                                          num_iterations=3)
    _, _, twelve = solve_light_travel_time(TIMES, 2, 1, orbits,
                                           num_iterations=12)

    assert np.allclose(three, twelve, rtol=0, atol=1e-15)


def test_one_sweep_is_visibly_not_converged(orbits):
    """A single sweep still differs, so the test above is not vacuous.

    Without this, a solve that ignored ``num_iterations`` entirely would
    pass the convergence check.
    """
    _, _, one = solve_light_travel_time(TIMES, 2, 1, orbits,
                                        num_iterations=1)
    _, _, twelve = solve_light_travel_time(TIMES, 2, 1, orbits,
                                           num_iterations=12)

    assert not np.allclose(one, twelve, rtol=0, atol=1e-12)


def test_the_delay_is_positive_and_of_arm_length(orbits):
    """A sanity floor: the answer is a light time across a LISA arm."""
    _, _, ltt = solve_light_travel_time(TIMES, 2, 1, orbits)

    assert np.all(ltt > 0.0)
    assert np.all((ltt > 1.0) & (ltt < 100.0))


def test_supplied_positions_are_used_rather_than_refetched(orbits):
    """Seeding with already-evaluated positions gives the same answer.

    The caller passes ``x_r`` and ``x_s`` to avoid a redundant spline call,
    which must be an optimisation and not a change of result.
    """
    positions = orbits.compute_position(TIMES, sc=[1, 2])

    _, _, seeded = solve_light_travel_time(
        TIMES, 2, 1, orbits,
        x_r=positions[:, 0, :], x_s=positions[:, 1, :])
    _, _, unseeded = solve_light_travel_time(TIMES, 2, 1, orbits)

    assert np.allclose(seeded, unseeded, rtol=0, atol=1e-15)


# ------------------------------------------------------------ the unit vector

def test_the_link_direction_is_a_unit_vector(orbits):
    """Everything downstream contracts it as one."""
    x_s, x_r, _ = solve_light_travel_time(TIMES, 2, 1, orbits)

    r_hat = link_direction(x_s, x_r)

    assert np.allclose(np.linalg.norm(r_hat, axis=1), 1.0, atol=1e-15)


def test_the_link_direction_points_from_sender_to_receiver(orbits):
    """The sign convention, stated as a test.

    Reversed, every projection in the response changes sign and the result
    still looks like a plausible response.
    """
    x_s, x_r, _ = solve_light_travel_time(TIMES, 2, 1, orbits)

    r_hat = link_direction(x_s, x_r)

    assert np.all(np.sum(r_hat * (x_r - x_s), axis=1) > 0.0)


def test_the_two_ends_of_an_arm_are_almost_opposed(orbits):
    """r(s,r) is close to -r(r,s), but not exactly.

    The residual is retardation: the two directions are evaluated at
    different emission times, so they differ at order v/c. Asserting near
    rather than exact opposition is the honest statement, and a much larger
    disagreement would mean the retarded solve had gone wrong.
    """
    forward = link_direction(
        *solve_light_travel_time(TIMES, 2, 1, orbits)[:2])
    backward = link_direction(
        *solve_light_travel_time(TIMES, 1, 2, orbits)[:2])

    alignment = np.sum(forward * backward, axis=1)

    assert np.all(alignment < -0.999)
    assert np.all(alignment > -1.0)


# ------------------------------------------------------------- the whole build

def test_the_link_order_asked_for_is_the_order_returned(orbits):
    """Axis 0 follows the ``links`` argument exactly.

    Every downstream index depends on this: ``link_convention[i]`` names
    what sits at row ``i`` of every geometry array. A reordering here would
    silently attribute one arm's geometry to another.
    """
    straight = build_link_geometry(TIMES, orbits, LINKS)
    reversed_links = build_link_geometry(TIMES, orbits, LINKS[::-1])

    for direct, flipped in zip(straight, reversed_links):
        assert np.allclose(direct, flipped[::-1], rtol=0, atol=1e-15)


def test_the_batched_build_matches_a_link_by_link_solve(orbits):
    """Sharing one position call across links changes nothing.

    All links use the same reception grid, so the positions are fetched once
    and reused as both receiver positions and iteration seeds. That is an
    optimisation, and this is what makes it behaviour-preserving rather than
    merely faster.
    """
    ltt, direction, receiver = build_link_geometry(TIMES, orbits, LINKS)

    for index, (sender, receiver_id) in enumerate(LINKS):
        x_s, x_r, alone = solve_light_travel_time(
            TIMES, sender, receiver_id, orbits)

        assert np.allclose(ltt[index], alone, rtol=0, atol=1e-15)
        assert np.allclose(receiver[index], x_r, rtol=0, atol=1e-15)
        assert np.allclose(direction[index], link_direction(x_s, x_r),
                           rtol=0, atol=1e-15)


def test_asking_for_the_sender_changes_nothing_else(orbits):
    """The extra output is kept, not computed differently.

    The solve produces the sender position either way; the flag only decides
    whether it is returned.
    """
    without = build_link_geometry(TIMES, orbits, LINKS)
    with_sender = build_link_geometry(TIMES, orbits, LINKS,
                                      return_sender_position=True)

    assert len(without) == 3
    assert len(with_sender) == 4
    for plain, extended in zip(without, with_sender):
        assert np.allclose(plain, extended, rtol=0, atol=1e-15)


def test_the_returned_sender_sits_at_emission_time(orbits):
    """x_s is the sender when it emitted, not when the light arrived.

    Checked through the relation the rest of the geometry relies on:
    x_r - x_s is exactly c * ell * r_hat.
    """
    ltt, direction, receiver, sender = build_link_geometry(
        TIMES, orbits, LINKS, return_sender_position=True)

    implied = c * ltt[..., np.newaxis] * direction

    assert np.allclose(receiver - sender, implied, rtol=1e-14)


@pytest.mark.parametrize("links", [
    (Link(2, 1),),
    (Link(2, 1), Link(1, 2)),
    (Link(2, 1), Link(3, 2), Link(1, 3), Link(3, 1), Link(2, 3), Link(1, 2)),
])
def test_any_subset_of_links_can_be_built(links, orbits):
    """A subset is a first-class request, for debugging a single arm."""
    ltt, direction, receiver = build_link_geometry(TIMES, orbits, links)

    assert ltt.shape == (len(links), len(TIMES))
    assert direction.shape == (len(links), len(TIMES), 3)
    assert receiver.shape == (len(links), len(TIMES), 3)


def test_plain_pairs_work_as_well_as_links(orbits):
    """The build accepts either spelling of a directed link."""
    typed = build_link_geometry(TIMES, orbits, LINKS)
    plain = build_link_geometry(
        TIMES, orbits, tuple(tuple(link) for link in LINKS))

    for from_typed, from_plain in zip(typed, plain):
        assert np.allclose(from_typed, from_plain, rtol=0, atol=1e-15)


def test_only_compute_position_is_required_of_a_provider(minimal_orbits):
    """The duck-typed contract is one method wide.

    Providers are supplied by the caller and range from analytic to
    file-backed, so the build must not reach for anything beyond the one
    method it documents.
    """
    ltt, direction, receiver = build_link_geometry(
        TIMES, minimal_orbits, LINKS)

    assert np.all(np.isfinite(ltt))
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-15)
    assert receiver.shape == (len(LINKS), len(TIMES), 3)
