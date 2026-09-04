"""Source-frame basis and the antenna pattern.

Two functions, and between them one exact identity worth more than any
tolerance-based check: with an orthonormal triad,

    F+^2 + Fx^2 = (1 - (r.n)^2)^2

holds identically. It ties the two functions together, so a sign error, a
swapped basis vector or a mis-contracted index breaks it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response.response_sub import (
    antenna_pattern_factors,
    build_sky_basis,
)


@pytest.fixture
def sky(rng):
    """A batch of random source directions."""
    theta = rng.uniform(0.0, np.pi, size=6)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=6)
    return jnp.asarray(theta), jnp.asarray(phi)


@pytest.fixture
def directions(rng):
    """Random unit vectors shaped like a link geometry array."""
    raw = rng.normal(size=(4, 9, 3))
    return jnp.asarray(raw / np.linalg.norm(raw, axis=-1, keepdims=True))


# ------------------------------------------------------------------ the triad

def test_the_triad_is_orthonormal(sky):
    """u, v and n are mutually orthogonal unit vectors.

    Everything downstream assumes it: the antenna pattern is a projection
    onto this basis, and the identity below follows from orthonormality
    alone.
    """
    u, v, n = build_sky_basis(*sky)

    for vector in (u, v, n):
        assert np.allclose(jnp.sum(vector * vector, axis=-1), 1.0, atol=1e-14)

    assert np.allclose(jnp.sum(u * v, axis=-1), 0.0, atol=1e-14)
    assert np.allclose(jnp.sum(u * n, axis=-1), 0.0, atol=1e-14)
    assert np.allclose(jnp.sum(v * n, axis=-1), 0.0, atol=1e-14)


def test_known_direction_gives_the_hand_computed_triad():
    """A spot check against the definitions, at theta = pi/2, phi = 0.

    Orthonormality alone would accept a triad rotated or reflected from the
    intended one; this pins the actual convention.
    """
    u, v, n = build_sky_basis(jnp.asarray(np.pi / 2), jnp.asarray(0.0))

    assert np.allclose(u, [0.0, 0.0, -1.0], atol=1e-15)
    assert np.allclose(v, [0.0, -1.0, 0.0], atol=1e-15)
    assert np.allclose(n, [1.0, 0.0, 0.0], atol=1e-15)


def test_n_points_from_the_barycentre_towards_the_source():
    """n is the source direction, so the wave propagates along -n.

    Getting this backwards flips every delay in the response, and nothing
    downstream would look obviously wrong.
    """
    _, _, n = build_sky_basis(jnp.asarray(0.0), jnp.asarray(0.0))

    assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-15)


# --------------------------------------------------------- the antenna pattern

def test_the_pattern_identity_holds_exactly(sky, directions):
    """F+^2 + Fx^2 = (1 - (r.n)^2)^2, to machine precision.

    The strongest single check available here. It follows from the triad
    being orthonormal and the two factors being the right contractions, so
    it fails if either half is wrong.
    """
    u, v, n = build_sky_basis(*sky)
    plus, cross = antenna_pattern_factors(directions, u, v)
    projection = jnp.tensordot(n, directions, axes=([-1], [-1]))

    assert np.allclose(plus ** 2 + cross ** 2,
                       (1.0 - projection ** 2) ** 2, atol=1e-14)


def test_a_link_along_the_line_of_sight_has_no_response():
    """A link parallel to n projects onto neither polarisation.

    The limiting case of the identity above: r.n = 1 makes the right-hand
    side vanish, so both factors must.
    """
    u, v, n = build_sky_basis(jnp.asarray(0.7), jnp.asarray(1.3))
    along = n[jnp.newaxis, jnp.newaxis, :]

    plus, cross = antenna_pattern_factors(along, u, v)

    assert np.allclose(plus, 0.0, atol=1e-15)
    assert np.allclose(cross, 0.0, atol=1e-15)


def test_the_pattern_is_blind_to_the_direction_of_travel():
    """Reversing a link leaves the pattern unchanged.

    Both factors are quadratic in r, so the two ends of an arm share them.
    """
    u, v, n = build_sky_basis(jnp.asarray(0.7), jnp.asarray(1.3))
    forward = jnp.asarray([[[0.3, -0.5, 0.81]]])

    plus, cross = antenna_pattern_factors(forward, u, v)
    reversed_plus, reversed_cross = antenna_pattern_factors(-forward, u, v)

    assert np.allclose(plus, reversed_plus, atol=1e-15)
    assert np.allclose(cross, reversed_cross, atol=1e-15)


# -------------------------------------------------------------------- shapes

def test_source_axes_come_before_link_axes(sky, directions):
    """Output is (source..., link...), the convention the response relies on.

    The two operands carry independent leading axes - u and v are indexed by
    source, the directions by link and time - and axis 0 of the result must
    be the source.
    """
    u, v, _ = build_sky_basis(*sky)

    plus, cross = antenna_pattern_factors(directions, u, v)

    assert plus.shape == (6, 4, 9)
    assert cross.shape == (6, 4, 9)


def test_scalar_angles_broadcast_against_array_angles():
    """A scalar sky position behaves like a length-one batch."""
    scalar_u, scalar_v, scalar_n = build_sky_basis(0.7, 1.3)
    array_u, array_v, array_n = build_sky_basis(
        jnp.asarray([0.7]), jnp.asarray([1.3]))

    assert scalar_u.shape == (3,)
    assert array_u.shape == (1, 3)
    assert np.allclose(scalar_u, array_u[0], atol=1e-15)
    assert np.allclose(scalar_v, array_v[0], atol=1e-15)
    assert np.allclose(scalar_n, array_n[0], atol=1e-15)


def test_one_angle_broadcasts_against_many(rng):
    """theta and phi broadcast against each other, not just elementwise."""
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, size=5))

    u, _, _ = build_sky_basis(jnp.asarray(0.7), phi)

    assert u.shape == (5, 3)
