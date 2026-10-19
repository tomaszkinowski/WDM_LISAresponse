"""Broadcasting, domains and the shape of what comes back.

The response follows the WDM transform's convention: leading axes are free and
broadcast, with the array's own axes last. So a scalar sky position with a
stack of waveforms, a stack of positions with one waveform, and matched stacks
all work, and the link axis is present or absent according to how the links
were asked for.

Shapes are cheap to get subtly wrong and expensive to notice, so every case is
checked both for its shape and - where broadcasting is involved - for agreeing
with the standalone single-source call it stands in for.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.utils.utils import Link

pytestmark = pytest.mark.slow

THETA, PHI = 0.7, 1.3


@pytest.fixture
def stacked(wavelet_waveform):
    """Five copies of the waveform, scaled apart to tell them apart."""
    h_plus, h_cross = wavelet_waveform
    weights = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])[:, None, None]
    return h_plus * weights, h_cross * weights


# ------------------------------------------------------------ the batch matrix

def test_a_single_source_has_no_batch_axis(response, wavelet_waveform):
    """Scalar angles and one waveform give a bare (Nt, Nf) grid per link."""
    got = response(THETA, PHI, *wavelet_waveform)

    assert got.shape == (response.num_links, response.Nt, response.Nf)


def test_a_stack_of_sky_positions_broadcasts_against_one_waveform(
        response, wavelet_waveform):
    """Many sky positions, one signal: the parameter-estimation case."""
    theta = jnp.linspace(0.3, 1.2, 5)
    phi = jnp.linspace(0.0, 2.0, 5)

    got = response(theta, phi, *wavelet_waveform)

    assert got.shape == (5, response.num_links, response.Nt, response.Nf)


def test_a_stack_of_waveforms_broadcasts_against_one_sky_position(
        response, stacked):
    """One sky position, many signals."""
    got = response(THETA, PHI, *stacked)

    assert got.shape == (5, response.num_links, response.Nt, response.Nf)


def test_matched_stacks_pair_elementwise(response, stacked):
    """Equal-length stacks pair up elementwise, not as an outer product."""
    theta = jnp.linspace(0.3, 1.2, 5)
    phi = jnp.linspace(0.0, 2.0, 5)

    got = response(theta, phi, *stacked)

    assert got.shape == (5, response.num_links, response.Nt, response.Nf)


def test_each_broadcast_row_equals_the_standalone_call(response, stacked):
    """Row i is what a single call with source i would have produced.

    Shape alone would accept a broadcast that mispairs angles with
    waveforms, or silently reuses one source for all rows. This is what
    makes the batch axis mean what it says.
    """
    theta = jnp.linspace(0.3, 1.2, 5)
    phi = jnp.linspace(0.0, 2.0, 5)
    h_plus, h_cross = stacked

    together = response(theta, phi, h_plus, h_cross)

    for source in range(5):
        alone = response(theta[source], phi[source],
                         h_plus[source], h_cross[source])
        assert np.allclose(together[source], alone, rtol=1e-13)


def test_a_multidimensional_batch_is_flattened_and_restored(
        response, wavelet_waveform):
    """Leading axes are free, not limited to one.

    A grid of sky positions is a natural way to sweep, and the batch axes
    are flattened internally only to be restored on the way out.
    """
    theta = jnp.full((2, 3), THETA)
    phi = jnp.full((2, 3), PHI)

    got = response(theta, phi, *wavelet_waveform)

    assert got.shape == (2, 3, response.num_links,
                         response.Nt, response.Nf)


# -------------------------------------------------------------- the link axis

@pytest.mark.parametrize("link", [Link(2, 1), 0, -1])
def test_a_single_link_specifier_drops_the_link_axis(
        link, response, wavelet_waveform):
    """End to end, a scalar specifier returns one grid rather than a stack."""
    got = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=link)

    assert got.shape == (response.Nt, response.Nf)


@pytest.mark.parametrize("link, expected", [
    ([Link(2, 1)], 1),
    ([0], 1),
    ([0, 3], 2),
    ([Link(2, 1), 3], 2),
    (None, 6),
])
def test_a_sequence_keeps_the_link_axis(
        link, expected, response, wavelet_waveform):
    """A sequence returns a stack, even of one."""
    got = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=link)

    assert got.shape == (expected, response.Nt, response.Nf)


def test_the_same_link_two_ways_gives_the_same_numbers(
        response, wavelet_waveform):
    """Boxing a specifier changes the rank and nothing else."""
    scalar = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=Link(2, 1))
    boxed = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=[Link(2, 1)])

    assert np.array_equal(np.asarray(scalar), np.asarray(boxed[0]))


def test_selecting_a_link_matches_its_row_of_the_full_result(
        response, wavelet_waveform):
    """Asking for one arm gives that arm, not a differently-computed one."""
    every = response(THETA, PHI, *wavelet_waveform)
    one = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=3)

    assert np.array_equal(np.asarray(every[3]), np.asarray(one))


def test_the_link_axis_sits_after_the_batch_axes(response, stacked):
    """Ordering is (batch..., links, Nt, Nf), which the cut relies on."""
    theta = jnp.linspace(0.3, 1.2, 5)

    got = response.one_arm_response(theta, PHI, *stacked, link=[0, 3])

    assert got.shape == (5, 2, response.Nt, response.Nf)


# ------------------------------------------------------------------- domains

def test_a_time_domain_waveform_matches_transforming_it_yourself(
        response, waveform, wavelet_waveform):
    """``in_domain="time"`` is a convenience, not a different calculation."""
    h_plus, h_cross = (jnp.asarray(part) for part in waveform)

    from_time = response.one_arm_response(
        THETA, PHI, h_plus, h_cross, in_domain="time")
    from_wavelets = response.one_arm_response(
        THETA, PHI, *wavelet_waveform)

    assert np.allclose(from_time, from_wavelets, rtol=1e-13)


def test_a_time_domain_result_is_the_inverse_of_the_wavelet_one(
        response, wavelet_waveform):
    """``out_domain="time"`` applies idwt, one link at a time."""
    wavelets = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=0)
    series = response.one_arm_response(
        THETA, PHI, *wavelet_waveform, link=0, out_domain="time")

    assert series.shape == (response.N,)
    assert np.allclose(series, response.wdm_class.idwt(wavelets), rtol=1e-12)


@pytest.mark.parametrize("domain", ["frequency", "wavelets", ""])
def test_an_unknown_output_domain_is_refused(
        domain, response, wavelet_waveform):
    """Only two domains exist, and a typo should say so."""
    with pytest.raises(AssertionError, match="out_domain must be"):
        response.one_arm_response(
            THETA, PHI, *wavelet_waveform, out_domain=domain)


def test_a_waveform_of_the_wrong_shape_is_refused(response, wdm):
    """The trailing axes must match the domain being claimed."""
    wrong = jnp.zeros((wdm.Nt, wdm.Nf + 3))

    with pytest.raises(AssertionError, match="must end in axes"):
        response.one_arm_response(THETA, PHI, wrong, wrong)


# -------------------------------------------------------------------- the call

def test_calling_the_response_is_the_default_path(
        response, wavelet_waveform):
    """``__call__`` is ``one_arm_response`` at its defaults.

    It exists so a likelihood has one obvious entry point; if the two ever
    diverge, the short path becomes a trap.
    """
    direct = response.one_arm_response(THETA, PHI, *wavelet_waveform)
    called = response(THETA, PHI, *wavelet_waveform)

    assert np.array_equal(np.asarray(direct), np.asarray(called))


def test_the_response_is_linear_in_the_waveform(response, wavelet_waveform):
    """Scaling the wave scales the response.

    On a CUDA device WDM runs the time shift in float32, so there linearity
    holds to float32 rounding (relative L2 ~1e-7) rather than to float64.
    """
    h_plus, h_cross = wavelet_waveform

    single = response(THETA, PHI, h_plus, h_cross)
    scaled = response(THETA, PHI, 3.0 * h_plus, 3.0 * h_cross)

    if jax.default_backend() == "gpu":
        rel_l2 = np.linalg.norm(scaled - 3.0 * single) \
            / np.linalg.norm(3.0 * single)
        assert rel_l2 < 1e-6
    else:
        assert np.allclose(scaled, 3.0 * single, rtol=1e-13)


def test_a_silent_waveform_gives_a_silent_response(response, wdm):
    """No wave in, no response out."""
    silence = jnp.zeros((wdm.Nt, wdm.Nf))

    got = response(THETA, PHI, silence, silence)

    assert np.array_equal(np.asarray(got), np.zeros_like(np.asarray(got)))
