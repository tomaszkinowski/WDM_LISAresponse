"""The time-domain response path.

The reference implementation: the retarded-phase integral evaluated by
quadrature, against which the wavelet path's local-delay approximation is
judged.

Most of what is tested is the machinery around the integral - that blocking
the record does not change the answer, that the two routes to the window width
agree, that the response is linear in the waveform. Those are relative checks:
they would all pass on a path that was self-consistently wrong.

The last test is the exception and the reason the rest can be trusted. It puts
the integral against a closed form for a monochromatic source, so this path is
anchored to an analytic answer rather than to a second implementation. That
matters because it is itself the yardstick: ``test_tdi_gen2`` judges the
wavelet route against this one, and an error here would pass unnoticed through
every comparison built on it.
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


# ------------------------------------------- against an absolute reference


def test_time_domain_response_matches_the_monochromatic_closed_form():
    r"""``one_arm_response_time_domain`` reproduces the analytic one-arm response.

    For a monochromatic plane wave the integral along the null geodesic has a
    closed form. Averaging :math:`h` over the retarded-phase window of width
    :math:`\tau = \ell_{sr}(1 - \hat\Omega\cdot\hat r)` gives the value at the
    window's centre times a sinc, so

    .. math::

        \delta\ell_{sr}(t) = \frac{\tau}{2(1 - \hat\Omega\cdot\hat r)}
            \,\mathrm{sinc}(f\tau)
            \left[F^+ h_+(\xi_{\mathrm{mid}})
                + F^\times h_\times(\xi_{\mathrm{mid}})\right] ,

    with no quadrature and no interpolation anywhere: :math:`h` is evaluated
    from its own formula at :math:`\xi_{\mathrm{mid}}`.

    The two computations share only the geometry. One integrates numerically
    with an interpolated waveform, the other evaluates a closed form, so
    agreement is a genuine check rather than a restatement - and measured, it
    is four to nine orders below the response itself.

    This is the strongest validation the reference path has. It matters
    because that path is what the wavelet response is judged against: an error
    here would be invisible to every comparison that only puts implementations
    beside each other.
    """
    import jax.numpy as jnp
    from scipy.signal.windows import tukey

    from WDM_LISAresponse.code.comparison import config
    from WDM_LISAresponse.code.lisa_response.response_sub import (
        build_link_geometry, compute_angular_geometry_td,
    )
    from WDM_LISAresponse.code.waveforms.waveforms import (
        generate_waveform_galactic_binary,
    )
    from WDM_LISAresponse.code.utils.utils import (
        apply_zero_padding, wdm_mismatch,
    )

    # Nf is left at the study default: m = 13 is then 5.078 mHz, 19.7 samples
    # per cycle. Halving Nf doubles dF and puts the same bin at 10.2 mHz,
    # where the quadrature and the interpolation are both an order worse and
    # this margin closes.
    cfg = config.ANALYTIC
    wdm, response = cfg.build()

    h_plus, h_cross = cfg.waveform()
    _, h_plus, h_cross = apply_zero_padding(
        cfg.signal_times, h_plus, h_cross,
        cfg.padding["pad_left"], cfg.padding["pad_right"],
    )

    times = np.asarray(response.times)
    ltt, direction, receiver, sender = build_link_geometry(
        times, response.orbits_class, response.link_convention,
        return_sender_position=True,
    )

    offsets = np.arange(cfg.N) * cfg.dt
    plus, cross, xi_receive, tau, omega_dot = compute_angular_geometry_td(
        jnp.array([cfg.theta]), jnp.array([cfg.phi]),
        jnp.asarray(ltt), jnp.asarray(direction),
        jnp.asarray(receiver), jnp.asarray(sender), jnp.asarray(offsets),
    )
    plus, cross, xi_receive, tau, omega_dot = (
        np.asarray(a)[0] for a in (plus, cross, xi_receive, tau, omega_dot)
    )

    midpoint = xi_receive - 0.5 * tau

    window = np.interp(
        midpoint, offsets,
        np.concatenate([
            np.zeros(cfg.padding["pad_left"]),
            tukey(cfg.n_signal, cfg.tukey_alpha),
            np.zeros(cfg.padding["pad_right"]),
        ]),
    )

    # The generator, not cfg.waveform(): this needs the untapered binary at
    # the retarded midpoints, which are neither uniformly spaced nor on the
    # signal grid. The taper is put back above, by interpolation.
    mid_plus, mid_cross = generate_waveform_galactic_binary(
        (cfg.t0 + midpoint).ravel(), cfg.amp, cfg.f_gb, cfg.fdot, cfg.iota,
        cfg.phi0, cfg.psi, t_ref=float(cfg.signal_times[0]),
    )
    mid_plus = mid_plus.reshape(midpoint.shape) * window
    mid_cross = mid_cross.reshape(midpoint.shape) * window

    exact = (tau / (2.0 * (1.0 - omega_dot))
             * np.sinc(cfg.f_gb * tau)
             * (plus * mid_plus + cross * mid_cross))
    exact = np.stack([np.asarray(wdm.dwt(row)) for row in exact])

    got = np.asarray(response.one_arm_response_time_domain(
        cfg.theta, cfg.phi, h_plus, h_cross,
        num_taps=20, num_integral_points=80,
    ))
    got = np.stack([np.asarray(wdm.dwt(row)) for row in got])

    # Away from the taper, where the closed form is exactly monochromatic.
    interior = slice(int(0.15 * cfg.Nt), int(0.85 * cfg.Nt))

    for link in range(len(response.link_convention)):
        result = wdm_mismatch(exact[link][interior], got[link][interior])
        assert result["rel_L2"] < 1e-5


def test_time_domain_response_converges_to_the_definition_on_the_esa_orbits():
    r"""On real orbits the only error left is the trapezoid's, as predicted.

    The reference shares nothing with the code under test: its own
    light-travel-time solve and link direction, the closed-form waveform at
    times measured from the record start, and 32-point Gauss-Legendre along
    the photon path, exact to rounding. Measured, the error then follows
    :math:`(2\pi f\tau)^2/12(P-1)^2` to 1 % from :math:`2.6\times10^{-4}` at
    :math:`P = 10` to :math:`3\times10^{-9}` at :math:`P = 2560`, with twelve
    and sixteen taps indistinguishable. Asserting the prediction rather than a
    bound means any other error above ~1e-7 fails it.
    """
    from lisaconstants import c

    from WDM_LISAresponse.code.comparison import config
    from WDM_LISAresponse.code.lisa_response.response_sub import build_sky_basis
    from WDM_LISAresponse.code.waveforms.waveforms import (
        generate_waveform_galactic_binary,
    )
    from WDM_LISAresponse.code.utils.utils import apply_zero_padding

    cfg = config.ComparisonConfig.for_esa(N=2 ** 16)
    wdm, response = cfg.build()
    orbits = response.orbits_class

    h_plus, h_cross = cfg.waveform()
    _, h_plus, h_cross = apply_zero_padding(
        cfg.signal_times, h_plus, h_cross,
        cfg.padding["pad_left"], cfg.padding["pad_right"],
    )

    # Mid-record, away from the taper. Times are offsets from the record
    # start, so float64 rounding at t ~ 2e9 s cannot enter the phase.
    start, count = cfg.N // 2, 1000
    offsets = (start + np.arange(count)) * cfg.dt
    u, v, n = (np.asarray(a) for a in build_sky_basis(cfg.theta, cfg.phi))
    nodes, weights = np.polynomial.legendre.leggauss(32)

    want, tau = [], []
    for sender, receiver in response.link_convention:
        t = cfg.t0 + offsets
        x_r = orbits.compute_position(t, sc=[receiver])[:, 0]
        ell = np.zeros_like(t)
        for _ in range(6):
            x_s = orbits.compute_position(t - ell, sc=[sender])[:, 0]
            ell = np.linalg.norm(x_r - x_s, axis=1) / c
        r = (x_r - x_s) / (c * ell)[:, None]

        # xi = t - Omega.x/c with Omega = -n.
        xi_r = offsets + x_r @ n / c
        xi_s = offsets - ell + x_s @ n / c
        mid, half = (xi_r + xi_s) / 2, (xi_r - xi_s) / 2

        hp, hc = generate_waveform_galactic_binary(
            (mid[:, None] + half[:, None] * nodes).ravel(), cfg.amp, cfg.f_gb,
            cfg.fdot, cfg.iota, cfg.phi0, cfg.psi,
            t_ref=cfg.padding["pad_left"] * cfg.dt,
        )
        integral_p = half * (hp.reshape(count, -1) @ weights)
        integral_c = half * (hc.reshape(count, -1) @ weights)

        ru, rv = r @ u, r @ v
        want.append(((ru ** 2 - rv ** 2) * integral_p
                     + 2.0 * ru * rv * integral_c) / (2.0 * (1.0 + r @ n)))
        tau.append((2 * half).max())

    for P in (40, 160):
        got = np.asarray(response.one_arm_response_time_domain(
            cfg.theta, cfg.phi, h_plus, h_cross, num_integral_points=P,
        ))[:, start:start + count]

        error = max(np.linalg.norm(g - w) / np.linalg.norm(w)
                    for g, w in zip(got, want))
        predicted = ((2.0 * np.pi * cfg.f_gb * max(tau)) ** 2
                     / 12.0 / (P - 1) ** 2)

        assert 0.8 * predicted < error < 1.2 * predicted
