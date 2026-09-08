"""Second-generation TDI: the algebra, then the wavelet implementation of it.

The tests are staged so a failure says *which* layer broke.

``gen2_terms`` is pure combinatorics, so the first two tests need no arrays at
all. The next two evaluate the whole combination in the time domain from
closed-form laser noises and a closed-form linear delay law - no transform, no
shift operator, no interpolation, so the only error is float64 and any
residual is the algebra's own. Only the last test brings in the wavelet
machinery, against the time-domain result those tests have just validated.

That staging is not decoration. An earlier version of ``gen2_terms`` had the
wrong operator ordering, and an orbit-based laser-noise test could not see it:
that test built its own link measurements with the shift operator, so it was
limited by errors TDI cannot cancel and read the same either way. The
time-domain test below separates gen-1 from gen-2 by five orders of magnitude.
"""

import numpy as np
import pytest

from WDM_LISAresponse.code.fastlisaresponse_comparison.tdi import gen2_terms
from WDM_LISAresponse.code.utils.utils import Link

#: Arm P is the 1<->3 pair, arm Q the 1<->2 pair. Channel X uses only these.
ARM_P = frozenset({Link(3, 1), Link(1, 3)})
ARM_Q = frozenset({Link(2, 1), Link(1, 2)})

#: Distinct arm lengths, so P and Q have no reason to commute.
L_P, L_Q = 8.339, 8.351

#: Opposite breathing, so a single rate parameter drives both arms.
RATE_Q_OVER_P = -0.7

TONES = np.linspace(2e-3, 8e-3, 9)


# -- analytic scaffolding ------------------------------------------------


def _laser(phases, i, t):
    """Laser noise of spacecraft ``i`` at times ``t``. Closed form."""
    out = np.zeros_like(t)
    for n, f in enumerate(TONES):
        out += np.cos(2.0 * np.pi * f * t + phases[i, n])
    return out


def _laws(rate):
    """link -> l(t), a linear delay law per arm."""
    def one(link):
        base, drift = ((L_P, rate) if link in ARM_P
                       else (L_Q, rate * RATE_Q_OVER_P))
        return lambda t: base + drift * t
    return {lk: one(lk) for lk in (ARM_P | ARM_Q)}


def _measurement(phases, link, t, laws):
    """y_sr(t) = p_s(t - l_sr(t)) - p_r(t), exact."""
    return (_laser(phases, link.sender - 1, t - laws[link](t))
            - _laser(phases, link.receiver - 1, t))


def _composed(chain, t, laws):
    """Cumulative delay of a chain, outermost arm first.

    The same walk ``tdi.composed_delay`` does, on the closed-form law rather
    than an orbit solve.
    """
    total = np.zeros_like(t)
    current = t.copy()
    for link in chain:
        d = laws[link](current)
        total += d
        current -= d
    return total


def _assemble(phases, terms, t, laws):
    """Sum the signed, delayed link measurements. Every term exact."""
    total = np.zeros_like(t)
    for link, chain, sign in terms:
        total += sign * _measurement(
            phases, link, t - _composed(chain, t, laws), laws,
        )
    return total


def _gen1_terms():
    """First generation X, for contrast: beams P,Q against Q,P."""
    Pc = (Link(3, 1), Link(1, 3))
    Qc = (Link(2, 1), Link(1, 2))
    a_legs = ((Link(3, 1), ()), (Link(1, 3), (Link(3, 1),)))
    b_legs = ((Link(2, 1), ()), (Link(1, 2), (Link(2, 1),)))

    terms = []
    for sign, op in ((+1, ()), (-1, Qc)):
        for link, inner in a_legs:
            terms.append((link, op + inner, sign))
    for sign, op in ((-1, ()), (+1, Pc)):
        for link, inner in b_legs:
            terms.append((link, op + inner, sign))
    return terms


@pytest.fixture(scope="module")
def phases():
    """Fixed laser phases, so a failure is reproducible."""
    return np.random.default_rng(7).uniform(
        0.0, 2.0 * np.pi, size=(3, len(TONES)))


# -- the algebra, no arrays ----------------------------------------------


@pytest.mark.parametrize("channel", ["X", "Y", "Z"])
def test_term_structure(channel):
    """Sixteen terms, chains running 0 to 7 arms, two at each length.

    Second generation sends each beam through eight arms, so the deepest chain
    is the three round trips of the leading operator plus the outbound leg.
    A form that tops out at five arms is first generation wearing a hat, which
    is exactly the bug this catches.
    """
    terms = gen2_terms(channel)

    assert len(terms) == 16

    lengths = sorted(len(chain) for _, chain, _ in terms)
    assert lengths == sorted(list(range(8)) * 2)

    assert sum(1 for _, chain, _ in terms if chain) == 14


@pytest.mark.parametrize("channel", ["X", "Y", "Z"])
def test_signs_cancel_per_link(channel):
    """With the delays removed, every link's net coefficient is zero.

    Necessary, not sufficient: it is what makes the combination vanish when
    the arms are static, and it fails loudly on a sign typo. Exact integer
    arithmetic, so there is no tolerance to choose.
    """
    net = {}
    for link, _, sign in gen2_terms(channel):
        net[link] = net.get(link, 0) + sign

    assert set(net.values()) == {0}


def test_channels_use_all_six_links():
    """X, Y and Z between them draw on every directed link exactly once each."""
    used = {link for channel in "XYZ" for link, _, _ in gen2_terms(channel)}
    assert len(used) == 6


# -- the algebra, time domain, float64 only ------------------------------


def test_gen1_residual_is_first_order_in_the_drift(phases):
    """First generation leaves the commutator, so its residual scales as the rate.

    The contrast case. Fitting a slope rather than testing a number against a
    threshold means the numerical floor bounds how far down the sweep can go,
    not whether the result is meaningful.
    """
    t = np.arange(0.0, 5.0e5, 10.0)
    terms = _gen1_terms()

    rates = np.array([1e-9, 1e-8, 1e-7, 1e-6])
    residuals = np.array([
        np.sqrt(np.mean(_assemble(phases, terms, t, _laws(float(r))) ** 2))
        for r in rates
    ])

    slope = np.polyfit(np.log10(rates), np.log10(residuals), 1)[0]

    assert slope == pytest.approx(1.0, abs=0.02)


def test_gen2_cancels_a_linear_drift_to_machine_precision(phases):
    """Second generation kills the commutator outright for a linear law.

    Not merely second order: with the delays strictly linear in time the next
    order vanishes identically, so the residual sits at float64 and does not
    move over five decades of drift rate. Against first generation at the same
    rates this is a separation of five orders, which is what makes the test
    able to reject a wrong operator ordering.
    """
    t = np.arange(0.0, 5.0e5, 10.0)
    terms = gen2_terms("X")
    scale = np.sqrt(np.mean(_laser(phases, 0, t) ** 2))

    residuals = [
        np.sqrt(np.mean(_assemble(phases, terms, t, _laws(rate)) ** 2))
        for rate in (0.0, 1e-9, 1e-8, 1e-7, 1e-6)
    ]

    for residual in residuals:
        assert residual / scale < 1e-9

    # Flat, not merely small: no trend across five decades.
    assert max(residuals) / min(residuals) < 10.0


# -- the wavelet implementation of it ------------------------------------


def test_wavelet_tdi_matches_the_analytic_combination(phases):
    """The wavelet path reproduces the time-domain result it was checked against.

    Everything the shift operator is asked to do is exercised here - fourteen
    shifts per channel, chains up to seven arms - against a reference carrying
    no wavelet error at all.

    Note the delay is evaluated on ``(arange(Nt) - rows_left) * dT``. The grid
    is padded, so its row zero sits ``rows_left`` rows *before* the signal
    starts, and taking the delay on a grid that begins at zero applies a law
    offset by the whole guard band. That mistake is invisible in a
    self-consistency check: it saturates the residual at a value that responds
    to neither ``max_lag_L`` nor ``num_interp_points``.

    Two things set the tolerance, and both are about this test rather than the
    code. ``TONES`` are spaced linearly in frequency, so they land off bin
    centre and smear across rows; a bin-centred tone does about thirty times
    better. And :math:`N_t` has to be comfortably larger than the kernel: at
    :math:`N_t = 512` the :math:`2L+1` span is a fifth of the grid and
    :math:`L = 51` scores *worse* than :math:`L = 25`, which says more about
    the record length than the lag count. Off-bin content on an adequate grid
    is the harder and more honest case, so it is what is asserted here.
    """
    from scipy.signal.windows import tukey

    from WDM_LISAresponse.code.fastlisaresponse_comparison import config

    cfg = config.ComparisonConfig(N=2 ** 17, Nf=64, max_lag_L=51)
    wdm, response = cfg.build()

    times = cfg.signal_times
    epoch = float(times[0])
    laws = _laws(1e-8)
    terms = gen2_terms("X")

    pad_left = cfg.padding["pad_left"]
    pad_right = cfg.padding["pad_right"]
    taper = tukey(len(times), 0.01)

    def to_grid(series):
        padded = np.concatenate(
            [np.zeros(pad_left), series * taper, np.zeros(pad_right)])
        return np.asarray(wdm.dwt(padded))

    links = {lk: to_grid(_measurement(phases, lk, times - epoch, laws))
             for lk in laws}

    # Row zero of the grid is rows_left rows before the signal begins.
    grid_times = (np.arange(cfg.Nt) - cfg.padding["rows_left"]) * cfg.dT

    cache = {}
    got = np.zeros((cfg.Nt, cfg.Nf))
    for link, chain, sign in terms:
        grid = links[link]
        if chain:
            if chain not in cache:
                cache[chain] = _composed(chain, grid_times, laws)
            grid = np.asarray(wdm.apply_variable_time_shift(
                response.filter_tables, grid, -cache[chain]))
        got = got + sign * grid

    want = to_grid(_assemble(phases, terms, times - epoch, laws))

    interior = slice(int(0.15 * cfg.Nt), int(0.85 * cfg.Nt))
    scale = np.sqrt(np.mean(
        np.stack(list(links.values()))[:, interior] ** 2))
    residual = np.sqrt(np.mean((got - want)[interior] ** 2))

    assert residual / scale < 1e-5


# -- the two implementations against each other --------------------------


def test_wavelet_and_time_domain_tdi_agree_on_the_same_links():
    """The two TDI paths, fed identical link responses, give the same channels.

    Both routes start from one set of wavelet-domain link responses, so the
    local-delay approximation in the response is common to both and cancels.
    What is left is the delay application alone: the wavelet time-shift
    operator against twelve-tap Lagrange interpolation of a sampled series.

    The two share ``gen2_terms`` and ``composed_delay`` on purpose - an error
    in the algebra or the geometry would appear identically in both and cancel
    from this comparison, which is the point. They share nothing else.

    An upper bound rather than a clean isolation: the time-domain route also
    carries an ``idwt``/``dwt`` round trip that the wavelet route does not.
    """
    from WDM_LISAresponse.code.fastlisaresponse_comparison import (
        config, gb_waveform,
    )
    from WDM_LISAresponse.code.fastlisaresponse_comparison.tdi import TDIgen2
    from WDM_LISAresponse.code.fastlisaresponse_comparison.tdi_reference import (
        gen2_xyz_time_domain,
    )
    from WDM_LISAresponse.code.utils.utils import (
        apply_zero_padding, wdm_mismatch,
    )

    cfg = config.ComparisonConfig(N=2 ** 15, Nf=64)
    wdm, response = cfg.build()

    h_plus, h_cross = gb_waveform.from_config(cfg)
    _, h_plus, h_cross = apply_zero_padding(
        cfg.signal_times, h_plus, h_cross,
        cfg.padding["pad_left"], cfg.padding["pad_right"],
    )

    links = response(cfg.theta, cfg.phi, wdm.dwt(h_plus), wdm.dwt(h_cross))

    from_wavelet = np.asarray(TDIgen2(response, channels=("X",))(links))[0]

    from_time = gen2_xyz_time_domain(
        response, np.asarray(wdm.idwt(links)), channels=("X",),
    )
    from_time = np.asarray(wdm.dwt(from_time[0]))

    interior = slice(cfg.padding["rows_left"],
                     cfg.Nt - cfg.padding["rows_right"])
    result = wdm_mismatch(from_wavelet[interior], from_time[interior])

    assert result["rel_L2"] < 1e-6
    assert result["mismatch"] < 1e-12


# -- the reference path against a closed form ----------------------------


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

    from WDM_LISAresponse.code.fastlisaresponse_comparison import (
        config, gb_waveform,
    )
    from WDM_LISAresponse.code.lisa_response.response_sub import (
        build_link_geometry, compute_angular_geometry_td,
    )
    from WDM_LISAresponse.code.utils.utils import (
        apply_zero_padding, wdm_mismatch,
    )

    # Nf is left at the study default: m = 13 is then 5.078 mHz, 19.7 samples
    # per cycle. Halving Nf doubles dF and puts the same bin at 10.2 mHz,
    # where the quadrature and the interpolation are both an order worse and
    # this margin closes.
    cfg = config.ANALYTIC.variant(N=2 ** 17)
    wdm, response = cfg.build()

    h_plus, h_cross = gb_waveform.from_config(cfg)
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
            tukey(cfg.n_signal, gb_waveform.DEFAULT_TUKEY_ALPHA),
            np.zeros(cfg.padding["pad_right"]),
        ]),
    )

    mid_plus, mid_cross = gb_waveform.galactic_binary(
        (cfg.t0 + midpoint).ravel(), cfg.amp, cfg.f_gb, cfg.fdot, cfg.iota,
        cfg.phi0, cfg.psi, tukey_alpha=0.0, t_ref=float(cfg.signal_times[0]),
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
