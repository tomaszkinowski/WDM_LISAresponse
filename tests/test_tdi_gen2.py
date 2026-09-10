"""Second-generation TDI: the algebra, then the wavelet implementation of it.

The tests are staged so a failure says *which* layer broke.

``gen2_terms`` is pure combinatorics, so the first two tests need no arrays at
all. The next two evaluate the whole combination in the time domain from
closed-form laser noises and a closed-form linear delay law - no transform, no
shift operator, no interpolation, so the only error is float64 and any
residual is the algebra's own. The last two bring in the wavelet machinery,
against the time-domain result those tests have just validated.

Nothing here touches the response's own physics: the link measurements are
either closed-form laser noise, or the same wavelet link responses fed to both
routes so that any response error is common and cancels. What validates the
links themselves is
``test_time_domain_path.test_time_domain_response_matches_the_monochromatic_closed_form``.
"""

import numpy as np
import pytest

from WDM_LISAresponse.code.comparison.tdi import gen2_terms
from WDM_LISAresponse.code.utils.utils import Link

#: One arm per spacecraft pair. Channel X uses P and Q; Y and Z bring in R,
ARM_P = frozenset({Link(3, 1), Link(1, 3)})
ARM_Q = frozenset({Link(2, 1), Link(1, 2)})
ARM_R = frozenset({Link(3, 2), Link(2, 3)})

#: Distinct arm lengths, so no two arms have any reason to commute.
L_P, L_Q, L_R = 8.339, 8.351, 8.347

#: Breathing rates relative to P, so one parameter drives the constellation.
#: Distinct and of mixed sign, so it is not a uniform scaling.
RATE_Q_OVER_P = -0.7
RATE_R_OVER_P = 0.4

TONES = np.linspace(2e-3, 8e-3, 9)


# -- analytic scaffolding ------------------------------------------------


def _laser(phases, i, t):
    """Laser noise of spacecraft ``i`` at times ``t``. Closed form."""
    out = np.zeros_like(t)
    for n, f in enumerate(TONES):
        out += np.cos(2.0 * np.pi * f * t + phases[i, n])
    return out


def _laws(rate):
    """link -> l(t), a linear delay law per arm, over all six directed links."""
    arms = {ARM_P: (L_P, rate),
            ARM_Q: (L_Q, rate * RATE_Q_OVER_P),
            ARM_R: (L_R, rate * RATE_R_OVER_P)}

    def one(base, drift):
        return lambda t: base + drift * t

    return {lk: one(base, drift)
            for arm, (base, drift) in arms.items() for lk in arm}


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
    lengths = sorted(len(chain) for _, chain, _ in gen2_terms(channel))

    # One assertion, not three: this multiset has sixteen entries of which
    # exactly two are empty, so a term count and a "fourteen have chains"
    # check would both be consequences of it and could never fail alone.
    assert lengths == sorted(list(range(8)) * 2)


def test_channels_between_them_use_all_six_links():
    """No channel is a copy of another: the union covers every directed link.

    Each channel draws on four of the six directed links, over its sixteen
    terms. Nothing else here would notice if Z came back as a duplicate of Y.
    """
    used = {link for channel in "XYZ" for link, _, _ in gen2_terms(channel)}
    assert len(used) == 6


# -- the algebra, time domain, float64 only ------------------------------


def test_gen1_residual_is_first_order_in_the_drift(phases):
    """First generation leaves the commutator, so its residual scales as the rate.

    The control for the test below. Without it, second generation cancelling
    would be just as consistent with a harness that returns zero whatever it is
    given; this shows the same machinery does produce a residual when the
    algebra cannot cancel one.

    A slope is fitted over three decades rather than a number tested against a
    threshold. The residual has no natural scale to compare against, but its
    *exponent* does: :math:`X_1 = [P, Q]p` is first order in the rate at which
    the arms change, so the slope must be one. Measured, the residual runs
    :math:`1.9\\times10^{-9}` to :math:`1.9\\times10^{-6}` across the sweep.
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


@pytest.mark.parametrize("channel", ["X", "Y", "Z"])
def test_gen2_cancels_a_linear_drift_to_machine_precision(phases, channel):
    """Second generation kills the commutator outright for a linear law.

    Five drift rates are swept - zero, then :math:`10^{-9}` to :math:`10^{-6}`
    s/s, three decades - and the residual is required to be both small and
    *flat*. Flatness is the sharper of the two: a merely second-order scheme
    would still trend across the sweep, whereas with the delays strictly linear
    in time the next order vanishes identically, so nothing is left to scale
    and the residual sits on the float64 floor. Measured, it holds at
    :math:`3\\times10^{-12}` of laser scale at every rate, against first
    generation rising from :math:`1.9\\times10^{-9}` to
    :math:`1.9\\times10^{-6}` over the same sweep. That is what lets this
    reject a wrong operator ordering.

    ``rate = 0.0`` earns its place separately: it is the static-arm case, where
    cancellation is down to the per-link signs alone and first generation is
    just as good. A flipped sign cannot cancel there - measured, any single
    flip lifts the residual to about 2.5, twelve orders above the threshold -
    which is why no separate sign test is needed. Running all three channels is
    what lets that stand in for one, so do not narrow it back to X.
    """
    t = np.arange(0.0, 5.0e5, 10.0)
    terms = gen2_terms(channel)
    scale = np.sqrt(np.mean(_laser(phases, 0, t) ** 2))

    residuals = [
        np.sqrt(np.mean(_assemble(phases, terms, t, _laws(rate)) ** 2))
        for rate in (0.0, 1e-9, 1e-8, 1e-7, 1e-6)
    ]

    for residual in residuals:
        assert residual / scale < 1e-9

    # Flat, not merely small: no trend across the three decades swept.
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

    from WDM_LISAresponse.code.comparison import config

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
    from WDM_LISAresponse.code.comparison import config
    from WDM_LISAresponse.code.comparison.tdi import TDIgen2
    from WDM_LISAresponse.code.comparison.tdi_reference import (
        gen2_xyz_time_domain,
    )
    from WDM_LISAresponse.code.utils.utils import (
        apply_zero_padding, wdm_mismatch,
    )

    cfg = config.ComparisonConfig(N=2 ** 15, Nf=64)
    wdm, response = cfg.build()

    h_plus, h_cross = cfg.waveform()
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
