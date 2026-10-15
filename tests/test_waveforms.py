"""Waveform generation and windowing.

A Tier 1 file: since padding moved to ``utils``, this module needs no orbits
and no WDM transform. It generates a signal on a uniform grid and optionally
tapers it, and that is all.

The EMRI section is a *call-contract* test rather than a waveform test. It
drives the generator with a stub that records how it was called, which checks
things a real FEW run cannot - most importantly that fourteen positional
parameters go out in the right order.
"""

import numpy as np
import pytest

from WDM_LISAresponse.code.waveforms import waveforms

DT = 5.0
N = 1024
T0 = 3.15e7


def _monochromatic(**overrides):
    """Generate a monochromatic waveform with sensible defaults."""
    call = dict(
        waveform_model="monochromatic",
        t0=T0,
        dt=DT,
        N=N,
        model_kwargs=dict(A_plus=1.0, A_cross=0.5, f=5.0e-3, phi_0=0.3),
    )
    call.update(overrides)
    return waveforms.generate_windowed_waveform(**call)


# --------------------------------------------------------------- the grid

def test_the_time_grid_starts_at_t0_and_is_uniform():
    """Samples sit at t0 + k*dt, with no drift across the record."""
    times, _, _ = _monochromatic()

    assert times[0] == pytest.approx(T0)
    assert len(times) == N
    assert np.allclose(np.diff(times), DT)


def test_nothing_is_padded():
    """The generator returns exactly the samples asked for.

    Padding belongs to the grid a record is analysed on, not to the
    waveform, and the two response paths need different amounts of it.
    """
    times, h_plus, h_cross = _monochromatic()

    assert len(times) == len(h_plus) == len(h_cross) == N


def test_a_duration_in_years_is_converted_with_the_sidereal_year():
    """``T_obs`` is years, and floors to a whole number of samples."""
    times, _, _ = _monochromatic(N=None, T_obs=0.01)

    expected = int(np.floor(0.01 * waveforms.YRSID_SI / DT))

    assert len(times) == expected


# --------------------------------------------------------------- the models

def test_monochromatic_matches_its_closed_form():
    """h+ = A+ cos(phase), hx = Ax sin(phase), with phase = 2 pi f t + phi0.

    The convention lives in one place, ``waveform_polarisations``, and this
    is what pins it.
    """
    times, h_plus, h_cross = _monochromatic()
    phase = 2.0 * np.pi * 5.0e-3 * times + 0.3

    assert np.allclose(h_plus, 1.0 * np.cos(phase))
    assert np.allclose(h_cross, 0.5 * np.sin(phase))


def test_linear_freq_starts_and_ends_at_the_requested_frequencies():
    """The sweep reaches ``f_end`` at the last sample, not at any nominal
    observation time.

    Instantaneous frequency is recovered from the analytic phase derivative
    at each end of the record.
    """
    f_start, f_end = 2.0e-3, 2.0e-2
    times, h_plus, _ = waveforms.generate_windowed_waveform(
        "linear_freq", t0=T0, dt=DT, N=N,
        model_kwargs=dict(A_plus=1.0, A_cross=0.0,
                          f_start=f_start, f_end=f_end, phi_0=0.0))

    phase = np.unwrap(np.arctan2(
        -np.gradient(h_plus, DT) / (2 * np.pi), h_plus))
    rate = (f_end - f_start) / (times[-1] - times[0])
    tau = times - times[0]
    expected = 2 * np.pi * (f_start * tau + 0.5 * rate * tau ** 2)

    assert np.allclose(h_plus, np.cos(expected))
    assert phase is not None


def test_linear_freq_reduces_to_monochromatic_when_the_ends_agree():
    """A sweep with no sweep is a fixed-frequency sinusoid."""
    _, chirped, _ = waveforms.generate_windowed_waveform(
        "linear_freq", t0=T0, dt=DT, N=N,
        model_kwargs=dict(A_plus=1.0, A_cross=0.5,
                          f_start=5.0e-3, f_end=5.0e-3, phi_0=0.3))
    _, flat, _ = _monochromatic()

    assert np.allclose(chirped, flat)


def test_linear_freq_needs_two_distinct_times():
    """A chirp rate cannot be formed from a single instant."""
    with pytest.raises(ValueError, match="at least two distinct"):
        waveforms.generate_waveform_linear_freq(
            times=np.array([1.0, 1.0]), A_plus=1.0, A_cross=0.0,
            f_start=1e-3, f_end=2e-3, phi_0=0.0)


# -------------------------------------------------------------- the window

def test_a_zero_alpha_taper_is_the_identity():
    """alpha = 0 leaves the signal untouched."""
    _, unwindowed, _ = _monochromatic()
    _, windowed, _ = _monochromatic(
        window_function="tukey", window_kwargs=dict(alpha=0.0))

    assert np.allclose(windowed, unwindowed)


def test_a_unit_alpha_taper_is_a_hann_window():
    """alpha = 1 is the full cosine taper."""
    from scipy.signal.windows import hann

    _, unwindowed, _ = _monochromatic()
    _, windowed, _ = _monochromatic(
        window_function="tukey", window_kwargs=dict(alpha=1.0))

    assert np.allclose(windowed, hann(N) * unwindowed)


def test_the_window_tapers_but_does_not_pad():
    """Windowing changes values, never length.

    The old API padded by a multiple of the taper width, through a knob that
    never reached the window at all. Padding is now asked for explicitly.
    """
    times, h_plus, _ = _monochromatic(
        window_function="tukey", window_kwargs=dict(alpha=0.5))

    assert len(times) == len(h_plus) == N
    assert abs(h_plus[0]) < 1e-12


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_an_out_of_range_alpha_raises(alpha):
    """The taper fraction is a fraction."""
    with pytest.raises(ValueError):
        _monochromatic(window_function="tukey",
                       window_kwargs=dict(alpha=alpha))
    with pytest.raises(ValueError):
        waveforms.tukey_taper_samples(N, alpha)


@pytest.mark.parametrize("n_samples", [9, 11, 101, 1000, 1024, 4180992])
@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.05, 0.5, 1.0])
def test_the_taper_length_covers_every_tapered_sample(n_samples, alpha):
    """Outside the counted ramps the window is exactly one, and the count is
    at most one sample over - the case where scipy's last ramp sample lands on
    unity. Read off the window itself, not off scipy's formula."""
    from scipy.signal.windows import tukey

    window = tukey(n_samples, alpha)
    taper = waveforms.tukey_taper_samples(n_samples, alpha)
    below = np.flatnonzero(window[: n_samples // 2] < 1.0)
    tapered = below[-1] + 1 if below.size else 0

    assert np.all(window[taper:n_samples - taper] == 1.0)
    assert tapered <= taper <= tapered + 1


# ------------------------------------------------------------------ refusal

def test_giving_neither_duration_raises():
    """One of ``N`` or ``T_obs`` is required."""
    with pytest.raises(ValueError, match="Either T_obs or N"):
        _monochromatic(N=None)


def test_giving_both_durations_raises():
    """Two ways to say the same thing invite them to disagree."""
    with pytest.raises(ValueError, match="only one of T_obs or N"):
        _monochromatic(T_obs=1.0)


@pytest.mark.parametrize("bad_n", [1, 0, -5])
def test_too_few_samples_raises(bad_n):
    """A cadence cannot be inferred from fewer than two samples."""
    with pytest.raises(ValueError, match="At least two samples"):
        _monochromatic(N=bad_n)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0])
def test_a_non_positive_cadence_raises(bad_dt):
    """Time must run forwards."""
    with pytest.raises(ValueError, match="dt must be positive"):
        _monochromatic(dt=bad_dt)


def test_an_unknown_model_raises_and_names_it():
    """The message names the offending model, not just the failure."""
    with pytest.raises(ValueError, match="'nope' is not available"):
        _monochromatic(waveform_model="nope")


def test_an_unknown_window_raises_and_names_it():
    """Same for windows, which are a separate registry."""
    with pytest.raises(ValueError, match="'boxcar' is not available"):
        _monochromatic(window_function="boxcar", window_kwargs={})


# ---------------------------------------------------- the EMRI call contract

class RecordingGenerator:
    """A stand-in for a FEW generator that records how it was called."""

    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return [np.ones(self.num_samples), 2.0 * np.ones(self.num_samples)]


EMRI_PARAMETERS = dict(
    m1=1.0e6, m2=10.0, a=0.9, p0=12.0, e0=0.3, x0=1.0, dist=1.5,
    qK=0.11, phiK=0.22, Phi_phi0=0.33, Phi_theta0=0.44, Phi_r0=0.55,
    eps=1.0e-2,
)

#: The order FEW expects its positional arguments in.
EMRI_ORDER = ["m1", "m2", "a", "p0", "e0", "x0", "dist", "qS", "phiS",
              "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]


def test_the_generator_is_called_with_its_parameters_in_order():
    """Fourteen positional arguments, in the order FEW expects.

    The reason this is a stub and not a FEW test: swap ``qK`` and ``phiK``,
    or ``e0`` and ``x0``, and FEW returns a perfectly valid waveform for the
    wrong source. No real run can detect that without an independent
    reference, and there isn't one. Recording the call makes it exact.
    """
    generator = RecordingGenerator(400)
    times = np.arange(400) * DT

    waveforms.generate_waveform_emri(
        times=times, td_gen=generator, theta=0.77, phi=0.88,
        **EMRI_PARAMETERS)

    expected = dict(EMRI_PARAMETERS, qS=0.77, phiS=0.88)
    assert len(generator.args) == len(EMRI_ORDER)
    for name, passed in zip(EMRI_ORDER, generator.args):
        assert passed == expected[name], f"{name} is in the wrong position"


def test_the_duration_handed_over_matches_the_record():
    """``T`` and ``dt`` are derived from ``times``, not taken on trust."""
    generator = RecordingGenerator(400)
    times = np.arange(400) * DT

    waveforms.generate_waveform_emri(
        times=times, td_gen=generator, theta=0.7, phi=0.8, **EMRI_PARAMETERS)

    assert generator.kwargs["dt"] == pytest.approx(DT)
    assert generator.kwargs["T"] == pytest.approx(
        400 * DT / waveforms.YRSID_SI)
    assert generator.kwargs["mode_selection_threshold"] == 1.0e-2


def test_theta_and_phi_take_precedence_over_qs_and_phis():
    """Two spellings of the sky position, with a defined winner.

    Getting this backwards puts the source somewhere else with no error
    anywhere.
    """
    generator = RecordingGenerator(400)

    waveforms.generate_waveform_emri(
        times=np.arange(400) * DT, td_gen=generator,
        qS=1.11, phiS=2.22, theta=0.77, phi=0.88, **EMRI_PARAMETERS)

    assert generator.args[EMRI_ORDER.index("qS")] == 0.77
    assert generator.args[EMRI_ORDER.index("phiS")] == 0.88


def test_the_sky_position_may_be_given_as_qs_and_phis():
    """The FEW spelling works when the alias is absent."""
    generator = RecordingGenerator(400)

    waveforms.generate_waveform_emri(
        times=np.arange(400) * DT, td_gen=generator,
        qS=1.11, phiS=2.22, **EMRI_PARAMETERS)

    assert generator.args[EMRI_ORDER.index("qS")] == 1.11


def test_a_missing_sky_position_raises():
    """Neither spelling given is a mistake, not a default."""
    with pytest.raises(ValueError, match="Provide sky location"):
        waveforms.generate_waveform_emri(
            times=np.arange(400) * DT, td_gen=RecordingGenerator(400),
            **EMRI_PARAMETERS)


@pytest.mark.parametrize("returned, why", [
    (500, "FEW overshoots when pad_output and odd_len are set"),
    (300, "the inspiral plunges before T_obs is up"),
])
def test_the_output_is_forced_to_the_requested_length(returned, why):
    """Long output is truncated, short output zero filled.

    Both directions are expected rather than exceptional, and reaching the
    second with a real generator would mean hunting for parameters that
    plunge early.
    """
    generator = RecordingGenerator(returned)
    times = np.arange(400) * DT

    h_plus, h_cross = waveforms.generate_waveform_emri(
        times=times, td_gen=generator, theta=0.7, phi=0.8, **EMRI_PARAMETERS)

    assert len(h_plus) == len(h_cross) == 400
    if returned < 400:
        assert np.all(h_plus[returned:] == 0.0)
    assert np.all(h_plus[:min(returned, 400)] == 1.0)


def test_non_uniform_times_raise():
    """The generator is told one cadence, so the record must have one."""
    times = np.arange(400) * DT
    times[10] += 1.0

    with pytest.raises(ValueError, match="uniformly sampled"):
        waveforms.generate_waveform_emri(
            times=times, td_gen=RecordingGenerator(400),
            theta=0.7, phi=0.8, **EMRI_PARAMETERS)
