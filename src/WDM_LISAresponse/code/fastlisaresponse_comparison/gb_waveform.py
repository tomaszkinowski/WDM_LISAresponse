r"""The galactic binary, defined once for both pipelines.

There is one waveform implementation in this study, and both codes are fed the
*same array* built from it - ``conventions.PaddedReplay`` hands
``fastlisaresponse`` the tapered, zero-padded record our own response receives,
rather than letting it generate its own. So the waveform conventions here - the
overall sign, the amplitude factor, the polarisation rotation - cannot differ
between the two, because neither code ever sees a second version.

That matters more than it first appears. Our response is not fed a raw binary
but a *windowed and padded* one, and a generator handed to ``ResponseWrapper``
would produce full-amplitude signal exactly where our record holds zeros. An
earlier version of this module exported such a generator; replaying the array
is what actually closes the gap.

Nothing here imports ``fastlisaresponse``.
"""

import numpy as np

from ..waveforms.waveforms import apply_tukey_window


#: Tukey fraction applied by default. Measured on the default tiling: an
#: untapered record leaks :math:`4\times10^{-5}` of its power into far
#: frequency rows, which is :math:`6\times10^{-3}` in amplitude and so above
#: the :math:`1.4\times10^{-3}` that a :math:`10^{-6}` mismatch allows. At
#: 0.01 that falls to :math:`9\times10^{-12}`, for 1% of the record - against
#: a signal whose science is a one-year modulation.
DEFAULT_TUKEY_ALPHA = 0.01


def galactic_binary(t,
                    amp,
                    f,
                    fdot,
                    iota,
                    phi0,
                    psi,
                    tukey_alpha: float = DEFAULT_TUKEY_ALPHA,
                    t_ref: float = None):
    r"""Generate a galactic binary in the SSB frame.

    .. math::

        \Phi(t) = 2\pi \left[ f \tau
            + \tfrac{1}{2} \dot{f} \tau^2 \right] + \phi_0 ,
        \qquad \tau = t - t_{\mathrm{ref}} ,

    with source-frame polarisations
    :math:`h^S_+ = A(1 + \cos^2\iota)\cos\Phi` and
    :math:`h^S_\times = 2 A \cos\iota \sin\Phi`, rotated into the SSB frame by
    :math:`\psi`:

    .. math::

        h_+ = h^S_+ \cos 2\psi - h^S_\times \sin 2\psi , \qquad
        h_\times = h^S_+ \sin 2\psi + h^S_\times \cos 2\psi .

    Parameters
    ----------
    t : np.ndarray
        Sample times, in seconds. Absolute, on the same timeline as the
        response's ``t0``. Array shape=(n,).
    amp : float
        Overall strain amplitude.
    f : float
        Frequency at :math:`t_{\mathrm{ref}}`, in Hz.
    fdot : float
        Frequency derivative, in Hz/s.
    iota : float
        Inclination, in radians.
    phi0 : float
        Phase at :math:`t_{\mathrm{ref}}`, in radians.
    psi : float
        Polarisation angle, in radians.
    tukey_alpha : float
        Tukey fraction, in :math:`[0, 1]`. Zero leaves the record
        untapered. Optional.
    t_ref : float
        Phase origin, in seconds: the absolute time at which the phase is
        ``phi0``. Defaults to ``t[0]``. Optional. See ``Notes``.

    Returns
    -------
    h_plus : np.ndarray
        Plus polarisation. Array shape=(n,).
    h_cross : np.ndarray
        Cross polarisation. Array shape=(n,).

    Notes
    -----
    ``t_ref`` is an argument rather than an implied zero because the study runs
    at a non-zero epoch. At :math:`t_0 = 3.15\times10^7` s the first sample
    already carries :math:`2\pi f t_0 \approx 10^6` rad, and a generator that
    silently measured phase from zero would disagree with one that did not by
    that much.

    It cannot be left to the alignment step to absorb. A phase measurement
    recovers an offset only modulo :math:`1/f`, here 197 s, whereas the
    antenna-pattern envelope this comparison is built on varies over a year.
    A gross origin error would wrap to a harmless-looking sub-197-second
    offset in the carrier while leaving the modulation misaligned, and nothing
    downstream would say so. Stating ``t_ref`` anchors the carrier and the
    envelope to one timeline.

    The :math:`\dot{f}` term is insensitive to the choice: shifting the origin
    by :math:`3\times10^{7}` s moves the frequency by
    :math:`3\times10^{-9}` Hz, five orders below :math:`\Delta F`. The
    sensitivity is all in the linear phase.

    The overall sign and the amplitude convention - whether the
    :math:`(1 + \cos^2\iota)` carries a half - multiply both polarisations
    equally, so ``utils.wdm_mismatch`` normalises them away and neither can
    affect the comparison. The :math:`\psi` rotation is different: it mixes the
    two, and a sign error there survives. It is applied here rather than by the
    response, matching where ``fastlisaresponse`` puts it.
    """
    t = np.asarray(t, dtype=float)

    if t.ndim != 1 or t.size < 2:
        raise ValueError(
            f"t must be a 1D array with at least two samples, got shape "
            f"{t.shape}."
        )

    origin = float(t[0]) if t_ref is None else float(t_ref)
    tau = t - origin

    phase = 2.0 * np.pi * (f * tau + 0.5 * fdot * tau * tau) + phi0

    cos_iota = np.cos(iota)
    h_source_plus = amp * (1.0 + cos_iota * cos_iota) * np.cos(phase)
    h_source_cross = 2.0 * amp * cos_iota * np.sin(phase)

    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)

    h_plus = h_source_plus * cos_2psi - h_source_cross * sin_2psi
    h_cross = h_source_plus * sin_2psi + h_source_cross * cos_2psi

    if tukey_alpha > 0.0:
        h_plus, h_cross = apply_tukey_window(h_plus, h_cross, tukey_alpha)

    return h_plus, h_cross


def from_config(cfg, **overrides):
    """Generate the binary described by a ``ComparisonConfig``.

    Parameters
    ----------
    cfg : ComparisonConfig
        Supplies the sample grid and every source parameter.
    **overrides
        Any ``galactic_binary`` keyword to override.

    Returns
    -------
    h_plus, h_cross : np.ndarray
        Polarisations on ``cfg.signal_times``. Array shape=(n_signal,).
    """
    times = cfg.signal_times

    kwargs = dict(
        amp=cfg.amp,
        f=cfg.f_gb,
        fdot=cfg.fdot,
        iota=cfg.iota,
        phi0=cfg.phi0,
        psi=cfg.psi,
        t_ref=float(times[0]),
    )
    kwargs.update(overrides)

    return galactic_binary(times, **kwargs)
