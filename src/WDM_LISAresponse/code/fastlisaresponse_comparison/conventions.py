r"""Everything that differs between the two codes, measured rather than assumed.

Each constant and conversion below was established by running both pipelines
and looking at the result, and the evidence is recorded beside it. That is the
point of collecting them here: when a residual comes back wrong months from
now, this is the file to read, and a value with no provenance is worse than no
value at all.

The one that matters is the variable. ``fastlisaresponse`` returns a
*fractional frequency*, this package returns a *delay in seconds*, and

.. math::

    \mathrm{theirs} = -\frac{\mathrm{d}}{\mathrm{d}t}\,\mathrm{ours} .

Nothing else needed correcting: the sky mapping is latitude against polar
angle with no frame rotation, the two sample grids differ by a whole number of
samples, and no scale or time offset has to be fitted.

Nothing here imports ``fastlisaresponse`` or ``lisatools``. The orbit provider
and the response wrapper are constructed by the caller and passed in, as
``waveforms`` does for FEW, so this package stays installable without either.

Environment note
----------------
On macOS ``lisatools`` and ``fastlisaresponse`` each ship their own copy of
``libstdc++.6.dylib``. Loading both into one process aborts with
``Abort trap: 6`` - on import, or later inside ``pyResponseTDI`` - because two
images of the same C++ runtime carry two sets of globals. Point one at the
other and dyld loads a single copy::

    cd .../site-packages/fastlisaresponse/.dylibs
    mv libstdc++.6.dylib libstdc++.6.dylib.bak
    ln -s ../../lisatools/.dylibs/libstdc++.6.dylib libstdc++.6.dylib

It does not survive reinstalling either package.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

from ..utils.utils import wdm_mismatch
from ..waveforms.waveforms import YRSID_SI


#: The runtime that actually clashes. ``libgcc_s.1.1.dylib`` is vendored twice
#: as well, and is deliberately not listed: a working environment - one where
#: only ``libstdc++`` has been pointed at a single copy - still carries two
#: distinct copies of it and runs the whole comparison without trouble.
#: Flagging it would fail a setup that demonstrably works, which is worse than
#: not checking at all.
_VENDORED = ("libstdc++.6.dylib",)


def check_environment(raise_on_problem: bool = True) -> str:
    r"""Check that ``lisatools`` and ``fastlisaresponse`` can share a process.

    On macOS the two ship their own copies of the GNU C++ runtime. Two images
    of it in one process carry two sets of globals, and the result is
    ``Abort trap: 6`` - on import, or later inside ``pyResponseTDI``.

    That abort is a ``SIGABRT``, so it cannot be caught and cannot be
    explained after the fact: the interpreter is simply gone, with no
    traceback. The only useful thing is to look for the condition *before*
    anything imports the extension, which is what this does. Call it first,
    ahead of ``fastlisaresponse``.

    Nothing is imported here and nothing is modified - the check is two file
    hashes.

    Parameters
    ----------
    raise_on_problem : bool
        Raise when a clash is found. Set False to get the report as a string
        and decide for yourself. Optional.

    Returns
    -------
    report : str
        What was found, and the fix if one is needed.

    Raises
    ------
    RuntimeError
        A clash was found and ``raise_on_problem`` is set.
    """
    if sys.platform != "darwin":
        return "not macOS: the vendored-runtime clash does not arise"

    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}." \
        f"{sys.version_info.minor}" / "site-packages"

    clashes = []
    for name in _VENDORED:
        ours = site / "fastlisaresponse" / ".dylibs" / name
        theirs = site / "lisatools" / ".dylibs" / name

        if not (ours.is_file() and theirs.is_file()):
            continue
        if ours.resolve() == theirs.resolve():
            continue

        digest = [hashlib.sha256(p.read_bytes()).hexdigest()[:12]
                  for p in (ours, theirs)]
        if digest[0] != digest[1]:
            clashes.append((name, ours, theirs))

    if not clashes:
        return "no vendored-runtime clash: the two packages can share a process"

    lines = [
        "lisatools and fastlisaresponse vendor separate copies of:",
        *(f"    {name}" for name, _, _ in clashes),
        "",
        "Loading both aborts the interpreter with 'Abort trap: 6', which",
        "cannot be caught. Point one at the other so dyld loads one copy:",
        "",
        f"    cd {clashes[0][1].parent}",
    ]
    for name, _, theirs in clashes:
        lines += [f"    mv {name} {name}.bak",
                  f"    ln -s {theirs} {name}"]
    lines += ["", "This does not survive reinstalling either package."]

    report = "\n".join(lines)

    if raise_on_problem:
        raise RuntimeError(report)

    return report


#: ``ResponseWrapper`` multiplies the cross polarisation by -1 when set. Our
#: waveform is handed over as :math:`h_+ + i h_\times`, which is what
#: ``pyResponseTDI`` expects, so no flip. Verified: with it False the channel
#: amplitudes agree to 1e-4 of unity and the mismatch is 1e-9, neither of
#: which survives a sign error on the cross polarisation.
FLIP_HX = False

#: Positions of the ecliptic longitude and latitude in the argument list
#: handed to the wrapper.
INDEX_LAMBDA = 6
INDEX_BETA = 7

#: Seconds ``ResponseWrapper`` treats as unusable at *each* end. With
#: ``REMOVE_GARBAGE`` set to ``"zero"`` those samples are zeroed rather than
#: deleted, so the record keeps its length. At 10 s that is 1000 samples,
#: comfortably inside the guard band ``config`` already provides - measured,
#: the returned channels are nonzero exactly on ``[1000, n-1001]``.
FLR_T0_BUFFER = 10000.0

#: ``"zero"`` rather than ``True``: it preserves the array length, which keeps
#: the two grids related by an integer sample offset instead of a slice whose
#: endpoints have to be tracked.
REMOVE_GARBAGE = "zero"

#: Second-generation TDI, Michelson channels.
TDI_GENERATION = "2nd generation"
TDI_CHANNELS = "XYZ"


def grid_offset(cfg) -> int:
    r"""Samples between the start of their record and the start of ours.

    ``lisatools`` rebases an orbit file's time axis to zero, so
    ``ResponseWrapper``'s first sample sits at the file's own epoch, while this
    package anchors its padded record ``ORBIT_LEAD_IN`` later. The two grids
    therefore differ by a fixed shift, and at the default tiling it is exactly
    128 samples.

    Raises if that shift is not a whole number of samples. It would then take
    an interpolation to bridge the two grids, and the systematic that
    introduces goes straight into the timing budget - 44 ms at 5.078 mHz -
    without anything failing.

    Parameters
    ----------
    cfg : ComparisonConfig
        Must be on the ESA orbits, since the analytic provider has no file
        epoch to measure against.

    Returns
    -------
    offset : int
        Whole samples from their first sample to ours.
    """
    if cfg.orbits_model != "esa":
        raise ValueError(
            "grid_offset needs an orbit file to define the epoch; build the "
            "configuration with ComparisonConfig.for_esa()."
        )

    exact = (cfg.t0 - cfg.orbit_span()[0]) / cfg.dt
    offset = int(round(exact))

    if abs(exact - offset) > 1e-9:
        raise ValueError(
            f"The two sample grids are offset by {exact:.6f} samples, which "
            f"is not an integer. Bridging them would need interpolation. Make "
            f"ORBIT_LEAD_IN a whole multiple of dt."
        )

    return offset


def to_flr_angles(theta, phi):
    r"""Sky position in this package's convention to ``fastlisaresponse``'s.

    Ecliptic longitude and latitude, :math:`\lambda = \phi` and
    :math:`\beta = \pi/2 - \theta`.

    **No frame rotation**, and that is worth a sentence because it is only
    true by construction. ``lisaorbits`` works in equatorial coordinates and
    ``lisatools.detector`` in ecliptic; their analytic equal-armlength models
    differ by the obliquity, :math:`23.4393^\circ`, or
    :math:`6\times10^{10}` m. Both codes here read the *same orbit file* and
    neither converts on load, so both live in the file's frame and the mapping
    is the latitude convention alone. Comparing against the analytic provider
    instead would need the rotation put back.

    Parameters
    ----------
    theta, phi : float
        Polar and azimuthal angle, as ``build_sky_basis`` defines them.

    Returns
    -------
    lam, beta : float
        Ecliptic longitude and latitude, in radians, for
        ``is_ecliptic_latitude=True``.
    """
    return float(phi), float(np.pi / 2.0 - theta)


def align_to_our_grid(series, cfg, offset: int = None) -> np.ndarray:
    """Slice a ``fastlisaresponse`` record onto this package's padded grid.

    Pure integer indexing - see ``grid_offset``.

    Parameters
    ----------
    series : np.ndarray
        Their output, shape=(..., n) with ``n >= offset + cfg.N``.
    cfg : ComparisonConfig
        Supplies the grid.
    offset : int
        Sample offset; computed from ``cfg`` when omitted. Optional.

    Returns
    -------
    aligned : np.ndarray
        Array shape=(..., cfg.N), on ``cfg.padded_times``.
    """
    series = np.asarray(series, dtype=float)
    offset = grid_offset(cfg) if offset is None else int(offset)

    if series.shape[-1] < offset + cfg.N:
        raise ValueError(
            f"Their record is {series.shape[-1]} samples, short of the "
            f"{offset + cfg.N} needed to cover ours. Build the wrapper with "
            f"flr_wrapper_kwargs, which sizes Tobs for this."
        )

    return series[..., offset:offset + cfg.N]


def to_wdm_units(series, cfg) -> np.ndarray:
    r"""Convert a ``fastlisaresponse`` channel to this package's variable.

    They carry a fractional frequency and this package carries a delay in
    seconds, related by

    .. math::

        \mathrm{theirs} = -\frac{\mathrm{d}}{\mathrm{d}t}\,\mathrm{ours} ,

    so the conversion is a negated time integral, taken spectrally: exact for
    a band-limited record, and with the zero-frequency bin dropped, which
    costs nothing since the source sits at :math:`m = 13` and the integration
    constant is unobservable anyway.

    Applied to *their* output and never to ours, so the quantity under test is
    not touched by the thing that makes it comparable.

    Established three ways rather than assumed. The raw amplitude ratio came
    out at 0.0319 on all three channels against
    :math:`2\pi f = 0.031906`. Correlating against a finite-difference
    :math:`-\mathrm{d}/\mathrm{d}t` of our own channels gave
    :math:`-0.999999998`. And the 1.7% residual scale in that check is
    explained exactly by the central difference itself, which underestimates a
    sinusoid's derivative by :math:`\sin(x)/x = 0.98311` at
    :math:`x = 2\pi f \delta t`, predicting 1.01718 against 1.01714 measured -
    so there is no leftover factor, only the derivative.

    Parameters
    ----------
    series : np.ndarray
        Their channels on this package's grid, shape=(..., cfg.N).
    cfg : ComparisonConfig
        Supplies ``N`` and ``dt``.

    Returns
    -------
    converted : np.ndarray
        The same channels as a delay in seconds. Array shape=(..., cfg.N).
    """
    series = np.asarray(series, dtype=float)

    if series.shape[-1] != cfg.N:
        raise ValueError(
            f"Expected {cfg.N} samples on the padded grid, got "
            f"{series.shape[-1]}. Run align_to_our_grid first."
        )

    frequencies = np.fft.rfftfreq(cfg.N, d=cfg.dt)

    # 1/(2 pi i f), with the unobservable constant term dropped.
    weight = np.zeros_like(frequencies, dtype=complex)
    weight[1:] = 1.0 / (2.0j * np.pi * frequencies[1:])

    return np.fft.irfft(-np.fft.rfft(series, axis=-1) * weight, n=cfg.N,
                        axis=-1)


def convert(series, cfg, wdm, offset: int = None) -> np.ndarray:
    """Their raw output to wavelet grids on this package's tiling.

    ``align_to_our_grid``, then ``to_wdm_units``, then the transform - the
    whole bridge in one call, so a notebook cannot do two of the three.

    Parameters
    ----------
    series : np.ndarray
        Their channels, shape=(num_channels, n).
    cfg : ComparisonConfig
        Supplies the grid.
    wdm : WDM.WDM_transform
        The transform, from ``cfg.build()``.
    offset : int
        Sample offset; computed when omitted. Optional.

    Returns
    -------
    grids : np.ndarray
        Array shape=(num_channels, Nt, Nf), in seconds.
    """
    converted = to_wdm_units(align_to_our_grid(series, cfg, offset), cfg)

    return np.stack([np.asarray(wdm.dwt(row)) for row in converted])


class PaddedReplay:
    r"""Hands ``ResponseWrapper`` the record this package's response received.

    ``ResponseWrapper`` calls a waveform generator rather than taking an
    array, and would otherwise have one built on its own grid: untapered,
    unpadded, and carrying full-amplitude signal exactly where our record
    holds zeros. This replays our array instead, so both responses see
    bit-identical input and the waveform cannot contribute to the residual.

    Replaying rather than regenerating is deliberate. The taper is
    ``scipy.signal.windows.tukey`` over the signal's *sample indices*;
    re-deriving it as a function of absolute time risks disagreeing with
    scipy at the window edges, at the level the comparison is trying to
    measure. Because the two grids differ by a whole number of samples,
    indexing one array is exact where re-evaluating a window is not.

    Parameters
    ----------
    h_plus, h_cross : np.ndarray
        The padded, tapered polarisations this package's response was given.
        Array shape=(cfg.N,).
    offset : int
        Samples from their first sample to ours, from ``grid_offset``.

    Attributes
    ----------
    record : np.ndarray
        The full complex record on their grid, zero before ``offset``.
    """

    def __init__(self, h_plus, h_cross, offset: int):
        h_plus = np.asarray(h_plus, dtype=float)
        h_cross = np.asarray(h_cross, dtype=float)

        if h_plus.shape != h_cross.shape:
            raise ValueError(
                f"Polarisations differ in shape: {h_plus.shape} against "
                f"{h_cross.shape}."
            )

        self.offset = int(offset)
        self.record = np.zeros(self.offset + h_plus.size, dtype=complex)
        self.record[self.offset:] = h_plus + 1j * h_cross

    def __call__(self, *args, T: float = 1.0, dt: float = 10.0, **kwargs):
        r"""Return :math:`h_+ + i h_\times` over ``T`` years at cadence ``dt``.

        Source parameters are accepted and ignored: the record is already
        built, and taking them keeps the call signature the wrapper expects.

        The sample count is *rounded*. ``ResponseWrapper`` fixes
        :math:`n = \lfloor T Y / \delta t \rfloor`, converts back with
        :math:`T = n \delta t / Y` and passes that here, so the quotient lands
        on an integer to within floating-point noise - and it carries its own
        sidereal year, 31558149.763545603 s against the ...599 s
        ``lisaconstants`` gives, one ULP apart and enough to drop it just
        below. Truncating returns one sample too few and
        ``get_projections`` fails a bare assertion with nothing to say why.

        Returns
        -------
        h : np.ndarray
            Array, dtype=complex128, shape=(round(T * YRSID_SI / dt),).
        """
        n = int(round(T * YRSID_SI / dt))

        out = np.zeros(n, dtype=complex)
        shared = min(n, self.record.size)
        out[:shared] = self.record[:shared]

        return out


def flr_wrapper_kwargs(cfg, orbits, offset: int = None) -> dict:
    """Arguments for ``ResponseWrapper``, so both codes are set up alike.

    ``orbits`` is a parameter rather than built here: constructing it needs
    ``lisatools``, which this package does not depend on. Build it from
    ``cfg.orbit_file`` - the same file the response uses - and configure it
    before passing it in.

    ``Tobs`` is sized so their record spans our whole padded grid *plus* the
    offset, since theirs starts earlier.

    Parameters
    ----------
    cfg : ComparisonConfig
        On the ESA orbits.
    orbits : lisatools.detector.Orbits
        Built from ``cfg.orbit_file`` and configured.
    offset : int
        Sample offset; computed when omitted. Optional.

    Returns
    -------
    kwargs : dict
        Pass as ``ResponseWrapper(generator, **kwargs)``.
    """
    offset = grid_offset(cfg) if offset is None else int(offset)

    return {
        "Tobs": (offset + cfg.N) * cfg.dt / YRSID_SI,
        "dt": cfg.dt,
        "index_lambda": INDEX_LAMBDA,
        "index_beta": INDEX_BETA,
        "t0": FLR_T0_BUFFER,
        "flip_hx": FLIP_HX,
        "remove_sky_coords": True,
        "is_ecliptic_latitude": True,
        "remove_garbage": REMOVE_GARBAGE,
        "orbits": orbits,
        "tdi": TDI_GENERATION,
        "tdi_chan": TDI_CHANNELS,
    }


def measure_time_offset(reference, other, dt: float, frequency: float) -> float:
    r"""Timing offset between two records of the same monochromatic source.

    Two stages, because neither suffices alone. A cross-correlation locates
    the offset to about a sample; the phase of a single Fourier bin then gives

    .. math::

        \Delta t = -\frac{\arg(B/A)}{2 \pi f}

    which is exact for a monochromatic source and resolves the 44 ms that a
    :math:`10^{-6}` mismatch allows - 0.4% of a sample, well beyond what
    fitting a correlation peak could reach. That phase is unique only modulo
    :math:`1/f`, here 197 s, which is what the first stage settles.

    The *value* is the diagnostic. Convention errors are quantised and none of
    them lands anywhere near 44 ms, so the number names the cause:

    ========== =============================================
    offset     cause
    ========== =============================================
    5 s        half-sample convention
    10 s       off-by-one sample
    8.34 s     reception against emission time
    640 s      wavelet row centring
    1280 s     ``ORBIT_LEAD_IN`` not applied
    10000 s    their ``t0`` buffer
    2.07e9 s   absolute against rebased file epoch
    ========== =============================================

    Parameters
    ----------
    reference, other : np.ndarray
        Two records on the same uniform grid. Array shape=(n,).
    dt : float
        Cadence, in seconds.
    frequency : float
        Source frequency, in Hz.

    Returns
    -------
    offset : float
        Seconds by which ``other`` lags ``reference``.
    """
    reference = np.asarray(reference, dtype=float)
    other = np.asarray(other, dtype=float)

    if reference.shape != other.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} against {other.shape}."
        )

    n = reference.size

    # Coarse: correlation peak, to about a sample.
    spectrum = np.fft.rfft(reference) * np.conj(np.fft.rfft(other))
    lag = int(np.argmax(np.fft.irfft(spectrum, n=n)))
    if lag > n // 2:
        lag -= n
    coarse = lag * dt

    # Fine: phase of the source's own bin, unique modulo 1/f.
    phase = np.exp(-2.0j * np.pi * frequency * np.arange(n) * dt)
    fine = -np.angle(np.dot(other, phase) / np.dot(reference, phase))
    fine /= 2.0 * np.pi * frequency

    # Lift the fine estimate onto the cycle the coarse one picked out.
    period = 1.0 / frequency

    return float(fine + np.round((coarse - fine) / period) * period)


def common_interior(cfg, offset: int = None) -> slice:
    """Wavelet rows on which both records are meaningful.

    Ours is limited by the guard band, theirs by the zeroed ``t0`` buffer.
    The guard dominates at every sane tiling - 52 rows against about 8 - but
    it is computed rather than assumed, so it stays right if ``L`` or their
    buffer changes.

    Parameters
    ----------
    cfg : ComparisonConfig
        Supplies the tiling and the padding.
    offset : int
        Sample offset; computed when omitted. Optional.

    Returns
    -------
    interior : slice
        Rows to compare on.
    """
    offset = grid_offset(cfg) if offset is None else int(offset)

    buffer_rows = int(np.ceil(FLR_T0_BUFFER / cfg.dt / cfg.Nf))

    first = max(cfg.padding["rows_left"], buffer_rows - offset // cfg.Nf)
    last = cfg.Nt - max(cfg.padding["rows_right"], buffer_rows)

    return slice(first, last)


def report(ours, theirs, cfg, channels=("X", "Y", "Z")) -> str:
    """Compare two sets of wavelet-domain channels and say how they differ.

    The calibration block: amplitude ratio, mismatch and relative L2 per
    channel, on the interior alone. A ratio away from unity points at the unit
    conversion, a mismatch near one at a sign or a 90 degree phase - the
    signature of an unapplied derivative - and a large relative L2 beside a
    small mismatch at scale rather than shape.

    Parameters
    ----------
    ours : np.ndarray
        This package's channels. Array shape=(num_channels, Nt, Nf).
    theirs : np.ndarray
        Theirs, already through ``convert``. Same shape.
    cfg : ComparisonConfig
        Supplies the interior.
    channels : sequence of str
        Names, for the table. Optional.

    Returns
    -------
    text : str
        Multi-line summary.
    """
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)

    if ours.shape != theirs.shape:
        raise ValueError(
            f"Shape mismatch: {ours.shape} against {theirs.shape}."
        )

    interior = common_interior(cfg)

    lines = [
        f"rows {interior.start}:{interior.stop} of {cfg.Nt}   "
        f"f = {cfg.f_gb * 1e3:.6f} mHz",
        f"{'ch':<3} {'ours rms':>12} {'theirs rms':>12} {'ratio':>10} "
        f"{'mismatch':>12} {'rel_L2':>12}",
    ]

    for name, mine, other in zip(channels, ours, theirs):
        a, b = mine[interior], other[interior]
        result = wdm_mismatch(a, b)
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        lines.append(
            f"{name:<3} {rms_a:>12.4e} {rms_b:>12.4e} "
            f"{rms_b / rms_a:>10.6f} {result['mismatch']:>12.3e} "
            f"{result['rel_L2']:>12.3e}"
        )

    return "\n".join(lines)
