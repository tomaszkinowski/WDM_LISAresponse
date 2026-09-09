r"""Everything that differs between the two codes, measured rather than assumed.

Each constant and conversion below was established by running both pipelines
and looking at the result, and the evidence is recorded beside it.

The one that matters is the variable. ``fastlisaresponse`` returns a
*fractional frequency*, this package returns a *delay in seconds*, and

.. math::

    \mathrm{theirs} = -\frac{\mathrm{d}}{\mathrm{d}t}\,\mathrm{ours} .

The sky mapping is latitude against polar angle with no frame rotation, the
two sample grids differ by a whole number of samples, and no scale or time
offset has to be fitted.

**The polarisation bases are not the same triad, and the minus sign above is
not only the derivative.** The two codes' :math:`\hat{u}`, :math:`\hat{v}`
differ by a 90 degree rotation about the propagation direction - see
``to_flr_angles``, which carries the mapping - and a rotation of a spin-2
basis by 90 degrees sends both :math:`e^+` and :math:`e^\times` to minus
themselves. Since the same :math:`(h_+, h_\times)` array is handed to both
codes, our response computes the negative of what theirs does for the same
source, and that :math:`-1` is folded into the sign applied by
``to_wdm_units``.

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
#: ``pyResponseTDI`` expects, so no flip. 
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
    r"""Translate their sample numbering into ours.

    Both codes read the same orbit file at the same cadence, so they sample the
    same instants - they only number them differently, because sample zero sits
    in a different place::

        file starts here
        |
        |<-- ORBIT_LEAD_IN -->|
        |                     |
        theirs: k = 0 ....... k = 128 ....................
        ours:                 j = 0 ......................

    Theirs is pinned to the file's first sample: ``pyResponseTDI`` builds its
    axis as ``arange(len(input)) * dt`` and cannot start elsewhere. Ours starts
    ``ORBIT_LEAD_IN`` later, because the light-travel-time solve reads the
    sender backwards and ``InterpolatedOrbits`` will not extrapolate below the
    file's first sample.

    That is the whole role: their sample ``j + offset`` is our sample ``j``, to
    the bit. Exact, lossless, and true of every sample - including ones neither
    code can use. **Which samples to trust is a separate question**, answered by
    ``common_interior``; this only lets it state one code's dead zone in the
    other's frame. Computed from the configuration and never fitted to
    either code's output, so the comparison has no free alignment parameter.

    Parameters
    ----------
    cfg : ComparisonConfig
        Must be on the ESA orbits: the analytic provider has no file, so there
        is no epoch to measure the gap against.

    Returns
    -------
    offset : int
        Whole samples from their first sample to ours. 128 at the default
        tiling, which is also exactly one wavelet row.

    Raises
    ------
    ValueError
        The gap is not a whole number of samples. The two codes would then
        sample *interleaved* instants rather than the same ones, and comparing
        them would need an interpolation whose error is indistinguishable from
        a real timing disagreement.
    """
    if cfg.orbits_model != "esa":
        raise ValueError(
            "grid_offset needs an orbit file to define the epoch; build the "
            "configuration with ComparisonConfig.for_esa(). Do not work "
            "around it - see ComparisonConfig.build_orbits for why the "
            "analytic orbits cannot be compared against fastlisaresponse."
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

    Notes
    -----
    That holds for the sky *direction*, not for the polarisation basis built on
    it. With this mapping applied, ``fastlisaresponse``'s sky vectors (in the
    ``get_projections`` block) relate to ours as

    ================== ==================== ==========================
    theirs             ours                 kind of difference
    ================== ==================== ==========================
    :math:`\hat{k}`    :math:`-\hat{n}`     naming only
    :math:`\hat{u}`    :math:`\hat{v}`      real: 90 degree rotation
    :math:`\hat{v}`    :math:`-\hat{u}`     real: 90 degree rotation
    ================== ==================== ==========================

    The minus in the first row is not a disagreement: their :math:`\hat{k}`
    points *away* from the source and our :math:`\hat{n}` *towards* it, so
    :math:`\hat{k} = -\hat{n}` is what the two names mean. Both codes put the
    source in the same place, which is what the angle mapping achieves.

    The transverse pair is a real difference - the same object, turned 90
    degrees. Both triads are right-handed about the propagation direction
    (checked: :math:`\hat{u}\times\hat{v}` equals it in each), so it is a
    proper rotation: the sense of :math:`\psi` is shared, only its zero point
    differs. On a spin-2 basis a 90 degree turn sends :math:`e^+` and
    :math:`e^\times` to minus themselves *together* - verified as
    :math:`F^{+,\times}` coming out at exactly :math:`-1` times theirs - so it
    is a global sign with no :math:`+`/:math:`\times` mixing, invisible in the
    residual. Nothing here corrects for it: ``to_wdm_units`` already carries
    that :math:`-1`, and rotating the angles would double-count it.
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

    The minus is a *product* of two conventions, not the derivative relation
    on its own. The two codes' polarisation bases differ by a 90 degree
    rotation, worth a global :math:`-1` on both antenna-pattern factors - see
    ``to_flr_angles`` Notes. What is measured below is the net sign; this
    module does not separate the halves, and a comment claiming either one
    alone would be unfounded.

    Applied to *their* output and never to ours, so the quantity under test is
    not touched by the thing that makes it comparable.

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
    r"""A stand-in waveform generator that replays the array we already built.

    An adapter, not a reformulation: the samples are untouched. It changes only
    the interface (``ResponseWrapper`` insists on calling a generator rather
    than taking an array), the format (one complex :math:`h_+ + i h_\times`
    series instead of two real ones), and the placement (``offset`` zeros in
    front, so both codes' records sit on the same instants).

    Without it ``ResponseWrapper`` would build its own waveform: untapered,
    unpadded, and at full amplitude exactly where our record holds zeros.
    Replaying rather than regenerating also keeps the Tukey taper as the same
    samples, not a re-evaluation that could differ from scipy's at the window
    edges. Both codes then see bit-identical input, so the waveform cannot
    contribute to the residual.

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

    Raises
    ------
    ValueError
        Via ``grid_offset``, when ``cfg`` is not on the ESA orbits - the two
        packages' analytic equal-armlength models are in different frames.
        Passing ``offset`` explicitly skips that call, and so that check.
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
