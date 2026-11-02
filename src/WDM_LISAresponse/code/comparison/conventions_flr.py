r"""Running ``fastlisaresponse`` on this package's source and grid.

Needs ``fastlisaresponse >= 1.2.1a0``. Earlier releases compute the link as
:math:`h_\mathrm{rec} - h_\mathrm{em}`, the opposite sign to
``lisagwresponse``, and take the emitter's position at reception rather than
emission - a global :math:`-1` and up to :math:`6\times10^{-4}` per link.

Their clock starts at the orbit file's first sample, so ``t0`` is passed as the
time from there to ours. The response class and the orbit provider come from
the caller: nothing here imports ``fastlisaresponse`` or ``lisatools``.

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

from .common import LINK_LABELS, sky_angles


#: The runtime that actually clashes. ``libgcc_s.1.1.dylib`` is vendored twice
#: as well, and is deliberately not listed: a working environment - one where
#: only ``libstdc++`` has been pointed at a single copy - still carries two
#: distinct copies of it and runs the whole comparison without trouble.
#: Flagging it would fail a setup that demonstrably works, which is worse than
#: not checking at all.
_VENDORED = ("libstdc++.6.dylib",)

#: Seconds at each end of their record that are not valid, their ``t_buffer``.
T_BUFFER = 10000.0

#: Second-generation Michelson channels by default, as ``tdi.TDIgen2`` forms.
TDI_GENERATION = "2nd generation"
TDI_CHANNELS = "XYZ"


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


def dead_samples(cfg) -> int:
    """Samples at each end of their record that are not valid."""
    return int(np.ceil(T_BUFFER / cfg.dt))


def run(response_class, orbits, cfg, h_plus, h_cross, theta, phi,
        tdi: str = TDI_GENERATION) -> tuple:
    r"""Link responses and TDI channels from ``fastlisaresponse``, on our grid.

    Parameters
    ----------
    response_class : type
        ``fastlisaresponse.pyResponseTDI``.
    orbits : lisatools.detector.Orbits
        Built from ``cfg.orbit_file`` and configured.
    cfg : ComparisonConfig
        On the ESA orbits: their clock is defined by the file.
    h_plus, h_cross : np.ndarray
        The padded record our response was given. Array shape=(cfg.N,).
    theta, phi : float
        Source direction, in our convention.
    tdi : str
        ``"1st generation"`` or ``"2nd generation"``. Optional.

    Returns
    -------
    links : np.ndarray
        Fractional frequency, in our link order. Array shape=(6, cfg.N).
    channels : np.ndarray
        X, Y, Z as fractional frequency. Array shape=(3, cfg.N).
    """
    assert cfg.orbits_model == "esa", \
                "fastlisaresponse needs an orbit file to set its clock; build " \
                "the configuration with ComparisonConfig.for_esa()."

    assert tuple(int(label) for label in orbits.LINKS) == LINK_LABELS, \
                f"Their link order {list(orbits.LINKS)} is not ours, " \
                f"{list(LINK_LABELS)}."

    model = response_class(1.0 / cfg.dt, cfg.N, orbits=orbits,
                           tdi=tdi, tdi_chan=TDI_CHANNELS)

    assert model.num_pts == cfg.N, \
                f"fastlisaresponse cut the record to {model.num_pts} samples; " \
                f"the orbit file does not cover it."

    lam, beta = sky_angles(theta, phi)
    record = np.asarray(h_plus) + 1j * np.asarray(h_cross)

    model.get_projections(record, lam, beta,
                          t0=cfg.t0 - cfg.orbit_span()[0], t_buffer=T_BUFFER)

    channels = np.stack([np.asarray(channel)
                         for channel in model.get_tdi_delays()])

    return np.asarray(model.y_gw), channels
