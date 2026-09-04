"""Shared fixtures.

Tier 1 needs nothing but ``rng``. Tier 2 adds a small ``WDM_transform``; Tier 3
adds an orbit provider and a built ``LISAresponse``. The expensive ones are
session scoped because they are pure functions of their arguments.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


#: Epoch for every fixture that needs one. Deliberately *not* zero: a zero
#: epoch hides floating-point cancellation that only appears at a realistic
#: time, and any tolerance calibrated against it would be wrong in production.
T0 = 3.15e7

#: Tiling for the shared transform. Nt = 64 is large enough for L = 25 to be a
#: meaningful fraction of the record and small enough to be instant.
DT = 5.0
NF = 16
NT = 64
Q = 8
MAX_LAG_L = 25


@pytest.fixture
def rng():
    """A seeded generator, so a failure is reproducible."""
    return np.random.default_rng(20260902)


@pytest.fixture(scope="session")
def wdm():
    """A small WDM transform on the tiling above."""
    import WDM

    return WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=NF * NT, q=Q)


@pytest.fixture(scope="session")
def filter_tables(wdm):
    """Time-delay filter tables for ``MAX_LAG_L``."""
    return wdm.build_time_delay_filter_interpolants(MAX_LAG_L, 16)


@pytest.fixture(scope="session")
def orbits():
    """Analytic orbits.

    ``EqualArmlengthOrbits`` rather than the interpolated ESA file: it is
    closed form, so the light-travel-time solve converges exactly and
    tolerances can be tight. Tests that specifically care about the
    interpolated provider build it themselves.
    """
    from lisaorbits import EqualArmlengthOrbits

    return EqualArmlengthOrbits()


@pytest.fixture(scope="session")
def response(wdm, orbits):
    """A built response.

    Session scoped: ``setup_geometry`` is the slow part and is pure in its
    arguments.
    """
    from WDM_LISAresponse.code.lisa_response.response import LISAresponse

    return LISAresponse(wdm, orbits, t0=T0, max_lag_L=MAX_LAG_L)


class MinimalOrbits:
    """An orbit provider exposing nothing but ``compute_position``.

    The response duck-types its provider, and the documented contract is one
    method wide. Wrapping a real provider in an object with no other
    attributes is what proves that, and would fail the moment something
    downstream reached for ``compute_ltt`` or an ``args`` dictionary.
    """

    def __init__(self, inner):
        self._inner = inner

    def compute_position(self, t, sc):
        """Forward to the wrapped provider, so positions stay genuine."""
        return self._inner.compute_position(t, sc=sc)


@pytest.fixture
def minimal_orbits(orbits):
    """A provider stripped to the one method the response may rely on."""
    return MinimalOrbits(orbits)


@pytest.fixture
def waveform(wdm):
    """A tapered monochromatic signal on the shared grid."""
    times = np.arange(wdm.N) * wdm.dt
    taper = np.hanning(wdm.N)
    frequency = 5.0 / (2.0 * wdm.Nf * wdm.dt)
    return (taper * np.sin(2.0 * np.pi * frequency * times),
            taper * np.cos(2.0 * np.pi * frequency * times))


@pytest.fixture
def wavelet_waveform(wdm, waveform):
    """The same signal as wavelet coefficients."""
    import jax.numpy as jnp

    h_plus, h_cross = waveform
    return (wdm.dwt(jnp.asarray(h_plus)), wdm.dwt(jnp.asarray(h_cross)))
