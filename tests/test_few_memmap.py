"""The memory-mapped FEW amplitude table behind ``waveforms.emri_generator``.

``few_memmap`` changes one thing about FEW's Kerr model: the spline
coefficients are memory-mapped instead of read. So the checks are that the map
holds exactly what FEW would have read, that the interpolants keep views of it
rather than copies, and that the swap into FEW's waveform class leaves FEW's
own names as they were.

The stock model cannot be the reference: building it costs the 6 GB this
module exists to avoid. The comparison is made coefficient by coefficient
instead, against slices h5py reads one spin knot at a time. Since everything
downstream of the coefficients is FEW's own code, equal coefficients mean
equal waveforms. On a one-year EMRI the two were bit-identical.

Needs FEW and its amplitude file already on disk. Skipped otherwise, rather
than letting a test start a 5 GB download.
"""

import numpy as np
import pytest

pytest.importorskip("few")

import h5py  # noqa: E402
from few import get_file_manager  # noqa: E402
from few.amplitude import ampinterp2d  # noqa: E402
from few.waveform import waveform as few_waveform  # noqa: E402

from WDM_LISAresponse.code.waveforms import waveforms  # noqa: E402

FILENAME = "ZNAmps_l10_m10_n55_DS2Outer.h5"
PATH = get_file_manager().try_get_local_file(FILENAME)

if PATH is None:
    pytest.skip(f"FEW's {FILENAME} is not on disk.", allow_module_level=True)

from WDM_LISAresponse.code.waveforms.few_memmap import (  # noqa: E402
    MemmapAmpInterpKerrEccEq,
    memory_map_dataset,
)


@pytest.fixture(scope="module")
def generator():
    """The generator, built once: about ten seconds."""
    return waveforms.emri_generator()


@pytest.fixture(scope="module")
def amplitudes(generator):
    return generator.waveform_generator.amplitude_generator


def _is_memory_mapped(array):
    """Whether ``array`` is, or is a view of, an ``np.memmap``."""
    while isinstance(array, np.ndarray):
        if isinstance(array, np.memmap):
            return True
        array = array.base
    return False


# ------------------------------------------------------------ the wiring

def test_the_generator_uses_the_memory_mapped_amplitudes(amplitudes):
    assert isinstance(amplitudes, MemmapAmpInterpKerrEccEq)


def test_fews_own_amplitude_class_is_restored(generator):
    """The name is swapped only while the waveform class is built."""
    assert few_waveform.AmpInterpKerrEccEq is ampinterp2d.AmpInterpKerrEccEq


# ---------------------------------------------------------- the coefficients

@pytest.mark.parametrize("region", ["A", "B"])
def test_every_spin_knot_has_an_interpolant(amplitudes, region):
    """Nothing is dropped: the whole spin grid is there, as in FEW."""
    with h5py.File(PATH, "r") as f:
        z_knots = f[f"region{region}/z_knots"][()]

    assert len(getattr(amplitudes, f"spin_information_holder_{region}")) == z_knots.size
    assert np.array_equal(amplitudes.z_values, z_knots)


@pytest.mark.parametrize("region", ["A", "B"])
@pytest.mark.parametrize("knot", [0, 10, 11, 32])
def test_the_interpolants_hold_the_file_s_coefficients_as_views(amplitudes, region, knot):
    """Each knot's coefficients equal what h5py reads, and are not a copy.

    A copy would mean the table is back in memory. Knots 10 and 11 are the
    two an a = 0.9 source at p = 12, e = 0.3 evaluates: FEW's spin coordinate
    there is z = 0.314, not the spin itself.
    """
    holder = getattr(amplitudes, f"spin_information_holder_{region}")[knot]

    with h5py.File(PATH, "r") as f:
        expected = f[f"region{region}/CoeffsRegion{region}"][knot]

    assert _is_memory_mapped(holder.coeff)
    assert np.array_equal(holder.coeff, expected)


def test_a_compressed_dataset_is_refused(tmp_path):
    """Compressed storage has no byte offset to map from."""
    path = tmp_path / "compressed.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(100.0), compression="gzip")

    with h5py.File(path, "r") as f, pytest.raises(ValueError, match="cannot be memory-mapped"):
        memory_map_dataset(path, f["x"])


# ------------------------------------------------------------ end to end

def test_a_short_waveform_comes_out_through_the_package(generator):
    """Four days of a = 0.9 EMRI, through generate_windowed_waveform."""
    _, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "emri", t0=2.0e9, dt=10.0, N=34560,
        model_kwargs=dict(
            td_gen=generator, m1=1.0e6, m2=10.0, a=0.9, p0=12.0, e0=0.3, x0=1.0,
            dist=1.0, theta=np.pi / 3, phi=np.pi / 6, qK=np.pi / 3, phiK=np.pi / 3,
            Phi_phi0=np.pi / 3, Phi_theta0=0.0, Phi_r0=np.pi / 3, eps=1.0e-2))

    assert h_plus.shape == h_cross.shape == (34560,)
    assert np.all(np.isfinite(h_plus)) and np.all(np.isfinite(h_cross))
    assert np.abs(h_plus).max() > 0
