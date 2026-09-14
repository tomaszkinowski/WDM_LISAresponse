"""FEW's Kerr EMRI model without reading its amplitude table into memory.

Building ``FastKerrEccentricEquatorialFlux`` constructs its amplitude class,
``AmpInterpKerrEccEq``, which reads the spline coefficients in
``ZNAmps_l10_m10_n55_DS2Outer.h5`` into RAM in full: 5.1 GB, for a process
peak of 6.4 GB. That happens once, before any waveform is made and whatever
its length, and it is enough to exhaust an 8 GB machine.

The coefficients cover two regions of parameter space, A and B. Each region
is one array, and in both the first axis is the spin knots::

    CoeffsRegionA   (33, 6993, 2, 1089)   4.0 GB
    CoeffsRegionB   (33, 6993, 2,  289)   1.1 GB

The axes are spin knot, Teukolsky mode :math:`(l, m, n)`, real and imaginary
part, and bicubic spline coefficient over FEW's :math:`(w, u)`. The knots are
in FEW's spin coordinate :math:`z`, not in :math:`a` itself.

A waveform has a single spin, and FEW interpolates linearly in :math:`z`
between the two knots either side of it, or uses one knot if it lands on
one. So a waveform uses at most 2 of the 33 knots in each region, yet stock
FEW reads all 33.

This module memory-maps the two arrays instead of reading them, one map per
region. Each spin knot's interpolant, an ``AmpInterp2D``, holds a view of its
own slice of that map, ``coeffs[i]``, which is one dimension lower. Building
the generator therefore reads only the small knot arrays; coefficients are
read from disk only when a waveform accesses them. Three things make this
work:

- Both arrays are stored contiguous and uncompressed, so each is one block of
  raw bytes at a fixed offset in the file, which is what a memory map needs.
  ``memory_map_dataset`` refuses anything else.
- Spin is the first axis and the arrays are in C order, so each knot's slice
  is itself one contiguous block (122 MB in A, 32 MB in B). Accessing a knot
  reads only that block. Were spin the last axis, every page of the file
  would hold values from all 33 knots, and reading one knot would read the
  whole array.
- ``AmpInterp2D`` stores ``xp.asarray(coefficients)``, which on the CPU
  backend is a view rather than a copy, so the interpolants keep the map
  itself.

When a waveform is evaluated, FEW assigns each trajectory point to region A
or B and calls that region's interpolants at the 1-2 knots it needs. The
operating system reads those knots' pages from the file, and FEW makes a
temporary copy of the requested modes (all of them by default) for its
interpolation kernel, freed after the call. An inspiral that crosses from one
region into the other reads the same knots from both maps. Pages that have
been read stay cached, but they are backed by the file, so under memory
pressure the operating system drops them instead of compressing or swapping
them.

Nothing else changes: the classes below are FEW's own with only the loading
step replaced, and the waveforms are bit-identical. Measured on a one-year
EMRI at :math:`a = 0.9`, peak memory is 1.5 GB against 6.4 GB, and building
the generator takes 9 s against 33 s. On a GPU backend each knot's slice is
still copied to the device at build, as in FEW, so the saving is on the host.

This leans on FEW internals: the layout of ``AmpInterpKerrEccEq.__init__``, and
the module-level name through which ``FastKerrEccentricEquatorialFlux`` finds
its amplitude class. That is why the ``emri`` extra pins ``fastemriwaveforms``
to 2.0. Only ``waveforms.emri_generator`` imports this module, so FEW stays
optional for everything else in the package.
"""
import h5py
import numpy as np

from few import get_file_manager
from few.amplitude.ampinterp2d import AmpInterp2D, AmpInterpKerrEccEq
from few.amplitude.base import AmplitudeBase
from few.utils.baseclasses import KerrEccentricEquatorial
from few.waveform import waveform as few_waveform
from few.waveform.waveform import FastKerrEccentricEquatorialFlux


def memory_map_dataset(path, dataset):
    r"""
    Map a contiguous, uncompressed HDF5 dataset read-only, without reading it.

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 file holding ``dataset``.
    dataset : h5py.Dataset
        The dataset to map. Chunked or compressed storage has no single byte
        offset to map from, and is refused.

    Returns
    -------
    array : np.memmap
        The dataset's values, with its shape and dtype.
    """
    if dataset.chunks is not None or dataset.compression is not None:
        raise ValueError(
            f"{dataset.name} is chunked or compressed, so it cannot be "
            f"memory-mapped; read it with h5py instead."
        )

    return np.memmap(path, dtype=dataset.dtype, mode="r",
                     offset=dataset.id.get_offset(), shape=dataset.shape)


class MemmapAmpInterpKerrEccEq(AmpInterpKerrEccEq):
    r"""
    ``AmpInterpKerrEccEq`` with its spline coefficients memory-mapped.

    ``__init__`` repeats FEW's, region by region, with ``np.memmap`` in place
    of the full ``[()]`` read. Each spin knot's ``AmpInterp2D`` keeps
    ``xp.asarray`` of its slice, which on the CPU backend is a view, so the
    interpolants hold the map itself and the pages are read only when a
    waveform evaluates that knot. On a GPU backend the slices are copied to the
    device exactly as FEW copies them; the saving is on the host.
    """

    def __init__(self, filename=None, downsample_Z=1, force_backend=None, **kwargs):
        AmplitudeBase.__init__(self)
        KerrEccentricEquatorial.__init__(self, force_backend=force_backend, **kwargs)

        self.filename = "ZNAmps_l10_m10_n55_DS2Outer.h5" if filename is None else filename
        path = get_file_manager().get_file(self.filename)

        with h5py.File(path, "r") as f:
            for region in ("A", "B"):
                if f"region{region}" not in f:
                    continue

                group = f[f"region{region}"]
                coeffs = memory_map_dataset(path, group[f"CoeffsRegion{region}"])
                coeffs = coeffs[::downsample_Z]
                w_knots = group["w_knots"][()]
                u_knots = group["u_knots"][()]
                z_knots = group["z_knots"][()][::downsample_Z]

                setattr(self, f"spin_information_holder_{region}", [
                    self.build_with_same_backend(
                        AmpInterp2D,
                        args=[w_knots, u_knots, coeffs[i],
                              self.l_arr, self.m_arr, self.n_arr],
                    )
                    for i in range(z_knots.size)
                ])

        # As in FEW, the spin knots kept are the last region's.
        self.z_values = z_knots


class MemmapKerrEccentricEquatorialFlux(FastKerrEccentricEquatorialFlux):
    r"""
    ``FastKerrEccentricEquatorialFlux`` built on the memory-mapped amplitudes.

    FEW's ``__init__`` names its amplitude class through the module global
    ``few.waveform.waveform.AmpInterpKerrEccEq``, so that name is pointed at
    ``MemmapAmpInterpKerrEccEq`` for the length of construction and restored
    afterwards, error or not. Pass the class to ``GenerateEMRIWaveform`` in
    place of the name ``"FastKerrEccentricEquatorialFlux"``, or use
    ``waveforms.emri_generator``, which does.
    """

    def __init__(self, *args, **kwargs):
        original = few_waveform.AmpInterpKerrEccEq
        few_waveform.AmpInterpKerrEccEq = MemmapAmpInterpKerrEccEq
        try:
            super().__init__(*args, **kwargs)
        finally:
            few_waveform.AmpInterpKerrEccEq = original
