"""FEW's Kerr EMRI model without reading its amplitude table into memory.

``FastKerrEccentricEquatorialFlux`` builds its amplitude interpolant by reading
two arrays of spline coefficients, 4.0 and 1.1 GB, from
``ZNAmps_l10_m10_n55_DS2Outer.h5`` in full. That happens before any waveform is
made and whatever its length, and on an 8 GB machine it is enough to exhaust
memory. The table's first axis is the spin grid, and a waveform evaluates only
the one or two spin knots either side of its own spin: about a sixteenth of the
table.

Both arrays are stored contiguous and uncompressed, so here they are
memory-mapped rather than read. The operating system pages in what the
interpolant touches, and because those pages are backed by the file it can drop
them again under pressure instead of compressing or swapping them. Nothing else
changes: the classes below are FEW's own with only the loading step replaced,
and the waveforms are bit-identical. Measured on a one-year EMRI at
:math:`a = 0.9`, peak memory is 1.5 GB against 6.4 GB, and building the
generator takes 9 s against 33 s.

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
