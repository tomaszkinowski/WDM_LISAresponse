r"""Running ``lisagwresponse`` and ``pytdi`` on this package's source and grid.

``lisagwresponse`` takes the padded record and our sample times directly, so
nothing has to be bridged. ``pytdi`` is told to use TCB light travel times,
``DELAY_DATASET``: its default is spacecraft proper time, which is not what
our geometry solves.

Both return a fractional frequency in their polarisation basis, which
``common`` converts. Nothing here imports either package - the caller passes
the classes in.
"""

import numpy as np

from .common import LINK_LABELS, sky_angles


#: Light travel times in TCB, the quantity our geometry solves.
DELAY_DATASET = "tcb/ltt"


def dead_samples(cfg) -> int:
    """Samples at each end of the ``pytdi`` output that are not valid.

    X2 reaches back eight arm crossings, about 67 s, plus half of the
    32-point interpolation stencil.
    """
    return int(np.ceil(8 * 8.4 / cfg.dt)) + 16


def links(strain_class, cfg, h_plus, h_cross, theta, phi,
          chunk_size : int = 2 ** 18) -> np.ndarray:
    r"""Link responses from ``lisagwresponse``, on our grid.

    Evaluated in blocks of ``chunk_size`` samples: one call holds several
    (n, 6, 3) temporaries, a few GB for a 485-day record in one go.

    Parameters
    ----------
    strain_class : type
        ``lisagwresponse.ReadStrain``.
    cfg : ComparisonConfig
        On the ESA orbits.
    h_plus, h_cross : np.ndarray
        The padded record our response was given. Array shape=(cfg.N,).
    theta, phi : float
        Source direction, in our convention.
    chunk_size : int
        Samples per block. Optional.

    Returns
    -------
    links : np.ndarray
        Fractional frequency, in our link order. Array shape=(6, cfg.N).
    """
    ra, dec = sky_angles(theta, phi)
    strain = strain_class(t_interp=cfg.padded_times,
                          hplus=np.asarray(h_plus),
                          hcross=np.asarray(h_cross),
                          ra=ra, dec=dec, orbits=cfg.orbit_file)

    out = np.empty((len(LINK_LABELS), cfg.N))
    for start in range(0, cfg.N, chunk_size):
        stop = min(start + chunk_size, cfg.N)
        out[:, start:stop] = np.asarray(strain.compute_gw_response(
            cfg.padded_times[start:stop], LINK_LABELS)).T

    return out


def channels(data_class, combinations, cfg, links,
             block_size : int = 2 ** 19,
             margin : int = 2 ** 10) -> np.ndarray:
    r"""TDI channels from ``pytdi``, on our grid.

    Computed in overlapping blocks: a time-varying shift in ``pytdi`` stores
    (order + 1) x n coefficients, about 1 GB per shift for a 485-day record.
    The ``margin`` samples either side of a block absorb the delays and the
    stencil, and are dropped.

    Parameters
    ----------
    data_class : type
        ``pytdi.Data``.
    combinations : sequence
        The combinations to form, e.g. ``(pytdi.michelson.X2, Y2, Z2)``.
    cfg : ComparisonConfig
        On the ESA orbits.
    links : np.ndarray
        Fractional frequency, in our link order, from ``links``.
        Array shape=(6, cfg.N).
    block_size : int
        Samples kept per block. Optional.
    margin : int
        Samples computed and dropped either side of a block. Optional.

    Returns
    -------
    channels : np.ndarray
        Fractional frequency. Array shape=(len(combinations), cfg.N).
    """
    assert margin >= dead_samples(cfg), \
                f"margin must cover the dead samples, got {margin=} against " \
                f"{dead_samples(cfg)}."

    links = np.asarray(links, dtype=float)
    out = np.empty((len(combinations), cfg.N))

    for start in range(0, cfg.N, block_size):
        stop = min(start + block_size, cfg.N)
        lo, hi = max(start - margin, 0), min(stop + margin, cfg.N)

        # A GW enters only the science interferometers; pytdi sets the
        # unlisted ref_/tmi_ measurements to zero.
        data = data_class.from_orbits(
            cfg.orbit_file, fs=1.0 / cfg.dt, t0=float(cfg.padded_times[lo]),
            dataset=DELAY_DATASET,
            **{f"sci_{label}": links[k, lo:hi]
               for k, label in enumerate(LINK_LABELS)})

        for i, combination in enumerate(combinations):
            block = combination.build(**data.args)(data.measurements)
            out[i, start:stop] = np.asarray(block)[start - lo:stop - lo]

    assert np.all(np.isfinite(out)), "pytdi returned non-finite samples."

    return out
