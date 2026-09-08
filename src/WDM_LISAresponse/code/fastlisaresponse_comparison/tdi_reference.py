r"""Second-generation TDI in the time domain: the yardstick.

Forms the same combination as ``tdi``, from the same term expansion and the
same composed delays, but applies each delay by interpolating a sampled series
instead of shifting a coefficient grid. That is the *only* difference between
the two, and it is deliberate: the algebra and the geometry are shared so that
an error in either shows up identically in both and cancels from the
comparison, leaving the wavelet time-shift operator as the one thing being
measured.

Fed the output of ``LISAresponse.one_arm_response_time_domain`` this is the
full reference path - trapezoidal link integrals, then interpolated delays -
against which the wavelet pipeline's local-delay approximation can be priced.
Fed link series that came from the wavelet response instead, it isolates the
TDI stage alone.

It is slow, and meant to be. The delays are solved afresh on all :math:`N`
sample times rather than the :math:`N_t` wavelet centres, which is
:math:`N_f` times the density, and every term interpolates the full record.
Measured, it is about twelve times the wavelet path and some seven eighths of
the whole comparison - more than the time-domain *response* by an order, which
is not the obvious way round.

Most of that is the geometry, and it could be cut two- or threefold: the
chains share prefixes, :math:`Q` sitting inside :math:`QP` and so on, so a
trie walk would solve each distinct prefix once instead of the 168 arm-solves
XYZ currently costs. That is deliberately not done. This module exists to be
read and believed rather than to be fast, and a shared prefix cache is exactly
the kind of machinery whose correctness one then has to argue about. If the
runtime ever matters, the answer is a coarser grid, not a cleverer reference.
"""

import numpy as np

from ..lisa_response.response_sub import LTT_NUM_ITERATIONS
from ..utils.utils import interpolate_uniform_grid, resolve_time_chunk_size
from .tdi import DEFAULT_CHANNELS, CHANNEL_SPACECRAFT, composed_delay, gen2_terms
from ..utils.utils import resolve_link_index


def gen2_xyz_time_domain(response,
                         link_series,
                         channels=DEFAULT_CHANNELS,
                         num_taps: int = 12,
                         num_iterations: int = LTT_NUM_ITERATIONS,
                         memory_budget_bytes: float = 2 ** 30) -> np.ndarray:
    r"""Second-generation X, Y, Z from six time-domain link responses.

    Each of the sixteen terms per channel is one interpolation of one link
    series at :math:`t - \Delta(t)`, with :math:`\Delta` the cumulative delay
    of that term's chain - the same single-reparametrisation argument the
    wavelet path uses, so the two differ only in how the delay is applied.

    Parameters
    ----------
    response : LISAresponse
        A built response, supplying the sample-time grid, the orbit provider
        and the link convention.
    link_series : np.ndarray
        Link delay fluctuations as time series, in seconds, in the order of
        ``response.link_convention``. Array shape=(num_links, N).
    channels : sequence of str
        Channels to form, from ``"X"``, ``"Y"``, ``"Z"``. Optional.
    num_taps : int
        Lagrange taps for the delay interpolation. Twelve is the package
        default and is documented against the :math:`10^{-4}` local-delay
        term; the galactic binary of this study sits at 19.7 samples per
        cycle, far into that stencil's accurate range. Optional.
    num_iterations : int
        Fixed-point sweeps in each light-travel-time solve. Optional.
    memory_budget_bytes : float
        Ceiling on working memory, in bytes. The delays are solved and the
        interpolations evaluated in time blocks sized to fit; the input and
        output arrays are the caller's own and are not counted, as elsewhere
        in the package. Optional.

    Returns
    -------
    channels : np.ndarray
        Second-generation channels, in seconds, in the order given.
        Array shape=(len(channels), N).

    Notes
    -----
    The delays are evaluated on ``response.times``, which are *absolute* -
    the same timeline the orbit provider is queried on and the same one
    ``t0`` anchors. Nothing here works in samples or in time relative to the
    signal, because a delay law evaluated on a grid that starts at zero when
    the record starts at :math:`t_0` is offset by the whole leading guard
    band, and that mistake is close to invisible: it saturates the residual at
    a value that responds to neither the tap count nor the lag count.

    The interpolation stencil reaches ``num_taps/2`` samples beyond each query
    and the delays reach 58 s, about six samples at this cadence, so the
    outermost dozen or so samples of the record are not meaningful. They sit
    inside the guard band that ``config`` already provides, so no separate
    allowance is needed - but compare on the interior, as everywhere else.
    """
    link_series = np.asarray(link_series, dtype=float)

    if link_series.ndim != 2:
        raise ValueError(
            f"link_series must have shape (num_links, N), got "
            f"{link_series.shape}. Batched sources are not supported; loop "
            f"on the host."
        )

    num_links, num_samples = link_series.shape

    if num_links != len(response.link_convention):
        raise ValueError(
            f"link_series has {num_links} links against a convention of "
            f"{len(response.link_convention)}."
        )

    times = np.asarray(response.times, dtype=float)

    if num_samples != times.size:
        raise ValueError(
            f"link_series is {num_samples} samples against the response's "
            f"{times.size}."
        )

    for channel in channels:
        if channel not in CHANNEL_SPACECRAFT:
            raise ValueError(
                f"channel must be one of {tuple(CHANNEL_SPACECRAFT)}, got "
                f"{channel!r}."
            )

    grid_start = float(times[0])
    grid_step = float(times[1] - times[0])

    chunk = resolve_time_chunk_size(
        num_samples, 1, 1, memory_budget_bytes=memory_budget_bytes,
    )

    out = np.zeros((len(channels), num_samples), dtype=float)

    for position, channel in enumerate(channels):
        # Terms sharing a chain are gathered so its delay is solved once, the
        # solve being the expensive part at this density.
        by_chain = {}
        for link, chain, sign in gen2_terms(channel):
            index = resolve_link_index(link, response.link_convention)
            by_chain.setdefault(chain, []).append((index, float(sign)))

        for chain, members in by_chain.items():
            if not chain:
                for index, sign in members:
                    out[position] += sign * link_series[index]
                continue

            for start in range(0, num_samples, chunk):
                stop = min(start + chunk, num_samples)
                block = times[start:stop]

                query = block - composed_delay(
                    response.orbits_class, block, chain, num_iterations,
                )

                for index, sign in members:
                    values = interpolate_uniform_grid(
                        link_series[index][np.newaxis, :],
                        grid_start,
                        grid_step,
                        query[np.newaxis, :],
                        num_taps=num_taps,
                    )
                    out[position, start:stop] += sign * np.asarray(values[0])

    return out
