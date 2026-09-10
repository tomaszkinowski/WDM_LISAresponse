r"""Second-generation TDI in the time domain: the yardstick.

Forms the same combination as ``tdi``, from the same term expansion and the
same composed delays, but applies each delay by interpolating a sampled series
instead of shifting a coefficient grid. 

Fed the output of ``LISAresponse.one_arm_response_time_domain`` this is the
full reference path - trapezoidal link integrals, then interpolated delays -
against which the wavelet pipeline's local-delay approximation can be priced.
Fed link series that came from the wavelet response instead, it isolates the
TDI stage alone.
"""

import numpy as np

from ..lisa_response.response_sub import LTT_NUM_ITERATIONS
from ..utils.utils import (
    interpolate_uniform_grid, resolve_link_index, resolve_time_chunk_size,
)
from .tdi import (
    DEFAULT_CHANNELS, composed_delay, gen2_terms, validate_channels,
)


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

    channels = validate_channels(channels)

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
