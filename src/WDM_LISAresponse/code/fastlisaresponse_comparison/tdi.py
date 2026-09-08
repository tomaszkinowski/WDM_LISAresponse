r"""Second-generation TDI in the wavelet domain.

Takes the six link responses and forms X, Y, Z without leaving the
:math:`(N_t, N_f)` grid. With
:math:`a = y_{k\to i} + D_{k\to i} y_{i\to k}`,
:math:`b = y_{j\to i} + D_{j\to i} y_{i\to j}`,
:math:`P = D_{k\to i}D_{i\to k}` and :math:`Q = D_{j\to i}D_{i\to j}`,

.. math::

    X_2 = (1 - Q - QP + PQQ)\,a \;-\; (1 - P - PQ + QPP)\,b .

Each round trip returns only the home spacecraft's own laser noise,
:math:`a = (P-1)p_i` and :math:`b = (Q-1)p_i`, so a combination cancels it
exactly when its two beams traverse the same arms in orders that compose
alike. First generation sends them :math:`P,Q` against :math:`Q,P`, leaving
:math:`X_1 = [P,Q]p` - first order in the rate of change of the arms. Second
generation sends them :math:`P,Q,Q,P` against :math:`Q,P,P,Q`, and those
palindromes leave :math:`X_2 = (PQ^2P - QP^2Q)p`, second order. That is the
whole difference between the two, and the reason the chains here run to seven
arms rather than three.

Chained delay operators are one time reparametrisation,

.. math::

    D_A D_B f(t) = f\big(t - \ell_A(t) - \ell_B(t - \ell_A(t))\big) ,

so every term - including the five-arm ones - is a *single*
``apply_variable_time_shift`` of an original response grid, never a chain of
them. That keeps the pipeline two shifts deep, which is what the doubled guard
band in ``config`` is sized for.

The composed delays depend only on the orbits, not on the source, so they are
resolved once at construction and reused across every sky position and
waveform. A call is then 42 shifts and nothing else.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..lisa_response.response_sub import LTT_NUM_ITERATIONS, solve_light_travel_time
from ..utils.utils import Link, resolve_link_index


#: Channel name to its ``(home, j, k)`` spacecraft triple. X is built around
#: spacecraft 1; Y and Z are the cyclic relabellings :math:`1\to2\to3\to1`.
CHANNEL_SPACECRAFT = {
    "X": (1, 2, 3),
    "Y": (2, 3, 1),
    "Z": (3, 1, 2),
}

#: Channels formed by default.
DEFAULT_CHANNELS = ("X", "Y", "Z")


def gen2_terms(channel: str):
    r"""Expand a second-generation channel into signed, delayed link terms.

    Pure combinatorics - no arrays, no orbits, no grid - so the algebra can be
    checked against a written-out table without building anything.

    Parameters
    ----------
    channel : str
        One of ``"X"``, ``"Y"``, ``"Z"``.

    Returns
    -------
    terms : list of tuple
        Sixteen ``(link, chain, sign)`` entries. ``link`` is the ``Link`` whose
        response grid the term draws on; ``chain`` is a tuple of ``Link``
        naming the delays to apply, **outermost first**, empty for the two
        undelayed terms; ``sign`` is :math:`\pm1`.
    """
    if channel not in CHANNEL_SPACECRAFT:
        raise ValueError(
            f"channel must be one of {tuple(CHANNEL_SPACECRAFT)}, got "
            f"{channel!r}."
        )

    home, j, k = CHANNEL_SPACECRAFT[channel]

    # Round trips out from the home spacecraft and back.
    P = (Link(k, home), Link(home, k))
    Q = (Link(j, home), Link(home, j))

    # (link, inner chain) for each round trip's two one-way measurements. The
    # outbound leg is delayed to refer it to the home spacecraft's own
    # reception time.
    a_legs = ((Link(k, home), ()), (Link(home, k), (Link(k, home),)))
    b_legs = ((Link(j, home), ()), (Link(home, j), (Link(j, home),)))

    # Beam one traverses P, Q, Q, P; beam two traverses Q, P, P, Q. Reading
    # off what each accumulates and differencing gives
    #     (1 - Q - QP + PQQ) a  -  (1 - P - PQ + QPP) b .
    a_ops = ((+1, ()), (-1, Q), (-1, Q + P), (+1, P + Q + Q))
    b_ops = ((-1, ()), (+1, P), (+1, P + Q), (-1, Q + P + P))

    terms = []
    for ops, legs in ((a_ops, a_legs), (b_ops, b_legs)):
        for sign, op_chain in ops:
            for link, inner in legs:
                terms.append((link, op_chain + inner, sign))

    return terms


def composed_delay(orbits,
                   times,
                   chain,
                   num_iterations: int = LTT_NUM_ITERATIONS):
    r"""Cumulative delay of a chain of delay operators, in seconds.

    Walks the chain outermost first, accumulating

    .. math::

        \Delta \mathrel{+}= \ell_{\mathrm{arm}}(t_{\mathrm{cur}}) , \qquad
        t_{\mathrm{cur}} \mathrel{-}= \ell_{\mathrm{arm}}(t_{\mathrm{cur}}) ,

    so :math:`D_A D_B f(t) = f(t - \Delta(t))` exactly, for any chain length.

    The intermediate grids :math:`t - \ell_A(t)` are not the wavelet centre
    times, so the cached geometry on ``LISAresponse`` cannot serve them and the
    solve is rerun at each step. That is host ``numpy`` and negligible beside
    the shifts it feeds.

    Parameters
    ----------
    orbits : lisaorbits.Orbits
        Provider exposing ``compute_position(t, sc=[...])``.
    times : np.ndarray
        Reception times, in seconds. Array shape=(Nt,).
    chain : sequence of Link
        Delays to compose, outermost first. Empty gives zeros.
    num_iterations : int
        Fixed-point sweeps per solve. Optional.

    Returns
    -------
    delay : np.ndarray
        Cumulative delay :math:`\Delta(t)`, in seconds. Array shape=(Nt,).
    """
    times = np.asarray(times, dtype=float)

    total = np.zeros_like(times)
    current = times.copy()

    for link in chain:
        _, _, ltt = solve_light_travel_time(
            current, link.sender, link.receiver, orbits,
            num_iterations=num_iterations,
        )
        total += ltt
        current -= ltt

    return total


@partial(jax.jit, static_argnames=("wdm_class",))
def _shift(wdm_class, filter_tables, grid, delta):
    """One variable time shift. Jitted once, reused for every term."""
    return wdm_class.apply_variable_time_shift(filter_tables, grid, delta)


class TDIgen2:
    r"""Second-generation X, Y, Z from six wavelet-domain link responses.

    Construction resolves the term expansion and every composed delay, which
    depend on the orbits alone. Calls are then pure array work.

    Parameters
    ----------
    response : LISAresponse
        A built response. Supplies the transform, the filter tables, the orbit
        provider, the wavelet centre times and the link convention. Its
        ``link_convention`` must contain all six directed links.
    channels : sequence of str
        Channels to form, from ``"X"``, ``"Y"``, ``"Z"``. Optional.
    num_iterations : int
        Fixed-point sweeps in each light-travel-time solve. Optional.

    Attributes
    ----------
    channels : tuple of str
        Channels formed, indexing axis 0 of the result.
    num_shifts : int
        Non-trivial shifts per call, summed over channels. Fourteen per
        channel: sixteen terms less the two whose chains are empty.

    Notes
    -----
    The sign passed to the shift operator is negative. Its convention is that
    :math:`\delta` samples the wave at :math:`t + \delta`, whereas a TDI
    :math:`D` is a retardation, so a delay :math:`\Delta` is applied as
    :math:`-\Delta`. Checked by ``tests``, not reasoned about twice.

    Terms accumulate in place. Six link grids, three accumulators and one grid
    in flight is around 350 MiB at the default tiling; stacking all
    forty-two shifted terms first would be 1.4 GiB before any temporary.
    """

    def __init__(self,
                 response,
                 channels=DEFAULT_CHANNELS,
                 num_iterations: int = LTT_NUM_ITERATIONS):
        self.response = response
        self.channels = tuple(channels)

        for channel in self.channels:
            if channel not in CHANNEL_SPACECRAFT:
                raise ValueError(
                    f"channel must be one of {tuple(CHANNEL_SPACECRAFT)}, "
                    f"got {channel!r}."
                )

        times = np.asarray(response.wavelet_times)

        # Composed delays are cached by chain: several terms share a chain
        # while drawing on different links, and the solve need not repeat.
        cache = {}
        self._plan = {}

        for channel in self.channels:
            plan = []
            for link, chain, sign in gen2_terms(channel):
                index = resolve_link_index(link, response.link_convention)

                if chain:
                    if chain not in cache:
                        cache[chain] = jnp.asarray(
                            composed_delay(
                                response.orbits_class, times, chain,
                                num_iterations,
                            )
                        )
                    delay = cache[chain]
                else:
                    delay = None

                plan.append((index, delay, float(sign)))

            self._plan[channel] = plan

        self._delays = cache
        self.num_shifts = sum(
            sum(1 for _, delay, _ in plan if delay is not None)
            for plan in self._plan.values()
        )

    def __call__(self, link_wavelets) -> jnp.ndarray:
        r"""Form the channels from a set of link responses.

        Parameters
        ----------
        link_wavelets : jnp.ndarray
            Link delay fluctuations as wavelet coefficients, exactly as
            ``LISAresponse.__call__`` returns them. Array
            shape=(num_links, Nt, Nf).

        Returns
        -------
        channels : jnp.ndarray
            Second-generation channels, in seconds, in the order given by
            ``self.channels``. Array shape=(len(channels), Nt, Nf).
        """
        link_wavelets = jnp.asarray(link_wavelets)

        if link_wavelets.ndim != 3:
            raise ValueError(
                f"link_wavelets must have shape (num_links, Nt, Nf), got "
                f"{link_wavelets.shape}. Batched sources are not supported; "
                f"loop on the host."
            )

        wdm_class = self.response.wdm_class
        filter_tables = self.response.filter_tables

        out = []
        for channel in self.channels:
            total = jnp.zeros(link_wavelets.shape[1:], dtype=float)

            for index, delay, sign in self._plan[channel]:
                grid = link_wavelets[index]
                if delay is not None:
                    grid = _shift(wdm_class, filter_tables, grid, -delay)
                total = total + sign * grid

            out.append(total)

        return jnp.stack(out)

    def delay_summary(self) -> str:
        """Range of every distinct composed delay, in seconds.

        The headroom check: each must sit inside the filter tables' reach, and
        the longest chain is what decides that.
        """
        lines = [f"{len(self._delays)} distinct chains, "
                 f"{self.num_shifts} shifts per call"]
        for chain, delay in sorted(self._delays.items(), key=len):
            arms = " ".join(f"{c.sender}>{c.receiver}" for c in chain)
            lines.append(
                f"  [{arms}]  {float(jnp.min(delay)):8.3f} .. "
                f"{float(jnp.max(delay)):8.3f} s"
            )
        return "\n".join(lines)
