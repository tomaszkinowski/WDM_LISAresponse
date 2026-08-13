r"""Sky-independent link geometry for the LISA response.

The three quantities built here - the light-travel time :math:`\ell_{sr}`, the
link unit vector :math:`\hat{r}_{sr}` and the receiver position :math:`x_r` -
depend on the orbits and the WDM tiling but not on the source sky position or
the waveform. They are therefore precomputed once and reused across every
likelihood evaluation.

Everything in this module runs on the host in ``numpy``. Orbit providers
evaluate SciPy splines (``InterpolatedOrbits``) or plain ``numpy``
(``EqualArmlengthOrbits``), so ``compute_position`` cannot be traced and this
stage can never run under ``jit``. The results are converted to ``jnp`` at the
boundary by the caller, so the jitted apply stage receives device arrays.
"""

import numpy as np

from lisaconstants import c


#: Fixed-point iterations used to solve the retarded light-travel time.
LTT_NUM_ITERATIONS = 4


def solve_light_travel_time(times,
                            sender : int,
                            receiver : int,
                            orbits,
                            x_r : np.ndarray = None,
                            x_s : np.ndarray = None,
                            num_iterations : int = LTT_NUM_ITERATIONS):
    r"""
    Solve the retarded light-travel time for one directed link.

    The receiver is held at reception time :math:`t`, and the sender is
    iterated back along its own worldline until it is consistent with the
    delay,

    .. math::

        \ell_{sr}(t) = \frac{1}{c}
            \left| x_r(t) - x_s\big(t - \ell_{sr}(t)\big) \right| .

    The fixed point is seeded with the instantaneous separation and swept
    ``num_iterations`` times. Convergence is fast because a spacecraft moves
    by :math:`\sim 10^{-4}` of an arm length in one light-crossing time, so
    each sweep gains roughly four decimal digits.

    Parameters
    ----------
    times : np.ndarray
        Reception times, in seconds. Array shape=(Nt,).
    sender : int
        Sending spacecraft index, :math:`s`.
    receiver : int
        Receiving spacecraft index, :math:`r`.
    orbits : lisaorbits.Orbits
        Orbit provider exposing ``compute_position(t, sc=[...])``.
    x_r : np.ndarray
        Receiver positions at reception time, in metres, shape=(Nt, 3). Pass
        this when it has already been evaluated on the shared reception grid
        to avoid a redundant spline call. Optional.
    x_s : np.ndarray
        Sender positions at reception time, in metres, shape=(Nt, 3), used
        only to seed the iteration. Optional.
    num_iterations : int
        Number of fixed-point sweeps. Optional.

    Returns
    -------
    x_s : np.ndarray
        Sender positions at emission time :math:`t - \ell_{sr}`, in metres.
        Array shape=(Nt, 3).
    x_r : np.ndarray
        Receiver positions at reception time, in metres. Array shape=(Nt, 3).
    ltt : np.ndarray
        Light-travel times :math:`\ell_{sr}(t)`, in seconds.
        Array shape=(Nt,).
    """
    times = np.asarray(times, dtype=float)

    if x_r is None:
        x_r = orbits.compute_position(times, sc=[receiver])[:, 0, :]
    if x_s is None:
        x_s = orbits.compute_position(times, sc=[sender])[:, 0, :]

    ltt = np.linalg.norm(x_r - x_s, axis=1) / c

    for _ in range(int(num_iterations)):
        x_s = orbits.compute_position(times - ltt, sc=[sender])[:, 0, :]
        ltt = np.linalg.norm(x_r - x_s, axis=1) / c

    return x_s, x_r, ltt


def link_direction(x_s : np.ndarray, x_r : np.ndarray) -> np.ndarray:
    r"""
    Compute the directed link unit vector.

    .. math::

        \hat{r}_{sr}(t) = \frac{x_r(t) - x_s(t-\ell_{sr})}
                               {\left| x_r(t) - x_s(t-\ell_{sr}) \right|}

    Parameters
    ----------
    x_s : np.ndarray
        Sender positions at emission time, in metres. Array shape=(Nt, 3).
    x_r : np.ndarray
        Receiver positions at reception time, in metres. Array shape=(Nt, 3).

    Returns
    -------
    r_hat : np.ndarray
        Unit vectors pointing from sender to receiver. Array shape=(Nt, 3).
    """
    separation = x_r - x_s
    return separation / np.linalg.norm(separation, axis=1)[:, np.newaxis]


def build_link_geometry(times,
                        orbits,
                        links,
                        num_iterations : int = LTT_NUM_ITERATIONS):
    r"""
    Build the sky-independent geometry for every directed link.

    All links share the same reception-time grid, so the positions of all
    three spacecraft at those times are obtained in a single batched
    ``compute_position`` call and reused as both the receiver positions and
    the iteration seeds. Only the retarded sender lookups, which query a
    different time grid per link, are evaluated link by link. That is
    ``1 + n_links * num_iterations`` spline evaluations rather than
    ``n_links * (2 + num_iterations)``.

    Peak memory is dominated by the three output arrays and scales as
    :math:`56 \, n_{\mathrm{links}} N_t` bytes: roughly 2 MB for
    :math:`N_t = 6\times10^3`, and 170 MB for :math:`N_t = 4\times10^5`.
    Nothing here allocates a temporary larger than a single output array, so
    the peak is set by ``Nt`` alone; if that ever becomes the binding
    constraint, ``Nf`` is the knob, since :math:`N_t = N/N_f`.

    Parameters
    ----------
    times : np.ndarray
        Reception times, in seconds. For the WDM response these are the
        wavelet centre times :math:`t_n = t_0 + n \Delta T`, not the
        :math:`N` sample times. Array shape=(Nt,).
    orbits : lisaorbits.Orbits
        Orbit provider exposing ``compute_position(t, sc=[...])``.
    links : tuple
        Directed links as ``(sender, receiver)`` spacecraft index pairs. The
        given order indexes axis 0 of every returned array.
    num_iterations : int
        Number of fixed-point sweeps per link. Optional.

    Returns
    -------
    light_travel_time : np.ndarray
        Light-travel times :math:`\ell_{sr}(t_n)`, in seconds.
        Array shape=(n_links, Nt).
    link_direction : np.ndarray
        Unit vectors :math:`\hat{r}_{sr}(t_n)`. Array shape=(n_links, Nt, 3).
    receiver_position : np.ndarray
        Receiver positions :math:`x_r(t_n)` at reception time, in metres.
        Array shape=(n_links, Nt, 3).
    """
    times = np.asarray(times, dtype=float)
    links = tuple((int(sender), int(receiver)) for sender, receiver in links)

    num_times = times.size
    num_links = len(links)

    # One batched call on the shared reception grid covers every receiver and
    # seeds every sender. Axis 1 of the result indexes spacecraft.
    spacecraft = sorted({sc for link in links for sc in link})
    row_of = {sc: row for row, sc in enumerate(spacecraft)}
    positions = orbits.compute_position(times, sc=spacecraft)

    out_ltt = np.empty((num_links, num_times), dtype=float)
    out_direction = np.empty((num_links, num_times, 3), dtype=float)
    out_receiver = np.empty((num_links, num_times, 3), dtype=float)

    for index, (sender, receiver) in enumerate(links):
        x_s, x_r, ltt = solve_light_travel_time(
            times,
            sender,
            receiver,
            orbits,
            x_r=positions[:, row_of[receiver], :],
            x_s=positions[:, row_of[sender], :],
            num_iterations=num_iterations,
        )

        out_ltt[index] = ltt
        out_direction[index] = link_direction(x_s, x_r)
        out_receiver[index] = x_r

    return out_ltt, out_direction, out_receiver
