r"""Support routines for the LISA response.

The module has two halves, split by what they depend on and therefore by
whether they can be traced.

The *geometry* half builds the sky-independent link quantities - the
light-travel time :math:`\ell_{sr}`, the link unit vector :math:`\hat{r}_{sr}`
and the receiver position :math:`x_r`. These depend on the orbits and the time
grid but not on the source sky position or the waveform, so they are
precomputed once and reused across every likelihood evaluation. They run on the
host in ``numpy``: orbit providers evaluate SciPy splines
(``InterpolatedOrbits``) or plain ``numpy`` (``EqualArmlengthOrbits``), so
``compute_position`` cannot be traced and this stage can never run under
``jit``. The results are converted to ``jnp`` at the boundary by the caller, so
the apply stage receives device arrays.

The *apply* half - the sky basis, the antenna-pattern factors and the
time-domain one-arm kernel - depends on the source parameters and the
waveform, and runs in ``jnp`` under ``jit``. Following the rule documented on
:class:`LISAresponse`, every one of these is a module-level function taking its
arrays as explicit arguments rather than a method reading them off ``self``.

Both halves are physics. The machinery that surrounds them but describes no
part of the response - resolving the ``link`` selector, interpolating a
uniformly sampled series, and budgeting the memory that decides the block
length - lives in ``utils.utils`` instead.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from lisaconstants import c

from ..utils.utils import interpolate_uniform_grid


#: Fixed-point iterations used to solve the retarded light-travel time. The
#: solve stalls at ~5e-11 s after the second sweep, so a fourth buys nothing -
#: see the convergence note in ``solve_light_travel_time``.
LTT_NUM_ITERATIONS = 3


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
    ``num_iterations`` times. The first sweep gains about four decimal digits,
    because a spacecraft moves by :math:`\sim 10^{-4}` of an arm length in one
    light-crossing time. The second reaches a floor of :math:`\sim 5\times
    10^{-11}` s (about 1.4 cm), set by the interpolation error of the orbit
    provider itself, and further sweeps do not improve on it: the iteration
    settles into a period-2 cycle of that amplitude rather than converging to
    machine precision, which for an 8.3 s light time would be
    :math:`\sim 2\times 10^{-15}` s.

    Measured against a 12-sweep reference at seven epochs spanning the 8.83 yr
    of ``esa-trailing-orbits.h5``: one sweep leaves :math:`7\times 10^{-8}` s,
    two leave :math:`5\times 10^{-11}` s, three are exact to the reference in
    some cases and at the same floor in others. The default of three keeps one
    sweep of margin over the floor. For scale, :math:`5\times 10^{-11}` s is
    :math:`1.5\times 10^{-12}` rad of phase at 5 mHz.

    This is the flat-spacetime delay, with no solar Shapiro term. That is a
    deliberate choice about self-consistency rather than an omission. The
    response is a flat-spacetime plane-wave calculation from end to end: the
    link direction is the straight-line :math:`\hat{r}_{sr}`, and the retarded
    wave phase is :math:`\xi = t - \hat{\Omega}\cdot x/c` along that straight
    line. Adding a curved-spacetime correction to :math:`\ell_{sr}` alone,
    while every other quantity stays flat, breaks the exact relation
    :math:`x_r - x_s = c\,\ell_{sr}\hat{r}_{sr}` that the rest of the geometry
    relies on - the two ways of forming the emission phase then disagree by
    :math:`\Delta_{\mathrm{Sh}}(\hat{\Omega}\cdot\hat{r}_{sr})`, measured at
    :math:`3.4\times10^{-9}` s. The correction is worth 49 m of light time, so
    dropping it leaves that much disagreement with
    ``lisaorbits.compute_ltt``, which reports a Shapiro-inclusive delay. Both
    numbers are far below the :math:`10^{-4}` local-delay error that dominates
    this pipeline, so the consistent choice is the better one.

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
                        num_iterations : int = LTT_NUM_ITERATIONS,
                        return_sender_position : bool = False):
    r"""
    Build the sky-independent geometry for every directed link.

    All links share the same reception-time grid, so the positions of all
    three spacecraft at those times are obtained in a single batched
    ``compute_position`` call and reused as both the receiver positions and
    the iteration seeds. Only the retarded sender lookups, which query a
    different time grid per link, are evaluated link by link. That is
    ``1 + n_links * num_iterations`` spline evaluations rather than
    ``n_links * (2 + num_iterations)``.

    Peak memory is dominated by the output arrays and scales as
    :math:`56 \, n_{\mathrm{links}} N_t` bytes, or :math:`80 \,
    n_{\mathrm{links}} N_t` when the sender position is asked for: roughly
    2 MB for :math:`N_t = 6\times10^3`, and 170 MB for :math:`N_t =
    4\times10^5`. Nothing here allocates a temporary larger than a single
    output array, so the peak is set by ``Nt`` alone; if that ever becomes the
    binding constraint, ``Nf`` is the knob, since :math:`N_t = N/N_f`.

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
    return_sender_position : bool
        Also return the sender position at emission time. The solve computes
        it either way, so this only decides whether it is kept; it costs
        :math:`24 \, n_{\mathrm{links}} N_t` bytes and no extra orbit
        evaluations. The wavelet path does not need it - see
        ``one_arm_time_domain_chunk`` for the identity that removes it - but
        the time-domain path uses it to form the retarded phase at emission
        directly. Optional.

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
    sender_position : np.ndarray
        Sender positions :math:`x_s(t_n - \ell_{sr})` at emission time, in
        metres. Returned only when ``return_sender_position`` is set.
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
    out_sender = (
        np.empty((num_links, num_times, 3), dtype=float)
        if return_sender_position else None
    )

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
        if return_sender_position:
            out_sender[index] = x_s

    if return_sender_position:
        return out_ltt, out_direction, out_receiver, out_sender

    return out_ltt, out_direction, out_receiver


def build_sky_basis(theta, phi):
    r"""
    Build the orthonormal sky-frame triad for a source direction.

    With :math:`\theta` the polar and :math:`\phi` the azimuthal angle of the
    source in the solar-system barycentric frame,

    .. math::

        \hat{u} = (\cos\theta\cos\phi,\ \cos\theta\sin\phi,\ -\sin\theta) ,

    .. math::

        \hat{v} = (\sin\phi,\ -\cos\phi,\ 0) ,

    .. math::

        \hat{n} = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta) .

    :math:`\hat{n}` points from the barycentre towards the source, so the
    gravitational-wave propagation direction is
    :math:`\hat{\Omega} = -\hat{n}`. Callers form it themselves rather than
    receiving it, since it is one negation and carrying both invites sign
    confusion.

    The polarisation tensors :math:`e^+ = \hat{u}\hat{u} - \hat{v}\hat{v}` and
    :math:`e^\times = \hat{u}\hat{v} + \hat{v}\hat{u}` are deliberately not
    built. Every consumer contracts them twice with the same link vector, and
    doing that through the scalars :math:`\hat{r}\cdot\hat{u}` and
    :math:`\hat{r}\cdot\hat{v}` - see ``antenna_pattern_factors`` - is the same
    number for two dot products instead of a pair of :math:`3\times3`
    contractions per link and time sample.

    Parameters
    ----------
    theta : jnp.ndarray or float
        Polar angle of the source direction, in radians. Array shape=(...,),
        broadcast against ``phi``.
    phi : jnp.ndarray or float
        Azimuthal angle of the source direction, in radians.
        Array shape=(...,), broadcast against ``theta``.

    Returns
    -------
    u : jnp.ndarray
        Sky-frame basis vector :math:`\hat{u}`. Array shape=(..., 3).
    v : jnp.ndarray
        Sky-frame basis vector :math:`\hat{v}`. Array shape=(..., 3).
    n : jnp.ndarray
        Source direction :math:`\hat{n}`. Array shape=(..., 3).
    """
    theta, phi = jnp.broadcast_arrays(
        jnp.asarray(theta, dtype=float),
        jnp.asarray(phi, dtype=float),
    )

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    zero = jnp.zeros_like(phi)

    u = jnp.stack(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1)
    v = jnp.stack([sin_phi, -cos_phi, zero], axis=-1)
    n = jnp.stack(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=-1)

    return u, v, n


def antenna_pattern_factors(link_direction, u, v):
    r"""
    Project link directions onto the polarisation basis.

    .. math::

        F^+ = (\hat{r}\cdot\hat{u})^2 - (\hat{r}\cdot\hat{v})^2 , \qquad
        F^\times = 2 (\hat{r}\cdot\hat{u})(\hat{r}\cdot\hat{v}) .

    These are the double contractions :math:`\hat{r} \cdot e^{+,\times} \cdot
    \hat{r}` written out, which is why ``build_sky_basis`` never forms the
    tensors.

    The two operands carry independent leading axes: ``u`` and ``v`` are
    indexed by source, ``link_direction`` by link and time. The output takes
    the source axes first and the link axes second, so a batch of sources
    against a set of links comes back as ``(..., num_links, num_times)``.

    Parameters
    ----------
    link_direction : jnp.ndarray
        Unit vectors :math:`\hat{r}_{sr}`.
        Array shape=(num_links, num_times, 3), or any shape ending in 3.
    u : jnp.ndarray
        Sky-frame basis vector :math:`\hat{u}`. Array shape=(..., 3).
    v : jnp.ndarray
        Sky-frame basis vector :math:`\hat{v}`. Array shape=(..., 3).

    Returns
    -------
    projection_plus : jnp.ndarray
        Plus antenna-pattern factor.
        Array shape=(...,) + link_direction.shape[:-1].
    projection_cross : jnp.ndarray
        Cross antenna-pattern factor. Same shape as ``projection_plus``.
    """
    r_dot_u = jnp.tensordot(u, link_direction, axes=([-1], [-1]))
    r_dot_v = jnp.tensordot(v, link_direction, axes=([-1], [-1]))

    return r_dot_u ** 2 - r_dot_v ** 2, 2.0 * r_dot_u * r_dot_v


@partial(jax.jit, static_argnames=("num_integral_points", "num_taps"))
def one_arm_time_domain_chunk(h_plus,
                              h_cross,
                              u,
                              v,
                              n,
                              light_travel_time,
                              link_direction,
                              receiver_position,
                              sender_position,
                              chunk_times,
                              waveform_start : float,
                              waveform_step : float,
                              num_integral_points : int = 10,
                              num_taps : int = 12):
    r"""
    One-arm time-domain response for one block of reception times.

    This is the time-domain formulation, evaluated directly rather than through
    the WDM filter tables. The fractional frequency shift accumulated along a
    directed link is the wave averaged over the null geodesic that connects
    emission to reception, so the link delay fluctuation is

    .. math::

        \delta \ell_{sr}(t) =
            \frac{1}{2\left(1 - \hat{\Omega}\cdot\hat{r}_{sr}\right)}
            \int_{\xi_s}^{\xi_r}
                \left[ F^+ h_+(\xi) + F^\times h_\times(\xi) \right]
            \mathrm{d}\xi ,

    with the retarded wave phase evaluated at the two endpoints,

    .. math::

        \xi_r = t - \hat{\Omega}\cdot x_r(t)/c , \qquad
        \xi_s = t - \ell_{sr}(t) - \hat{\Omega}\cdot x_s(t-\ell_{sr})/c .

    Both endpoints come from their definitions, using the sender and receiver
    positions the geometry solve produced. They could instead be reached
    through :math:`\xi_r - \xi_s = \ell_{sr}(1 -
    \hat{\Omega}\cdot\hat{r}_{sr})`, which follows from
    :math:`x_r - x_s = c\,\ell_{sr}\hat{r}_{sr}` and needs no sender position
    - that is why the wavelet path stores receiver positions only. Both routes
    are exact, agreeing to :math:`10^{-12}` s, so this is not an accuracy
    choice: the definition is used because this is the path whose job is to be
    checkable, and because it survives a geometry term that breaks that
    relation, as a Shapiro delay on :math:`\ell_{sr}` alone did before it was
    removed; see ``solve_light_travel_time``.

    The prefactor stays an explicit ratio rather than folding into
    :math:`\ell_{sr}/2` by the same identity. It is regular as written, the
    window closing at the same rate as the denominator; only an exact float
    equality :math:`\hat{\Omega}\cdot\hat{r}_{sr} = 1` gives :math:`0/0`,
    and that direction is genuinely singular in this model.

    Two independent errors enter and neither knob repairs the other: the
    quadrature falls as :math:`(2\pi f W)^2/12(P-1)^2` with the node count
    :math:`P`, the interpolation as :math:`(2\pi f \delta t)^{n-1}` with the
    stencil width :math:`n`. The second matters more than it looks, because the
    window :math:`W` is *narrower than the sample spacing* - 1.63 intervals at
    LISA's 8.3 s arm and 5 s cadence - so every node is a sub-sample
    reconstruction rather than a lookup. At the default twelve taps the
    interpolation term sits far below the quadrature term, leaving
    ``num_integral_points`` as the only one worth tuning; see
    ``interpolate_uniform_grid``. The wavelet path meets the same difficulty
    and answers it through its filter tables instead.

    The mean is a trapezoidal rule on ``num_integral_points`` nodes spread
    uniformly across the window, accumulated in a ``jax.lax.fori_loop`` rather
    than materialised, so peak memory is one ``(batch, links, chunk)`` array
    per node and *not* ``num_integral_points`` times that. The node count
    therefore costs time and not memory.

    Parameters
    ----------
    h_plus : jnp.ndarray
        Plus polarisation sampled on the waveform grid.
        Array shape=(batch, num_samples).
    h_cross : jnp.ndarray
        Cross polarisation sampled on the waveform grid.
        Array shape=(batch, num_samples).
    u : jnp.ndarray
        Sky-frame basis vector :math:`\hat{u}`. Array shape=(batch, 3).
    v : jnp.ndarray
        Sky-frame basis vector :math:`\hat{v}`. Array shape=(batch, 3).
    n : jnp.ndarray
        Source direction :math:`\hat{n}`. Array shape=(batch, 3).
    light_travel_time : jnp.ndarray
        Light-travel time :math:`\ell_{sr}`, in seconds, on this block of
        reception times. Array shape=(num_links, chunk).
    link_direction : jnp.ndarray
        Unit vectors :math:`\hat{r}_{sr}` on this block.
        Array shape=(num_links, chunk, 3).
    receiver_position : jnp.ndarray
        Receiver positions :math:`x_r`, in metres, on this block.
        Array shape=(num_links, chunk, 3).
    sender_position : jnp.ndarray
        Sender positions :math:`x_s` at emission time, in metres, on this
        block. Array shape=(num_links, chunk, 3).
    chunk_times : jnp.ndarray
        Reception times of this block, in seconds. Array shape=(chunk,).
    waveform_start : float
        Time of waveform sample zero, in seconds.
    waveform_step : float
        Waveform sample spacing, in seconds.
    num_integral_points : int
        Number of trapezoidal nodes per link integral, at least two. Static
        under ``jit``. Optional.
    num_taps : int
        Stencil width for the waveform interpolation, even and at least two.
        Static under ``jit``. Optional.
    Returns
    -------
    response : jnp.ndarray
        Link delay fluctuation :math:`\delta \ell_{sr}`, in seconds.
        Array shape=(batch, num_links, chunk).
    """
    propagation_direction = -n

    projection_plus, projection_cross = antenna_pattern_factors(
        link_direction, u, v,
    )

    # (batch, num_links, chunk) from (batch, 3) against (num_links, chunk, 3).
    omega_dot_direction = jnp.tensordot(
        propagation_direction, link_direction, axes=([-1], [-1]),
    )
    omega_dot_receiver = jnp.tensordot(
        propagation_direction, receiver_position / c, axes=([-1], [-1]),
    )
    omega_dot_sender = jnp.tensordot(
        propagation_direction, sender_position / c, axes=([-1], [-1]),
    )

    xi_receive = chunk_times - omega_dot_receiver
    xi_emit = chunk_times - light_travel_time - omega_dot_sender
    window = xi_receive - xi_emit

    batch_size, num_links, chunk_size = window.shape
    flat_shape = (batch_size, num_links * chunk_size)

    node_spacing = 1.0 / (num_integral_points - 1)

    def accumulate(node, carry):
        r"""Add one trapezoidal node to the running plus/cross means."""
        total_plus, total_cross = carry

        # xi runs from the emission endpoint up to the reception endpoint.
        offset = window * (1.0 - node * node_spacing)
        query = (xi_receive - offset).reshape(flat_shape)

        weight = jnp.where(
            (node == 0) | (node == num_integral_points - 1),
            0.5 * node_spacing,
            node_spacing,
        )

        sampled_plus = interpolate_uniform_grid(
            h_plus, waveform_start, waveform_step, query, num_taps,
        )
        sampled_cross = interpolate_uniform_grid(
            h_cross, waveform_start, waveform_step, query, num_taps,
        )

        return (
            total_plus + weight * sampled_plus.reshape(window.shape),
            total_cross + weight * sampled_cross.reshape(window.shape),
        )

    mean_plus, mean_cross = jax.lax.fori_loop(
        0,
        num_integral_points,
        accumulate,
        (jnp.zeros_like(window), jnp.zeros_like(window)),
    )

    return 0.5 * window / (1.0 - omega_dot_direction) * (
        projection_plus * mean_plus + projection_cross * mean_cross
    )
