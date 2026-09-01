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

The *apply* half is pure ``jnp`` and traces cleanly. It is written as small
named steps - the source basis, the antenna-pattern factors, the angular
geometry of each path, the detector amplitudes, the batched shift, the domain
conversion - composed by two jitted entry points, ``one_arm_time_domain_chunk``
and ``one_arm_response_wdm``. Only those two carry ``@jax.jit``: the steps are
plain functions, so XLA inlines them into the caller and fuses across the
boundaries. Splitting the work this way is therefore free, and it is what lets
``response.py`` read as a sequence of named physical operations while the
detail stays findable here.

Following the rule documented on :class:`LISAresponse`, every one of these
takes its arrays as explicit arguments rather than reading them off ``self``.
Machinery that describes no part of the response - resolving the ``link``
selector, interpolating a uniformly sampled series and budgeting memory -
lives in ``utils.utils`` instead.
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


def resolve_domain(wdm_class, array, from_domain : str, to_domain : str):
    r"""
    Move an array between the time and wavelet representations.

    A single place for the ``dwt``/``idwt`` decision, used on the way in by
    ``prepare_sources`` and on the way out by ``finalise_response``, so neither
    has to know which transform is needed.

    Parameters
    ----------
    wdm_class : WDM.WDM_transform
        The transform supplying ``dwt`` and ``idwt``.
    array : jnp.ndarray
        Array in the ``from_domain`` representation, shape (..., N) for
        ``"time"`` or (..., Nt, Nf) for ``"wdm"``.
    from_domain : str
        Representation ``array`` is in, ``"time"`` or ``"wdm"``.
    to_domain : str
        Representation wanted, ``"time"`` or ``"wdm"``.

    Returns
    -------
    array : jnp.ndarray
        The same data in ``to_domain``; the input unchanged when the two agree.
    """
    if from_domain == to_domain:
        return array

    return (wdm_class.dwt(array) if to_domain == "wdm"
            else wdm_class.idwt(array))


def compute_angular_geometry_td(theta,
                                phi,
                                light_travel_time,
                                link_direction,
                                receiver_position,
                                sender_position,
                                reception_times):
    r"""
    Sky-dependent quantities for the time-domain response.

    Everything the quadrature needs from the source direction: the antenna
    response, the two ends of the retarded-phase window, and the projection of
    the propagation direction on the link.

    The window runs between the retarded wave phase at emission and at
    reception,

    .. math::

        \xi_r = t - \hat{\Omega}\cdot x_r(t)/c , \qquad
        \xi_s = t - \ell_{sr}(t)
                  - \hat{\Omega}\cdot x_s(t-\ell_{sr})/c ,

    both formed from the positions the geometry solve produced rather than
    from the identity :math:`\xi_r - \xi_s = \ell_{sr}(1 -
    \hat{\Omega}\cdot\hat{r}_{sr})`. The two agree to :math:`10^{-12}` s, so
    this is not an accuracy choice: the definition is used because this is the
    path whose job is to be checkable, and because it survives a geometry term
    that breaks that relation.

    Parameters
    ----------
    theta, phi : jnp.ndarray
        Source direction angles, in radians. Array shape=(batch,). The sky
        basis is built here rather than passed in: it is angular geometry, and
        nothing outside this function needs it.
    light_travel_time : jnp.ndarray
        :math:`\ell_{sr}`, in seconds. Array shape=(num_links, chunk).
    link_direction : jnp.ndarray
        :math:`\hat{r}_{sr}`. Array shape=(num_links, chunk, 3).
    receiver_position : jnp.ndarray
        :math:`x_r`, in metres. Array shape=(num_links, chunk, 3).
    sender_position : jnp.ndarray
        :math:`x_s` at emission, in metres. Array shape=(num_links, chunk, 3).
    reception_times : jnp.ndarray
        Reception times, in seconds. Array shape=(chunk,).

    Returns
    -------
    projection_plus, projection_cross : jnp.ndarray
        Antenna-pattern factors. Array shape=(batch, num_links, chunk).
    xi_receive : jnp.ndarray
        Retarded phase at reception, :math:`\xi_r`, in seconds. Same shape.
    tau : jnp.ndarray
        Effective light travel time :math:`\tau_{sr} = \xi_r - \xi_s`, in
        seconds - the same quantity ``compute_angular_geometry_wdm`` reaches
        through the identity. Same shape.
    omega_dot_direction : jnp.ndarray
        :math:`\hat{\Omega}\cdot\hat{r}_{sr}`, which sets the response
        prefactor. Same shape.
    """
    u, v, n = build_sky_basis(theta, phi)
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

    xi_receive = reception_times - omega_dot_receiver
    xi_emit = reception_times - light_travel_time - omega_dot_sender

    return (projection_plus, projection_cross, xi_receive,
            xi_receive - xi_emit, omega_dot_direction)


def compute_angular_geometry_wdm(theta,
                                 phi,
                                 light_travel_time,
                                 link_direction):
    r"""
    Sky-dependent quantities for the wavelet response.

    The antenna response and :math:`\tau_{sr}`, the effective light travel
    time of the link measured along the gravitational-wave propagation
    direction,

    .. math::

        \tau_{sr}(t_n) = \ell_{sr}(t_n)
            \left( 1 + \hat{n}\cdot\hat{r}_{sr}(t_n) \right) .

    This is the width of the retarded-phase window the wave is averaged over,
    :math:`\tau_{sr} = \xi_r - \xi_s`, so it is the same quantity
    ``compute_angular_geometry_td`` returns the ends of - reached here through
    the identity, since the wavelet path stores no sender position.

    The factor :math:`(1 + \hat{n}\cdot\hat{r}_{sr})` is the closing rate
    between photon and wavefront. It vanishes when the wave travels along the
    arm with the photon, which surfs the wavefront and accumulates no phase;
    it is :math:`\ell_{sr}` for a wave crossing the arm, and :math:`2\ell_{sr}`
    for one travelling against the photon.

    Nothing here carries a frequency axis, so these stay
    :math:`(\ldots, N_t)` and only ``compute_detector_amplitudes`` builds the
    large arrays.

    Parameters
    ----------
    theta, phi : jnp.ndarray
        Source direction angles, in radians. Array shape=(batch,). The sky
        basis is built here rather than passed in: it is angular geometry.
    light_travel_time : jnp.ndarray
        :math:`\ell_{sr}(t_n)`, in seconds. Array shape=(num_links, Nt).
    link_direction : jnp.ndarray
        :math:`\hat{r}_{sr}(t_n)`. Array shape=(num_links, Nt, 3).

    Returns
    -------
    projection_plus, projection_cross : jnp.ndarray
        Antenna-pattern factors. Array shape=(batch, num_links, Nt).
    tau : jnp.ndarray
        Effective light travel time :math:`\tau_{sr}`, in seconds. Same shape.
    source_direction : jnp.ndarray
        :math:`\hat{n}`, returned because ``compute_time_shifts`` needs it to
        form :math:`\hat{n}\cdot x_r/c`. Array shape=(batch, 3).
    """
    u, v, n = build_sky_basis(theta, phi)

    projection_plus, projection_cross = antenna_pattern_factors(
        link_direction, u, v,
    )

    direction_dot_source = jnp.tensordot(
        n, link_direction, axes=([-1], [-1]),
    )
    tau = light_travel_time * (1.0 + direction_dot_source)

    return projection_plus, projection_cross, tau, n


def compute_time_shifts(source_direction, receiver_position, tau):
    r"""
    Delay carrying each pixel to the centre of its retarded-phase window.

    .. math::

        \delta_n = \hat{n}\cdot x_r(t_n)/c - \tfrac{1}{2}\tau_{sr}(t_n)

    The wave reaching the receiver at :math:`t_n` was sampled, on average, at
    :math:`t_n + \delta_n`: the first term is the plane-wave phase at the
    receiver, the second steps back by half the effective light travel time,
    to the middle of the window.

    Parameters
    ----------
    source_direction : jnp.ndarray
        :math:`\hat{n}`, from ``compute_angular_geometry_wdm``.
        Array shape=(batch, 3).
    receiver_position : jnp.ndarray
        :math:`x_r(t_n)`, in metres. Array shape=(num_links, Nt, 3).
    tau : jnp.ndarray
        Effective light travel time, in seconds.
        Array shape=(batch, num_links, Nt).

    Returns
    -------
    time_shift : jnp.ndarray
        :math:`\delta_n`, in seconds. Array shape=(batch, num_links, Nt).
    """
    return jnp.tensordot(
        source_direction / c, receiver_position, axes=([-1], [-1]),
    ) - 0.5 * tau


def compute_detector_amplitudes(light_travel_time,
                                projection_plus,
                                projection_cross,
                                tau,
                                wavelet_freqs):
    r"""
    Per-pixel amplitudes of the one-arm transfer function.

    .. math::

        A^{+,\times}_{nm} = \tfrac{1}{2} \ell_{sr}(t_n) F^{+,\times}(t_n)
            \, \mathrm{sinc}\!\left( f_m \tau_{sr}(t_n) \right)

    The sinc is the window average of a wave at the pixel's centre frequency:
    averaging :math:`e^{2\pi i f \xi}` over a window of width
    :math:`\tau_{sr}` gives exactly :math:`\mathrm{sinc}(f \tau_{sr})`, the
    closed form the time-domain path reaches by quadrature instead.

    Evaluating it at :math:`f_m` rather than at the wave's true frequency is
    the *local-delay approximation*, the dominant error in this pipeline: L2
    of order :math:`10^{-4}`, mismatch :math:`10^{-8}`, for a source on a band
    centre, rising to :math:`5\times10^{-3}` for one halfway between two.

    This is where the frequency axis enters, so these are the first arrays of
    size :math:`(\ldots, N_t, N_f)`.

    Parameters
    ----------
    light_travel_time : jnp.ndarray
        :math:`\ell_{sr}(t_n)`, in seconds. Array shape=(num_links, Nt).
    projection_plus, projection_cross : jnp.ndarray
        Antenna-pattern factors. Array shape=(batch, num_links, Nt).
    tau : jnp.ndarray
        Effective light travel time :math:`\tau_{sr}`, in seconds. Same shape.
    wavelet_freqs : jnp.ndarray
        Pixel centre frequencies :math:`f_m`, in hertz. Array shape=(Nf,).

    Returns
    -------
    amplitude_plus, amplitude_cross : jnp.ndarray
        :math:`A^{+}` and :math:`A^{\times}`.
        Array shape=(batch, num_links, Nt, Nf).
    """
    transfer = jnp.sinc(tau[..., jnp.newaxis] * wavelet_freqs)
    scale = 0.5 * light_travel_time[..., jnp.newaxis]

    return (scale * projection_plus[..., jnp.newaxis] * transfer,
            scale * projection_cross[..., jnp.newaxis] * transfer)


def apply_batch_shift(wdm_class,
                      filter_tables,
                      wavelet_coefficients,
                      time_shift,
                      lag_block : int = None):
    r"""
    Apply a per-link variable time shift to a batch of coefficient grids.

    ``wdm_class.apply_variable_time_shift`` takes one :math:`(N_t, N_f)` grid
    and one :math:`(N_t,)` delay, so the source and link axes are flattened
    together and swept in a ``lax.scan``. Sequential is deliberate: vmapping
    the six links measured 18.5% *slower* on CPU, because XLA already threads
    within each operation.

    The delay is passed with its own sign, not negated. Shifting the
    *coefficients* is the active transformation and carries the opposite sign
    to shifting the basis functions, since
    :math:`\langle f(t-\delta), g_{nm}(t)\rangle = \langle f(t),
    g_{nm}(t+\delta)\rangle`, so passing :math:`\delta_n` samples the wave at
    :math:`t + \delta_n` - the window centre. The shift itself is good to
    :math:`6\times10^{-7}` at :math:`L=25`, far below the local-delay term.

    The shift wraps periodically at the ends of the record rather than running
    off it, so roughly :math:`L` pixel rows at each end are contaminated; drop
    :math:`L/2` per side before comparing against anything. Zero padding the
    input remains the caller's responsibility.

    Parameters
    ----------
    wdm_class : WDM.WDM_transform
        The transform supplying ``apply_variable_time_shift``.
    filter_tables : jnp.ndarray
        Time-delay filter tables, shape=(2, 2L+1, num_interp_points).
    wavelet_coefficients : jnp.ndarray
        Unshifted coefficients. Array shape=(batch, num_links, Nt, Nf).
    time_shift : jnp.ndarray
        Delay per pixel row, in seconds. Array shape=(batch, num_links, Nt).
    lag_block : int
        Lags per iteration of the inner sum; ``None`` unrolls. Optional.

    Returns
    -------
    shifted : jnp.ndarray
        Array shape=(batch, num_links, Nt, Nf).
    """
    batch_size, num_links, num_times, num_freqs = wavelet_coefficients.shape
    flat = (batch_size * num_links,)

    def shift_one_link(carry, inputs):
        r"""Delay one link's coefficient grid."""
        coefficients, delta = inputs

        return carry, wdm_class.apply_variable_time_shift(
            filter_tables, coefficients, delta, lag_block=lag_block,
        )

    _, shifted = jax.lax.scan(
        shift_one_link,
        None,
        (wavelet_coefficients.reshape(flat + (num_times, num_freqs)),
         time_shift.reshape(flat + (num_times,))),
    )

    return shifted.reshape(batch_size, num_links, num_times, num_freqs)


@partial(jax.jit, static_argnames=("wdm_class", "lag_block"))
def one_arm_response_wdm(wdm_class,
                         filter_tables,
                         wavelet_plus,
                         wavelet_cross,
                         theta,
                         phi,
                         light_travel_time,
                         link_direction,
                         receiver_position,
                         wavelet_freqs,
                         lag_block : int = None):
    r"""
    One-arm response in the wavelet domain, for every link and source.

    This is the production path. Where ``one_arm_time_domain_chunk`` evaluates
    the retarded-phase integral node by node, this evaluates it once per pixel
    in closed form and then moves the result to the retarded phase with the
    WDM time-shift operator, which is what makes it fast enough for a
    likelihood.

    The closed form is exact for a monochromatic wave. Averaging
    :math:`e^{2\pi i f \xi}` over the window of width :math:`2\tau_n` gives
    :math:`\mathrm{sinc}(2 f \tau_n)`, so the pixel amplitude is

    .. math::

        A^{+,\times}_{nm} = \tfrac{1}{2} \ell_{sr}(t_n) F^{+,\times}(t_n)
            \, \mathrm{sinc}\!\left( 2 f_m \tau_n \right) ,

    and the unshifted coefficients are
    :math:`w^\xi = A^+ w^{h_+} + A^\times w^{h_\times}`. Evaluating the sinc at
    the pixel's own centre frequency :math:`f_m` is the *local-delay
    approximation*, and it is the dominant error in this pipeline at an L2 of
    order :math:`10^{-4}` - far above the light-travel-time solve, the filter
    tables, or anything the time-domain path contributes. That is why the
    time-domain path exists to check it.

    The shift then carries each pixel to :math:`t_n + \delta_n` through
    ``wdm_class.apply_variable_time_shift``, which applies a time-varying delay
    directly in the wavelet domain rather than returning to the time series.

    Links are swept in a ``lax.scan``, not vmapped: vmapping the six links
    measured 18.5% slower on CPU. The scan also keeps the :math:`(N_t, N_f)`
    amplitudes from existing for every link at once.

    Parameters
    ----------
    wdm_class : WDM.WDM_transform
        The transform supplying ``apply_variable_time_shift``. Static under
        ``jit``.
    filter_tables : jnp.ndarray
        Time-delay filter tables, shape=(2, 2L+1, num_interp_points).
    wavelet_plus : jnp.ndarray
        Wavelet coefficients of :math:`h_+`. Array shape=(batch, Nt, Nf).
    wavelet_cross : jnp.ndarray
        Wavelet coefficients of :math:`h_\times`. Array shape=(batch, Nt, Nf).
    theta, phi : jnp.ndarray
        Source direction angles, in radians. Array shape=(batch,).
    light_travel_time : jnp.ndarray
        :math:`\ell_{sr}(t_n)`, in seconds. Array shape=(num_links, Nt).
    link_direction : jnp.ndarray
        :math:`\hat{r}_{sr}(t_n)`. Array shape=(num_links, Nt, 3).
    receiver_position : jnp.ndarray
        :math:`x_r(t_n)`, in metres. Array shape=(num_links, Nt, 3).
    wavelet_freqs : jnp.ndarray
        Pixel centre frequencies :math:`f_m`, in hertz. Array shape=(Nf,).
    lag_block : int
        Lags handled per iteration of the shift's inner sum. ``None`` unrolls
        it, fastest for small :math:`L`; for :math:`L \gtrsim 30` a small block
        such as 8 avoids an unwieldy graph. Static under ``jit``. Optional.

    Returns
    -------
    response : jnp.ndarray
        Link delay fluctuations :math:`\delta \ell_{sr}`, in seconds, as
        wavelet coefficients. Array shape=(batch, num_links, Nt, Nf).
    """
    projection_plus, projection_cross, tau, source_direction = (
        compute_angular_geometry_wdm(
            theta, phi, light_travel_time, link_direction,
        )
    )

    time_shift = compute_time_shifts(source_direction, receiver_position, tau)

    amplitude_plus, amplitude_cross = compute_detector_amplitudes(
        light_travel_time, projection_plus, projection_cross, tau,
        wavelet_freqs,
    )

    unshifted = (amplitude_plus * wavelet_plus[:, jnp.newaxis]
                 + amplitude_cross * wavelet_cross[:, jnp.newaxis])

    return apply_batch_shift(
        wdm_class, filter_tables, unshifted, time_shift, lag_block=lag_block,
    )


def broadcast_sources(wdm_class,
                      theta,
                      phi,
                      h_plus,
                      h_cross,
                      in_domain : str,
                      target_domain : str):
    r"""
    Validate a batch of sources, put it in one domain and give it one axis.

    Plumbing shared by both response methods, which differ only in the domain
    their kernel wants the waveform in. The leading axes of ``theta``, ``phi``
    and the two waveforms are broadcast against each other and then flattened
    to a single batch axis, so the jitted kernels see a fixed rank whatever the
    caller passed. ``batch_shape`` comes back so ``finalise_response`` can
    restore the caller's axes at the end.

    The sky basis is deliberately *not* built here: it is a physical quantity
    with its own name, so the caller does ``build_sky_basis(theta, phi)`` on
    the broadcast angles returned below, and it stays visible alongside the
    other named steps.

    Parameters
    ----------
    wdm_class : WDM.WDM_transform
        The transform supplying ``dwt``/``idwt`` and the grid sizes.
    theta, phi : jnp.ndarray or float
        Source direction angles, in radians. Array shape=(...,).
    h_plus, h_cross : jnp.ndarray
        Polarisations in the ``in_domain`` representation.
    in_domain : str
        Domain the waveforms arrive in, ``"time"`` or ``"wdm"``.
    target_domain : str
        Domain the kernel needs, ``"time"`` or ``"wdm"``.

    Returns
    -------
    h_plus, h_cross : jnp.ndarray
        Waveforms in ``target_domain``, with one leading batch axis.
    theta, phi : jnp.ndarray
        Angles broadcast to the batch and flattened. Array shape=(batch,).
    batch_shape : tuple
        The broadcast leading shape, for restoring the output's axes.
    """
    assert in_domain in ("time", "wdm"), \
                f"in_domain must be 'time' or 'wdm', got {in_domain!r}."

    h_plus = jnp.asarray(h_plus)
    h_cross = jnp.asarray(h_cross)

    given = ((wdm_class.Nt, wdm_class.Nf) if in_domain == "wdm"
             else (wdm_class.N,))
    for name, array in (("h_plus", h_plus), ("h_cross", h_cross)):
        assert array.shape[-len(given):] == given, \
                    f"With {in_domain=}, {name} must end in axes " \
                    f"{given}, got {array.shape=}."

    h_plus = resolve_domain(wdm_class, h_plus, in_domain, target_domain)
    h_cross = resolve_domain(wdm_class, h_cross, in_domain, target_domain)

    wanted = ((wdm_class.Nt, wdm_class.Nf) if target_domain == "wdm"
              else (wdm_class.N,))

    theta = jnp.asarray(theta, dtype=float)
    phi = jnp.asarray(phi, dtype=float)

    batch_shape = jnp.broadcast_shapes(
        theta.shape,
        phi.shape,
        h_plus.shape[:-len(wanted)],
        h_cross.shape[:-len(wanted)],
    )
    batch_size = int(np.prod(batch_shape, dtype=int))

    return (
        jnp.broadcast_to(
            h_plus, batch_shape + wanted).reshape((batch_size,) + wanted),
        jnp.broadcast_to(
            h_cross, batch_shape + wanted).reshape((batch_size,) + wanted),
        jnp.broadcast_to(theta, batch_shape).reshape(batch_size),
        jnp.broadcast_to(phi, batch_shape).reshape(batch_size),
        batch_shape,
    )


@partial(jax.jit, static_argnames=("num_integral_points", "num_taps"))
def one_arm_time_domain_chunk(h_plus,
                              h_cross,
                              theta,
                              phi,
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
    theta, phi : jnp.ndarray
        Source direction angles, in radians. Array shape=(batch,).
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
    projection_plus, projection_cross, xi_receive, tau, \
        omega_dot_direction = compute_angular_geometry_td(
            theta, phi, light_travel_time, link_direction, receiver_position,
            sender_position, chunk_times,
        )

    batch_size, num_links, chunk_size = tau.shape
    flat_shape = (batch_size, num_links * chunk_size)

    node_spacing = 1.0 / (num_integral_points - 1)

    def accumulate(node, carry):
        r"""Add one trapezoidal node to the running plus/cross means."""
        total_plus, total_cross = carry

        # xi runs from the emission endpoint up to the reception endpoint.
        offset = tau * (1.0 - node * node_spacing)
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
            total_plus + weight * sampled_plus.reshape(tau.shape),
            total_cross + weight * sampled_cross.reshape(tau.shape),
        )

    mean_plus, mean_cross = jax.lax.fori_loop(
        0,
        num_integral_points,
        accumulate,
        (jnp.zeros_like(tau), jnp.zeros_like(tau)),
    )

    return 0.5 * tau / (1.0 - omega_dot_direction) * (
        projection_plus * mean_plus + projection_cross * mean_cross
    )


def one_arm_time_domain_blocked(orbits,
                                links,
                                t0 : float,
                                dt : float,
                                num_samples : int,
                                chunk_size : int,
                                h_plus,
                                h_cross,
                                theta,
                                phi,
                                num_integral_points : int = 10,
                                num_taps : int = 12):
    r"""
    Drive ``one_arm_time_domain_chunk`` over the reception grid in blocks.

    The kernel is jitted and the geometry solve is not, so the two meet here:
    each block has its geometry built on the host and is then handed to the
    device in one call. Blocking is what makes the path runnable at all - see
    ``utils.resolve_time_chunk_size`` for the arithmetic - and the result does
    not depend on the block length.

    Blocks are padded to a fixed length by repeating the last reception time,
    so the kernel sees one shape and compiles once. The padded tail is
    discarded on the way out.

    Parameters
    ----------
    orbits : lisaorbits.Orbits
        Orbit provider exposing ``compute_position(t, sc=[...])``.
    links : tuple
        Directed links as ``(sender, receiver)`` pairs, in output order.
    t0 : float
        Time of sample zero, in seconds.
    dt : float
        Sample cadence, in seconds.
    num_samples : int
        Length of the reception grid, :math:`N`.
    chunk_size : int
        Block length in samples.
    h_plus, h_cross : jnp.ndarray
        Polarisations on the sample grid. Array shape=(batch, N).
    theta, phi : jnp.ndarray
        Source direction angles, in radians. Array shape=(batch,).
    num_integral_points : int
        Trapezoidal nodes per link integral. Optional.
    num_taps : int
        Waveform interpolation stencil width. Optional.

    Returns
    -------
    response : jnp.ndarray
        Link delay fluctuations, in seconds. Array shape=(batch, num_links, N).
    """
    blocks = []

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)

        reception_times = t0 + np.arange(start, stop) * dt
        padding = chunk_size - reception_times.size
        if padding:
            reception_times = np.pad(
                reception_times, (0, padding), mode="edge")

        light_travel_time, direction, receiver, sender = build_link_geometry(
            reception_times, orbits, links, return_sender_position=True,
        )

        block = one_arm_time_domain_chunk(
            h_plus,
            h_cross,
            theta,
            phi,
            jnp.asarray(light_travel_time),
            jnp.asarray(direction),
            jnp.asarray(receiver),
            jnp.asarray(sender),
            jnp.asarray(reception_times),
            t0,
            dt,
            num_integral_points=num_integral_points,
            num_taps=num_taps,
        )

        blocks.append(block[..., :stop - start] if padding else block)

    return jnp.concatenate(blocks, axis=-1)


def finalise_response(wdm_class,
                      response,
                      from_domain : str,
                      out_domain : str,
                      batch_shape : tuple,
                      num_links : int,
                      drop_link_axis : bool):
    r"""
    Convert a response to the requested domain and restore its axes.

    The two response methods differ in the domain their kernel produces but
    agree on everything after it, so the conversion, the unflattening of the
    batch axis and the link-axis squeeze are done once here.

    A ``"time"`` to ``"wdm"`` conversion is driven one link at a time. The
    transform builds a complex intermediate the size of its input, so
    transforming every link together would multiply the peak allocation by the
    number of links for no gain; the reverse direction has no such intermediate
    and goes in one call.

    Parameters
    ----------
    wdm_class : WDM.WDM_transform
        The transform supplying ``dwt``/``idwt``.
    response : jnp.ndarray
        Response with a single leading batch axis, shape=(batch, num_links, N)
        or (batch, num_links, Nt, Nf).
    from_domain : str
        Domain the kernel produced, ``"time"`` or ``"wdm"``.
    out_domain : str
        Domain requested by the caller, ``"time"`` or ``"wdm"``.
    batch_shape : tuple
        Leading shape to restore, from ``prepare_sources``.
    num_links : int
        Number of links on axis 1.
    drop_link_axis : bool
        Whether the caller asked for a single link by a scalar specifier.

    Returns
    -------
    response : jnp.ndarray
        Array shape=batch_shape + (num_links, N), or
        batch_shape + (num_links, Nt, Nf), with the link axis dropped when
        ``drop_link_axis``.
    """
    if from_domain != out_domain:
        # One link at a time when transforming forward: the transform builds a
        # complex intermediate the size of its input, so doing every link at
        # once would multiply the peak by the link count for no gain. The
        # inverse has no such intermediate and goes in one call.
        if out_domain == "wdm":
            response = jnp.stack(
                [resolve_domain(wdm_class, response[:, index],
                                from_domain, out_domain)
                 for index in range(num_links)],
                axis=1,
            )
        else:
            response = resolve_domain(
                wdm_class, response, from_domain, out_domain)

    tail = ((wdm_class.Nt, wdm_class.Nf) if out_domain == "wdm"
            else (wdm_class.N,))
    response = response.reshape(batch_shape + (num_links,) + tail)

    # The link axis sits just after the batch axes in either domain.
    if drop_link_axis:
        response = jnp.squeeze(response, axis=len(batch_shape))

    return response
