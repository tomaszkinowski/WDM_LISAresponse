#this is where the LISAresponse class will live
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from .response_sub import (
    build_link_geometry,
    build_sky_basis,
    one_arm_time_domain_chunk,
)
from ..utils.utils import (
    resolve_link_selection,
    resolve_time_chunk_size,
)


#: Directed links as ``(sender, receiver)`` spacecraft index pairs.
#:
#: This is ``lisaorbits.LINKS`` written in sender-first tuple form. lisaorbits
#: labels links by the integer code ``AB`` meaning *receiver* ``A`` and
#: *emitter* ``B``, so its ``[12, 23, 31, 13, 32, 21]`` decodes element for
#: element to the pairs below. Keeping the same order means axis 0 of every
#: geometry array lines up with the orbit file's own link ordering.
DEFAULT_DIRECTED_LINKS = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))


class LISAresponse:
    r"""
    Compute the time-frequency or time-domain LISA response.

    Attributes
    ----------
    wdm_class : WDM.WDM_transform
        The WDM transform fixing the time-frequency tiling. Every grid below
        derives from it, so it sets the resolution of every downstream stage.
    orbits_class : lisaorbits.Orbits
        Orbit provider exposing ``compute_position(t, sc=[...])``. Any object
        meeting that one-method contract is accepted.
    t0 : float
        Absolute time of the first sample, :math:`t_0`, in seconds. Places the
        WDM grid on the provider's timeline; the transform itself knows only
        relative times.
    link_convention : tuple
        Directed links as ``(sender, receiver)`` pairs. The given order indexes
        axis 0 of every geometry array, so it is part of the class contract:
        downstream code reads link ``i`` as ``link_convention[i]``.
    num_links : int
        ``len(link_convention)``; six for the full constellation.
    dt, N, Nf, Nt, dT, dF : float or int
        Tiling parameters copied from ``wdm_class``: sample cadence
        :math:`\delta t`, series length :math:`N`, frequency and time band
        counts :math:`N_f` and :math:`N_t = N/N_f`, and the wavelet
        resolutions :math:`\Delta T = N_f \delta t` and
        :math:`\Delta F = 1/(2 N_f \delta t)`, which satisfy
        :math:`\Delta F \Delta T = 1/2`.
    times : jnp.ndarray
        Sample times :math:`t_k = t_0 + k \delta t`. Array shape=(N,).
    wavelet_times : jnp.ndarray
        Wavelet pixel centre times :math:`t_n = t_0 + n \Delta T`. Array
        shape=(Nt,). The stored geometry lives on this grid, not on ``times``.
    wavelet_freqs : jnp.ndarray
        Wavelet pixel centre frequencies :math:`f_m = m \Delta F`.
        Array shape=(Nf,).
    max_lag_L : int
        Maximum retained wavelet time-pixel lag, :math:`L`, in the filter
        tables.
    num_interp_points : int
        Interpolation points tabulated across
        :math:`-\Delta T/2 \leq \delta \leq \Delta T/2`.
    filter_tables : jnp.ndarray
        Time-delay filter tables, shape=(2, 2L+1, num_interp_points). Index 0
        is :math:`T'_\ell(\delta)`, index 1 is :math:`T_\ell(\delta)`.
    light_travel_time : jnp.ndarray
        :math:`\ell_{sr}(t_n)` per directed link, in seconds.
        Array shape=(num_links, Nt).
    link_direction : jnp.ndarray
        Unit vector :math:`\hat{r}_{sr}(t_n)` from sender at emission to
        receiver at reception. Array shape=(num_links, Nt, 3).
    receiver_position : jnp.ndarray
        :math:`x_r(t_n)` at reception time, in metres.
        Array shape=(num_links, Nt, 3).

    Notes
    -----
    Neither setup stage can run under ``jit``: ``setup_geometry`` calls the
    orbit provider's untraceable SciPy spline, and ``setup_time_freq`` is
    one-off grid arithmetic with nothing to gain. Both run eagerly in
    ``__init__`` and store ``jnp`` arrays, so the apply stage gets
    device-resident data and pays no host transfer per likelihood evaluation.

    Those arrays must be passed *explicitly* into the jitted apply functions
    rather than read off ``self``. This is a JAX requirement, not a style
    choice, and the same rule ``WDM_transform`` documents above its own filter
    methods: a jitted body runs only at trace time, so anything read off a
    static ``self`` freezes into the compiled code as a literal - rebuilding
    the attribute would not invalidate the cache, and stale values would be
    returned silently.
    """

    def __init__(self,
           wdm_class,
           orbits_class,
           t0 : float = 0.,
           max_lag_L : int = 25,
           num_interp_points : int = 16,
           link_convention : tuple = DEFAULT_DIRECTED_LINKS,
           ) -> None:
        r"""
        Initialize the LISAresponse class.

        Both setup stages run here. They are pure functions of the arguments,
        so no information arriving later can change their result and there is
        nothing to defer. Treat the resulting object as immutable: to change
        the WDM grid, build a new instance.

        Parameters
        ----------
        wdm_class : WDM.WDM_transform
            The WDM transform fixing the time-frequency tiling. Must expose
            ``dt``, ``N``, ``Nf``, ``Nt``, ``dT``, ``dF`` and
            ``build_time_delay_filter_interpolants``.
        orbits_class : lisaorbits.Orbits
            LISA orbit provider exposing ``compute_position(t, sc=[...])``,
            already loaded by the caller.
        t0 : float
            Absolute time of the first sample, in seconds. Must lie inside the
            span covered by ``orbits_class``. Optional.
        max_lag_L : int
            Maximum retained wavelet time-pixel lag, :math:`L`. Must satisfy
            :math:`0 < L < N_t`. Optional.
        num_interp_points : int
            Number of interpolation points across
            :math:`-\Delta T/2 \leq \delta \leq \Delta T/2`. Optional.
        link_convention : tuple
            Directed links as ``(sender, receiver)`` spacecraft index pairs,
            each drawn from :math:`\{1,2,3\}` with :math:`s \neq r`. The given
            order is preserved and indexes axis 0 of every geometry array.
            Pass a subset to build geometry for fewer links, for example
            ``((2, 1),)`` when debugging a single arm. Optional.

        Returns
        -------
        None
        """
        self.wdm_class = wdm_class
        self.orbits_class = orbits_class
        self.t0 = float(t0)
        self.max_lag_L = int(max_lag_L)
        self.num_interp_points = int(num_interp_points)
        self.link_convention = tuple((int(s), int(r)) for s, r in link_convention)
        self.num_links = len(self.link_convention)

        self.validate_parameters()

        self.setup_time_freq()
        self.setup_geometry()

    def validate_parameters(self) -> None:
        r"""
        Validate the parameters provided to the LISAresponse __init__ method.
        Raises an AssertionError if any parameters are invalid.

        The orbit time span is deliberately not checked here. Providers are
        duck-typed and analytic ones such as ``EqualArmlengthOrbits`` are
        unbounded, so there is no span to test against. An out-of-range ``t0``
        surfaces as an interpolation-bounds error from ``setup_geometry``.

        Returns
        -------
        None
        """
        for attribute in ("dt", "N", "Nf", "Nt", "dT", "dF",
                          "build_time_delay_filter_interpolants"):
            assert hasattr(self.wdm_class, attribute), \
                        f"wdm_class must expose {attribute!r}, got " \
                        f"{type(self.wdm_class).__name__}."

        assert callable(getattr(self.orbits_class, "compute_position", None)), \
                    f"orbits_class must expose a callable compute_position, " \
                    f"got {type(self.orbits_class).__name__}."

        assert np.isfinite(self.t0), \
                    f"t0 must be finite, got {self.t0=}."

        assert 0 < self.max_lag_L < int(self.wdm_class.Nt), \
                    f"max_lag_L must satisfy 0<L<Nt={int(self.wdm_class.Nt)}, " \
                    f"got {self.max_lag_L=}."

        assert self.num_interp_points > 1, \
                    f"num_interp_points must exceed 1, got " \
                    f"{self.num_interp_points=}."

        assert self.num_links > 0, \
                    "At least one directed link is required, got none."

        for sender, receiver in self.link_convention:
            assert sender in (1, 2, 3) and receiver in (1, 2, 3), \
                        f"Spacecraft indices are 1-based and must lie in " \
                        f"{{1,2,3}}, got link {(sender, receiver)}."

            assert sender != receiver, \
                        f"A link must join two distinct spacecraft, got " \
                        f"{(sender, receiver)}."

        assert len(set(self.link_convention)) == self.num_links, \
                    f"Directed links must be distinct, got {self.link_convention}."

    def setup_time_freq(self) -> None:
        r"""
        Precompute the time-frequency elements.

        Recovers the tiling from ``wdm_class`` and anchors it to the absolute
        timeline at :math:`t_0`. Two grids are built: the sample grid

        .. math::

            t_k = t_0 + k \delta t , \quad k\in\{0,\ldots,N-1\} ,

        and the wavelet pixel-centre grids

        .. math::

            t_n = t_0 + n \Delta T , \quad n\in\{0,\ldots,N_t-1\} ,

        .. math::

            f_m = m \Delta F , \quad m\in\{0,\ldots,N_f-1\} .

        Only :math:`t_n` is used by ``setup_geometry``; the geometry is never
        evaluated at the :math:`N` sample times.

        The time-delay filter tables are also built here, by delegating to
        ``wdm_class.build_time_delay_filter_interpolants``. They are tabulated
        on :math:`-\Delta T/2 \leq \delta \leq \Delta T/2` because the integer
        part of any delay is carried by the lag index :math:`\ell` and only
        the fractional residual is interpolated. That range is fixed by the
        WDM tiling alone, so the tables need nothing from the orbits and this
        stage stays independent of ``setup_geometry``.

        Returns
        -------
        None
        """
        self.dt = float(self.wdm_class.dt)
        self.N = int(self.wdm_class.N)
        self.Nf = int(self.wdm_class.Nf)
        self.Nt = int(self.wdm_class.Nt)
        self.dT = float(self.wdm_class.dT)
        self.dF = float(self.wdm_class.dF)

        # Derived grids
        self.times = self.t0 + jnp.arange(self.N) * self.dt
        self.wavelet_times = self.t0 + jnp.arange(self.Nt) * self.dT
        self.wavelet_freqs = jnp.arange(self.Nf) * self.dF

        # Time-delay filter interpolants, shape (2, 2L+1, num_interp_points)
        self.filter_tables = self.wdm_class.build_time_delay_filter_interpolants(
            self.max_lag_L,
            self.num_interp_points,
        )

    def setup_geometry(self) -> None:
        r"""
        Precompute the orbit elements.

        For every directed link :math:`(s, r)` this evaluates the three
        sky-independent quantities needed by the WDM response, on the wavelet
        centre-time grid :math:`t_n` only:

        - the light-travel time :math:`\ell_{sr}(t_n)`;
        - the link unit vector :math:`\hat{r}_{sr}(t_n)`;
        - the receiver position :math:`x_r(t_n)` at reception time.

        These depend on the orbits and the WDM tiling but not on the source
        sky position or the waveform, so they are built once and reused across
        every likelihood evaluation.

        The solve itself lives in ``response_sub.build_link_geometry`` and runs
        on the host in ``numpy``, since orbit providers cannot be traced. The
        results are moved to the device here, at the boundary.

        Returns
        -------
        None
        """
        light_travel_time, link_direction, receiver_position = (
            build_link_geometry(
                np.asarray(self.wavelet_times),
                self.orbits_class,
                self.link_convention,
            )
        )

        self.light_travel_time = jnp.asarray(light_travel_time)
        self.link_direction = jnp.asarray(link_direction)
        self.receiver_position = jnp.asarray(receiver_position)

    def one_arm_response_time_domain(self,
                                     theta,
                                     phi,
                                     h_plus,
                                     h_cross,
                                     in_domain : str = "time",
                                     out_domain : str = "time",
                                     link = None,
                                     num_integral_points : int = 10,
                                     num_taps : int = 12,
                                     time_chunk_size : int = None,
                                     memory_budget_bytes : float = 2 ** 30):
        r"""
        Compute the one-arm response in the time domain.

        This is the direct time-domain formulation: for each directed link the
        wave is integrated along the null geodesic joining emission to
        reception, giving the link delay fluctuation

        .. math::

            \delta \ell_{sr}(t) =
                \frac{1}{2\left(1 - \hat{\Omega}\cdot\hat{r}_{sr}\right)}
                \int_{\xi_s}^{\xi_r}
                    \left[ F^+ h_+(\xi) + F^\times h_\times(\xi) \right]
                \mathrm{d}\xi ,

        taken between the retarded wave phases at emission and reception,

        .. math::

            \xi_r = t - \hat{\Omega}\cdot x_r(t)/c , \qquad
            \xi_s = t - \ell_{sr}(t)
                      - \hat{\Omega}\cdot x_s(t-\ell_{sr})/c ,

        both formed from the positions the geometry solve produced rather than
        from an identity between them. The full statement of the model, and
        what that choice buys, is in
        ``response_sub.one_arm_time_domain_chunk``. The result is a *time*, in
        seconds, not the optical path length :math:`c\,\delta\ell_{sr}`.

        This is the slow, reference path. It is the WDM path's yardstick, not
        its competitor: the quadrature is a straight trapezoidal rule with no
        approximation beyond node count and waveform interpolation, so it is
        the thing to compare against when the local-delay approximation in the
        wavelet response is under suspicion. Two costs follow from that.
        First, the geometry cannot be reused - the stored arrays live on the
        :math:`N_t` wavelet centre times, whereas this needs all :math:`N`
        sample times, so the light-travel-time solve is rerun here at
        :math:`N_f` times the density. Second, the waveform is interpolated
        ``num_integral_points`` times per link and sample.

        Batching follows the WDM transform's convention: leading axes are free
        and broadcast, with the array's own axes last. ``theta`` and ``phi``
        broadcast against the leading axes of ``h_plus`` and ``h_cross``, so a
        scalar sky position with a stack of waveforms, a stack of sky positions
        with one waveform, and matched stacks of both all work.

        Parameters
        ----------
        theta : jnp.ndarray or float
            Polar angle of the source direction, in radians. Array
            shape=(...,), broadcast against ``phi`` and the waveforms.
        phi : jnp.ndarray or float
            Azimuthal angle of the source direction, in radians. Array
            shape=(...,), broadcast against ``theta`` and the waveforms.
        h_plus : jnp.ndarray
            Plus polarisation, in the domain named by ``in_domain``. Array
            shape=(..., N).
        h_cross : jnp.ndarray
            Cross polarisation, in the domain named by ``in_domain``. Array
            shape=(..., N).
        in_domain : str
            Domain in which ``h_plus`` and ``h_cross`` are supplied.
            ``"time"`` is a real series on the ``times`` grid, shape (..., N);
            ``"wdm"`` is its wavelet coefficients ``wdm_class.dwt(h)``, shape
            (..., Nt, Nf). The response model is a time-domain one, so
            ``"wdm"`` input is inverted with ``wdm_class.idwt`` first, and that
            round trip - not the response - sets the accuracy of this option:
            raise ``wdm_class`` :math:`q` towards :math:`N_t/2`, and set
            ``calc_m0=True`` if any power sits in the lowest frequency band,
            where the default is wrong at the tens of percent level. Passing
            the time series is exact and costs nothing. Optional.
        out_domain : str
            Domain the coefficients are returned in, ``"time"`` or ``"wdm"``,
            with the same conventions. ``"wdm"`` applies ``wdm_class.dwt`` to
            the time-domain result, one link at a time. Optional.
        link : None or int or tuple or list
            Which directed links to evaluate. ``None`` gives all of
            ``link_convention``. A single index or ``(sender, receiver)`` pair
            drops the link axis from the result; a list keeps it. See
            ``utils.utils.resolve_link_selection``. Optional.
        num_integral_points : int
            Number of trapezoidal nodes per link integral, :math:`P`, at least
            two. Convergence is :math:`(2\pi f \ell_{sr})^2/12(P-1)^2`, so
            what matters is the wave cycles inside one arm crossing, not the
            sample count: ten nodes leave :math:`2.6\times10^{-5}` in the
            record interior for a millihertz source, and only the edges (see
            Notes) resist more nodes. Raise it for a shorter arm or a higher
            frequency. Optional.
        num_taps : int
            Stencil width for reconstructing the waveform at each quadrature
            node, even and at least two. This is an error source independent of
            ``num_integral_points`` and neither knob repairs the other; twelve
            keeps it negligible across the band, EMRI harmonics included. See
            ``utils.utils.interpolate_uniform_grid``. Optional.
        time_chunk_size : int
            Length of the time blocks the calculation is broken into. When
            ``None`` it is chosen from ``memory_budget_bytes``. Set it
            explicitly only to reproduce a particular blocking; the result does
            not otherwise depend on it. Optional.
        memory_budget_bytes : float
            Ceiling on the peak allocation, in bytes. Raising it lets longer
            blocks run at fewer host round trips. Optional.

        Returns
        -------
        response : jnp.ndarray
            Link delay fluctuations :math:`\delta \ell_{sr}`, in seconds, or
            their DFT when ``out_domain`` is ``"frequency"``. Array
            shape=(..., num_selected_links, N) - or (..., num_selected_links,
            Nt, Nf) in the ``"wdm"`` domain - with the link axis absent when
            ``link`` selects a single link with a scalar specifier.

        Notes
        -----
        Accuracy degrades near either end of the record and no node count
        rescues it: once the integration window straddles :math:`t_0` or the
        last sample the integrand steps where the waveform stops, and the
        trapezoidal rule is only first order on a discontinuity. The affected
        band is the retarded-phase excursion, up to 499 s - a hundred samples
        at 5 s cadence - at each end, and over a 4096-sample record it alone
        lifts the error from :math:`2.6\times10^{-5}` to
        :math:`1.4\times10^{-3}`. This is a property of a finite record, not
        of the method, and the wavelet path has its own edge behaviour, so
        compare the two on the interior or pad both.
        """
        assert in_domain in ("time", "wdm"), \
                    f"in_domain must be 'time' or 'wdm', got {in_domain!r}."

        assert out_domain in ("time", "wdm"), \
                    f"out_domain must be 'time' or 'wdm', got {out_domain!r}."

        assert int(num_integral_points) >= 2, \
                    f"num_integral_points must be at least two to form a " \
                    f"trapezoid, got {num_integral_points=}."

        num_integral_points = int(num_integral_points)

        link_indices, drop_link_axis = resolve_link_selection(
            link, self.link_convention,
        )
        links = tuple(self.link_convention[index] for index in link_indices)
        num_selected = len(links)

        h_plus = jnp.asarray(h_plus)
        h_cross = jnp.asarray(h_cross)

        expected = (self.Nt, self.Nf) if in_domain == "wdm" else (self.N,)
        for name, array in (("h_plus", h_plus), ("h_cross", h_cross)):
            assert array.shape[-len(expected):] == expected, \
                        f"With {in_domain=}, {name} must end in axes " \
                        f"{expected}, got {array.shape=}."

        # The response model is a time-domain one, so wavelet input is
        # inverted here and everything downstream sees a series on the sample
        # grid.
        if in_domain == "wdm":
            h_plus = self.wdm_class.idwt(h_plus)
            h_cross = self.wdm_class.idwt(h_cross)

        # Leading axes broadcast together; the batch is then flattened to one
        # axis so the jitted kernel sees a fixed rank.
        theta = jnp.asarray(theta, dtype=float)
        phi = jnp.asarray(phi, dtype=float)

        batch_shape = jnp.broadcast_shapes(
            theta.shape, phi.shape, h_plus.shape[:-1], h_cross.shape[:-1],
        )
        batch_size = int(np.prod(batch_shape, dtype=int))

        u, v, n = build_sky_basis(
            jnp.broadcast_to(theta, batch_shape).reshape(batch_size),
            jnp.broadcast_to(phi, batch_shape).reshape(batch_size),
        )

        h_plus = jnp.broadcast_to(
            h_plus, batch_shape + (self.N,)).reshape(batch_size, self.N)
        h_cross = jnp.broadcast_to(
            h_cross, batch_shape + (self.N,)).reshape(batch_size, self.N)

        if time_chunk_size is None:
            time_chunk_size = resolve_time_chunk_size(
                self.N,
                batch_size,
                num_selected,
                output_domain=out_domain,
                memory_budget_bytes=memory_budget_bytes,
            )
        else:
            time_chunk_size = min(int(time_chunk_size), self.N)

            assert time_chunk_size > 0, \
                        f"time_chunk_size must be positive, got " \
                        f"{time_chunk_size=}."

        blocks = []
        for start in range(0, self.N, time_chunk_size):
            stop = min(start + time_chunk_size, self.N)

            # Held on the host: the geometry solve calls the orbit provider,
            # and the block is padded to a fixed length by repeating its last
            # reception time so the jitted kernel compiles once.
            chunk_times = self.t0 + np.arange(start, stop) * self.dt
            padding = time_chunk_size - chunk_times.size
            if padding:
                chunk_times = np.pad(chunk_times, (0, padding), mode="edge")

            light_travel_time, link_direction, receiver_position, \
                sender_position = build_link_geometry(
                    chunk_times,
                    self.orbits_class,
                    links,
                    return_sender_position=True,
                )

            block = one_arm_time_domain_chunk(
                h_plus,
                h_cross,
                u,
                v,
                n,
                jnp.asarray(light_travel_time),
                jnp.asarray(link_direction),
                jnp.asarray(receiver_position),
                jnp.asarray(sender_position),
                jnp.asarray(chunk_times),
                self.t0,
                self.dt,
                num_integral_points=num_integral_points,
                num_taps=num_taps,
            )

            blocks.append(block[..., :stop - start] if padding else block)

        response = jnp.concatenate(blocks, axis=-1)

        if out_domain == "wdm":
            # One link at a time: the transform's complex intermediate is the
            # size of its input, so driving all links through it at once would
            # multiply the peak by the number of links for no gain.
            response = jnp.stack(
                [self.wdm_class.dwt(response[:, index])
                 for index in range(num_selected)],
                axis=1,
            )
            response = response.reshape(
                batch_shape + (num_selected, self.Nt, self.Nf))
        else:
            response = response.reshape(batch_shape + (num_selected, self.N))

        # The link axis sits just after the batch axes in either domain.
        if drop_link_axis:
            response = jnp.squeeze(response, axis=len(batch_shape))

        return response

    def one_arm_response(self,
                        theta,
                        phi,
                        h_plus,
                        h_cross,
                        in_domain,
                        out_domain,
                        link,):
        r"""
        Compute the one-arm response in the time-freuquency domain.
        """

    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the time-frequency to time-frequency lisa respone for all six links.
        """
