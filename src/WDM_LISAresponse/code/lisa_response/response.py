#this is where the LISAresponse class will live
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from .response_sub import (
    broadcast_sources,
    build_link_geometry,
    finalise_response,
    one_arm_response_wdm,
    one_arm_time_domain_blocked,
)
from ..utils.utils import (
    Link,
    resolve_link_selection,
    resolve_time_chunk_size,
)


#: Directed links as ``Link(sender, receiver)`` spacecraft index pairs.
#:
#: This is ``lisaorbits.LINKS`` written in sender-first form. lisaorbits labels
#: links by the integer code ``AB`` meaning *receiver* ``A`` and *emitter*
#: ``B``, so its ``[12, 23, 31, 13, 32, 21]`` decodes element for element to
#: the links below. Keeping the same order means axis 0 of every geometry array
#: lines up with the orbit file's own link ordering.
DEFAULT_DIRECTED_LINKS = (Link(2, 1), Link(3, 2), Link(1, 3),
                          Link(3, 1), Link(2, 3), Link(1, 2))


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
        Directed links as ``Link`` pairs. The given order indexes axis 0 of
        every geometry array, so it is part of the class contract: downstream
        code reads link ``i`` as ``link_convention[i]``.
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
            Directed links as ``Link(sender, receiver)`` pairs, each drawn
            from :math:`\{1,2,3\}` with :math:`s \neq r`. Plain
            ``(sender, receiver)`` tuples are accepted too, since a ``Link``
            compares equal to one. The given order is preserved and indexes
            axis 0 of every geometry array. Pass a subset to build geometry for
            fewer links, for example ``(Link(2, 1),)`` when debugging a single
            arm. Optional.

        Returns
        -------
        None
        """
        self.wdm_class = wdm_class
        self.orbits_class = orbits_class
        self.t0 = float(t0)
        self.max_lag_L = int(max_lag_L)
        self.num_interp_points = int(num_interp_points)
        self.link_convention = tuple(Link(int(s), int(r)) for s, r in link_convention)
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
        link : None or int or Link or sequence
            Which directed links to evaluate. ``None`` gives all of
            ``link_convention``. A single ``Link(sender, receiver)`` or index
            drops the link axis from the result; a sequence of either keeps it.
            See ``utils.utils.resolve_link_selection``. Optional.
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
            Ceiling on *working* memory, in bytes - the per-block geometry and
            quadrature temporaries, which at four years and six links run to
            tens of gibibytes if unblocked. The returned array is not counted:
            its size follows from the arguments and is the caller's to judge.
            Raising this lets longer blocks run at fewer host round trips.
            Optional.

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
        assert out_domain in ("time", "wdm"), \
                    f"out_domain must be 'time' or 'wdm', got {out_domain!r}."

        assert int(num_integral_points) >= 2, \
                    f"num_integral_points must be at least two to form a " \
                    f"trapezoid, got {num_integral_points=}."

        link_indices, drop_link_axis = resolve_link_selection(
            link, self.link_convention,
        )
        links = tuple(self.link_convention[index] for index in link_indices)

        h_plus, h_cross, theta, phi, batch_shape = broadcast_sources(
            self.wdm_class, theta, phi, h_plus, h_cross, in_domain,
            target_domain="time",
        )

        if time_chunk_size is None:
            time_chunk_size = resolve_time_chunk_size(
                self.N,
                h_plus.shape[0],
                len(links),
                memory_budget_bytes=memory_budget_bytes,
            )
        time_chunk_size = min(int(time_chunk_size), self.N)

        assert time_chunk_size > 0, \
                    f"time_chunk_size must be positive, got " \
                    f"{time_chunk_size=}."

        response = one_arm_time_domain_blocked(
            self.orbits_class,
            links,
            self.t0,
            self.dt,
            self.N,
            time_chunk_size,
            h_plus,
            h_cross,
            theta,
            phi,
            num_integral_points=int(num_integral_points),
            num_taps=num_taps,
        )

        return finalise_response(
            self.wdm_class, response, "time", out_domain,
            batch_shape, len(links), drop_link_axis,
        )

    def one_arm_response(self,
                         theta,
                         phi,
                         h_plus,
                         h_cross,
                         in_domain : str = "wdm",
                         out_domain : str = "wdm",
                         link = None,
                         lag_block : int = None):
        r"""
        Compute the one-arm response in the time-frequency domain.

        The production path. For each directed link the retarded-phase integral
        is evaluated in closed form per wavelet pixel and the result is carried
        to the retarded phase by the WDM time-shift operator, never returning
        to the time series:

        .. math::

            w^{sr}_{nm} = \mathcal{S}[\delta_n] \left\{
                \tfrac{1}{2} \ell_{sr} \,
                \mathrm{sinc}\!\left( 2 f_m \tau_n \right)
                \left( F^+ w^{h_+}_{nm}
                     + F^\times w^{h_\times}_{nm} \right) \right\} ,

        with :math:`\tau_n` the half-window, :math:`\delta_n` the delay to the
        window centre, and :math:`\mathcal{S}` the variable time shift. The
        model is stated in full in ``response_sub.one_arm_response_wdm``. The
        result is a *time*, in seconds, not the optical path length.

        Evaluating the sinc at the pixel's own centre frequency is the
        local-delay approximation, which costs an L2 of order
        :math:`10^{-4}` - the dominant error in this pipeline, and what
        ``one_arm_response_time_domain`` exists to measure.

        Batching follows the WDM transform: leading axes are free and
        broadcast, with the array's own axes last. ``theta`` and ``phi``
        broadcast against the leading axes of the waveforms, so one sky
        position with a stack of waveforms, a stack of sky positions with one
        waveform, and matched stacks all work.

        Parameters
        ----------
        theta : jnp.ndarray or float
            Polar angle of the source direction, in radians. Array
            shape=(...,), broadcast against ``phi`` and the waveforms.
        phi : jnp.ndarray or float
            Azimuthal angle of the source direction, in radians. Array
            shape=(...,), broadcast against ``theta`` and the waveforms.
        h_plus : jnp.ndarray
            Plus polarisation, in the domain named by ``in_domain``.
        h_cross : jnp.ndarray
            Cross polarisation, in the domain named by ``in_domain``.
        in_domain : str
            Domain in which the waveforms are supplied. ``"wdm"`` is wavelet
            coefficients, shape (..., Nt, Nf); ``"time"`` is a real series on
            the ``times`` grid, shape (..., N), transformed here with
            ``wdm_class.dwt``. Optional.
        out_domain : str
            Domain the coefficients are returned in, ``"wdm"`` or ``"time"``,
            the latter via ``wdm_class.idwt``. Optional.
        link : None or int or Link or sequence
            Which directed links to evaluate. ``None`` gives all of
            ``link_convention``. A single ``Link(sender, receiver)`` or index
            drops the link axis from the result; a sequence of either keeps it.
            Optional.
        lag_block : int
            Lags handled per iteration of the shift's inner sum. ``None``
            unrolls it, which is fastest while :math:`L` is small; past
            :math:`L \sim 30` the unrolled graph costs more in scheduling and
            compile time than it saves, and a small block such as 8 wins.
            Optional.

        Returns
        -------
        response : jnp.ndarray
            Link delay fluctuations :math:`\delta \ell_{sr}`, in seconds.
            Array shape=(..., num_selected_links, Nt, Nf), or
            (..., num_selected_links, N) in the ``"time"`` domain, with the
            link axis absent when ``link`` selects a single link with a
            scalar specifier.

        Notes
        -----
        The time shift wraps periodically at the ends of the record rather than
        running off it, so power leaving one end reappears at the other. Zero
        padding the waveform is the caller's responsibility, as it is for
        ``wdm_class.apply_variable_time_shift`` itself, and the amount that
        buys a clean record is :math:`\lceil L/2 \rceil` wavelet rows at
        *each* end - :math:`\lceil L/2 \rceil N_f` samples per side, with the
        leading count even. See ``response_sub.apply_batch_shift`` for why both
        ends and why half of :math:`L`. This is the wavelet path's counterpart
        to the edge behaviour noted on ``one_arm_response_time_domain``, and
        the two differ, so compare the paths on the interior of the record.
        """
        assert out_domain in ("time", "wdm"), \
                    f"out_domain must be 'time' or 'wdm', got {out_domain!r}."

        link_indices, drop_link_axis = resolve_link_selection(
            link, self.link_convention,
        )
        indices = jnp.asarray(link_indices)

        wavelet_plus, wavelet_cross, theta, phi, batch_shape = (
            broadcast_sources(
                self.wdm_class, theta, phi, h_plus, h_cross, in_domain,
                target_domain="wdm",
            )
        )

        response = one_arm_response_wdm(
            self.wdm_class,
            self.filter_tables,
            wavelet_plus,
            wavelet_cross,
            theta,
            phi,
            self.light_travel_time[indices],
            self.link_direction[indices],
            self.receiver_position[indices],
            self.wavelet_freqs,
            lag_block=lag_block,
        )

        return finalise_response(
            self.wdm_class, response, "wdm", out_domain,
            batch_shape, len(link_indices), drop_link_axis,
        )

    def __call__(self,
                 theta,
                 phi,
                 h_plus,
                 h_cross,
                 lag_block : int = None) -> jnp.ndarray:
        r"""
        Compute the wavelet-domain response for all links.

        The shortest path through the class, and the one a likelihood should
        use: wavelet coefficients in, wavelet coefficients out, every link in
        ``link_convention``, no domain conversion at either end. It is
        ``one_arm_response`` at its defaults, which is to say the whole
        calculation is one jitted call per link with nothing on the host.

        Parameters
        ----------
        theta : jnp.ndarray or float
            Polar angle of the source direction, in radians.
            Array shape=(...,).
        phi : jnp.ndarray or float
            Azimuthal angle of the source direction, in radians.
            Array shape=(...,).
        h_plus : jnp.ndarray
            Wavelet coefficients of :math:`h_+`. Array shape=(..., Nt, Nf).
        h_cross : jnp.ndarray
            Wavelet coefficients of :math:`h_\times`.
            Array shape=(..., Nt, Nf).
        lag_block : int
            Lags per iteration of the shift's inner sum. Optional.

        Returns
        -------
        response : jnp.ndarray
            Link delay fluctuations, in seconds, as wavelet coefficients.
            Array shape=(..., num_links, Nt, Nf).
        """
        return self.one_arm_response(
            theta, phi, h_plus, h_cross, lag_block=lag_block,
        )
