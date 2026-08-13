#this is where the LISAresponse class will live
import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple
from functools import partial

from .response_sub import build_link_geometry


#: Directed links as ``(sender, receiver)`` spacecraft index pairs.
DEFAULT_DIRECTED_LINKS = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))


class LISAresponse:
    r"""
    This class is used to compute the time-frequency or time domain LISA responses

    Attributes
    ----------
    wdm_class : WDM.WDM_transform
        The WDM transform defining the time-frequency tiling. All time and
        frequency grids below are derived from this object, so it fixes the
        resolution of every downstream stage.
    orbits_class : lisaorbits.Orbits
        LISA orbit provider exposing ``compute_position(t, sc=[...])``. Any
        object satisfying that one-method contract is accepted.
    t0 : float
        Absolute time of the first sample, :math:`t_0`, in seconds. This places
        the WDM grid onto the orbit provider's timeline; the WDM transform
        itself only knows relative times.
    links : tuple
        The six directed links as ``(sender, receiver)`` spacecraft index
        pairs, :math:`(s, r)`. Ordering is fixed by
        ``DEFAULT_DIRECTED_LINKS`` and indexes axis 0 of every geometry array.
    dt : float
        Time series cadence, :math:`\delta t`, in seconds. Taken from
        ``wdm_class``.
    N : int
        Length of the input time series, :math:`N`. Taken from ``wdm_class``.
    Nf : int
        Number of wavelet frequency bands, :math:`N_f`. Taken from
        ``wdm_class``.
    Nt : int
        Number of wavelet time bands, :math:`N_t = N/N_f`. Taken from
        ``wdm_class``.
    dT : float
        Wavelet time resolution, :math:`\Delta T = N_f \delta t`, in seconds.
    dF : float
        Wavelet frequency resolution, :math:`\Delta F = 1/(2 N_f \delta t)`,
        in hertz. Satisfies :math:`\Delta F \Delta T = 1/2`.
    times : np.ndarray
        Sample times of the time series, :math:`t_k = t_0 + k \delta t` for
        :math:`k\in\{0,\ldots,N-1\}`. Array shape=(N,).
    wavelet_times : np.ndarray
        Central times of the wavelet pixels, :math:`t_n = t_0 + n \Delta T`
        for :math:`n\in\{0,\ldots,N_t-1\}`. Array shape=(Nt,). All link
        geometry is evaluated on this grid, not on ``times``.
    wavelet_freqs : jnp.ndarray
        Central frequencies of the wavelet pixels, :math:`f_m = m \Delta F`
        for :math:`m\in\{0,\ldots,N_f-1\}`. Array shape=(Nf,).
    max_lag_L : int
        Maximum retained wavelet time-pixel lag, :math:`L`, in the time-delay
        filter tables.
    num_interp_points : int
        Number of interpolation points tabulated across
        :math:`-\Delta T/2 \leq \delta \leq \Delta T/2`.
    filter_tables : jnp.ndarray
        Time-delay filter tables :math:`T'_\ell(\delta)` and
        :math:`T_\ell(\delta)`, built by ``wdm_class``.
        Array shape=(2, 2L+1, num_interp_points). Index 0 is :math:`T'_\ell`,
        index 1 is :math:`T_\ell`.
    light_travel_time : jnp.ndarray
        Light-travel time :math:`\ell_{sr}(t_n)` for each directed link,
        in seconds. Array shape=(6, Nt).
    link_direction : jnp.ndarray
        Unit vector :math:`\hat{r}_{sr}(t_n)` pointing from the sender at
        emission to the receiver at reception. Array shape=(6, Nt, 3).
    receiver_position : jnp.ndarray
        Receiver position :math:`x_r(t_n)` at reception time, in metres.
        Array shape=(6, Nt, 3).

    Notes
    -----
    Neither setup stage can run under ``jit``. ``setup_geometry`` calls
    ``orbits_class.compute_position``, which evaluates a SciPy spline and
    cannot be traced; ``setup_time_freq`` is one-off grid arithmetic with
    nothing to gain. Both therefore run eagerly, once, and store their results
    as ``jnp`` arrays so the jitted apply stage receives device-resident data
    and pays no host transfer per likelihood evaluation.

    Those arrays must be passed *explicitly* into the jitted apply functions
    rather than read off ``self`` inside them. This is a JAX requirement, not
    a style choice, and is the same rule ``WDM_transform`` documents above its
    own filter methods: a jitted function body runs only during tracing, so
    anything it reads off a static ``self`` is frozen into the compiled code
    as a literal. Rebuilding an attribute would not invalidate the cache, and
    the stale values would be returned silently.
    """

    def __init__(self,
           wdm_class,
           orbits_class,
           t0 : float = 0.,
           max_lag_L : int = 25,
           num_interp_points : int = 1024
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

        Returns
        -------
        None
        """
        self.wdm_class = wdm_class
        self.orbits_class = orbits_class
        self.t0 = float(t0)
        self.max_lag_L = int(max_lag_L)
        self.num_interp_points = int(num_interp_points)
        self.links = DEFAULT_DIRECTED_LINKS

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
                self.links,
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
                                     in_domain,
                                     out_domain,
                                     link,):
        r"""
        Compute the one-arm response in the time domain.
        """

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
    