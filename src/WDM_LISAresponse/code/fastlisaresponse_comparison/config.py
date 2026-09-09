r"""Locked parameters for the ``fastlisaresponse`` comparison.

Every number defining the study lives here once, so the two pipelines cannot
drift onto different grids, sources or orbits. ``DEFAULT`` is the real run;
``SMOKE`` is a thirty-day version for shaking out conventions before spending
minutes on the time-domain reference path. Both share :math:`\Delta F`, and so
the same source frequency, because :math:`\Delta F = 1/(2 \delta t N_f)` does
not depend on the record length.

Nothing here imports ``fastlisaresponse``. Angle and unit translation between
the two codes belongs in ``conventions``.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..utils.utils import wdm_padding_info
from ..waveforms.waveforms import generate_windowed_waveform


#: Orbit models this study can build. ``"equal_armlength"`` is closed form;
#: ``"esa"`` reads an orbit file into ``lisaorbits.InterpolatedOrbits``.
ORBIT_MODELS = ("equal_armlength", "esa")

#: The ESA trailing-orbits file beside the example notebooks. Resolved from
#: this file's location, so it is right for a source checkout and may be
#: absent for an installed wheel - hence nothing validates it until asked for.
EXAMPLES_ORBIT_FILE = str(
    Path(__file__).resolve().parents[4] / "examples" / "esa-trailing-orbits.h5"
)

#: Margin between an orbit file's first sample and ``t0``, in seconds. The
#: light-travel-time solve queries the sender *backwards* along its worldline,
#: and ``InterpolatedOrbits`` does not extrapolate by default.
ORBIT_LEAD_IN = 1280.0


@dataclass(frozen=True)
class ComparisonConfig:
    r"""The full parameter set for one comparison run.

    Frozen: derive variants with ``variant()``, which keeps ``__post_init__``
    on the path of every object that exists. ``build()`` turns one into a
    transform and a response, and ``waveform()`` generates the source it
    names, so a caller never assembles either by hand.

    Attributes
    ----------
    dt : float
        Sampling cadence, in seconds.
    Nf : int
        Number of WDM frequency bands.
    N : int
        *Padded* record length, in samples; the signal length follows by
        subtracting the guard band. This way round because the transform does
        length-:math:`N` FFTs, so :math:`N` wants to be a power of two.
    q : int
        WDM truncation parameter.
    max_lag_L : int
        Maximum retained wavelet time-pixel lag :math:`L`. 51 rather than the
        usual 25; see ``Notes``. Sets the guard band - see ``padding``.
    num_interp_points : int
        Interpolation points for the time-delay filter tables. See ``Notes``.
    t0 : float
        Absolute time of the first *padded* sample, in seconds.
    m_bin : int
        WDM frequency bin holding the binary, so :math:`f = m \Delta F`.
        Bin-centred, so the source occupies one row - which is what makes the
        residual map legible and the time-offset measurement in
        ``conventions`` exact.
    amp, fdot, iota, psi, phi0 : float
        Galactic binary parameters, from the ``fastlisaresponse`` tutorial
        binary. See ``waveform()``.
    tukey_alpha : float
        Tukey fraction applied by ``waveform()``, in :math:`[0, 1]`. Zero
        leaves the record untapered.
    theta, phi : float
        Sky position in this package's convention - polar angle from
        :math:`+z`, azimuth in the SSB frame, :math:`\hat{n}` towards the
        source, as ``build_sky_basis`` defines them. ``fastlisaresponse`` uses
        ecliptic longitude and *latitude* instead;
        ``conventions.to_flr_angles`` translates.
    orbits_model : str
        One of ``ORBIT_MODELS``. See ``build_orbits()``.
    orbit_file : str or None
        Orbit file path, required when ``orbits_model == "esa"``. Build these
        with ``for_esa()``, which also sets a workable ``t0``.

    Notes
    -----
    ``max_lag_L`` is 51 rather than 25, and it is the *only* knob that moves
    the shift error. Measured against an analytic reference - a tone shifted
    in closed form, not one setting compared with another - a single shift at
    :math:`m = 13`, :math:`\delta = 8.34` s leaves, in relative L2:

    ====== =========
    L      rel_L2
    ====== =========
    25     6.9e-06
    51     1.1e-07
    75     1.9e-08
    101    8.2e-09
    ====== =========

    Over the fourteen shifts of a TDI channel, :math:`L = 25` reaches
    :math:`10^{-4}` - level with the local-delay approximation this comparison
    exists to measure - while :math:`L = 51` reaches
    :math:`1.5\times10^{-6}`. The shift costs :math:`O(L)`, so 51 doubles the
    runtime and keeps the operator out of the measurement.
    """

    # Grid
    dt: float = 10.0
    Nf: int = 128
    N: int = 2 ** 22
    q: int = 16

    # Response
    max_lag_L: int = 51
    num_interp_points: int = 256
    # Only ever reached by an equal-armlength config; for_esa() overrides it
    # from the file. Not zero, so phase accumulation is exercised at an epoch
    # as realistic as the ESA one.
    t0: float = 3.15e7

    # Source
    m_bin: int = 13
    amp: float = 1.084702251e-22
    fdot: float = 2.446712e-16
    iota: float = 1.11820901
    psi: float = 2.3290324
    phi0: float = 4.91128699
    theta: float = 0.59022203
    phi: float = 5.22979888
    tukey_alpha: float = 0.01

    # Orbits
    orbits_model: str = "equal_armlength"
    orbit_file: str = None

    def __post_init__(self) -> None:
        """Validate. Raises rather than warns: a bad grid caught here costs a
        second, and caught later costs a run."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if self.Nf < 1:
            raise ValueError(f"Nf must be >= 1, got {self.Nf}.")
        if self.N % (2 * self.Nf) != 0:
            raise ValueError(
                f"N must be an even multiple of Nf, got N={self.N}, "
                f"Nf={self.Nf}."
            )
        if not 0 < self.max_lag_L < self.N // self.Nf:
            raise ValueError(
                f"max_lag_L must satisfy 0 < L < Nt, got L={self.max_lag_L}, "
                f"Nt={self.N // self.Nf}."
            )
        if not 0 < self.m_bin < self.Nf:
            raise ValueError(
                f"m_bin must satisfy 0 < m < Nf, got m={self.m_bin}, "
                f"Nf={self.Nf}."
            )
        if self.n_signal < 1:
            raise ValueError(
                f"The guard band leaves no signal: N={self.N} against "
                f"{4 * (self.guard_rows)} guard rows of "
                f"{self.Nf} samples. Raise N or lower max_lag_L."
            )
        if self.orbits_model not in ORBIT_MODELS:
            raise ValueError(
                f"orbits_model must be one of {ORBIT_MODELS}, got "
                f"{self.orbits_model!r}."
            )
        if self.orbits_model == "esa":
            if self.orbit_file is None:
                raise ValueError(
                    "orbits_model='esa' requires orbit_file to be set."
                )
            if not Path(self.orbit_file).is_file():
                raise FileNotFoundError(
                    f"orbit_file not found: {self.orbit_file}"
                )

        target = self.padding["target_length"]
        if target != self.N:
            raise ValueError(
                f"Padding does not close on N: wdm_padding_info returned "
                f"target_length={target} against N={self.N}."
            )

    # -- Tiling ------------------------------------------------------------

    @property
    def Nt(self) -> int:
        """Wavelet time rows, :math:`N_t = N / N_f`."""
        return self.N // self.Nf

    @property
    def dF(self) -> float:
        r"""Wavelet frequency spacing, :math:`1/(2 \delta t N_f)`, in Hz."""
        return 1.0 / (2.0 * self.dt * self.Nf)

    @property
    def dT(self) -> float:
        r"""Wavelet time spacing, :math:`\delta t N_f`, in seconds."""
        return self.dt * self.Nf

    @property
    def f_gb(self) -> float:
        r"""Binary frequency, :math:`m \Delta F`, in Hz."""
        return self.m_bin * self.dF

    # -- Padding -----------------------------------------------------------

    @property
    def guard_rows(self) -> int:
        r"""What ``wdm_padding_info`` adds per side, :math:`\lceil L/2 \rceil`."""
        return -(-self.max_lag_L // 2)

    @property
    def n_signal(self) -> int:
        """Unpadded signal length, in samples.

        Both pads are exact multiples of ``Nf``, so this is a whole number of
        wavelet rows and the signal starts on a row boundary.
        """
        rows = 2 * self.guard_rows 
        return self.N - 2 * rows * self.Nf

    @property
    def padding(self) -> dict:
        r"""Padding plan from ``utils.wdm_padding_info``.

        Hand ``pad_left``/``pad_right`` to ``utils.apply_zero_padding``, and
        keep the dictionary for ``utils.wdm_cut_info`` afterwards.

        Notes
        -----
        The guard is :math:`P = 2\lceil L/2 \rceil` rows per side, twice what
        ``wdm_padding_info`` adds alone and so :math:`\geq L`. The signal
        passes through two variable time shifts - one in ``one_arm_response``
        to reach the retarded phase, one in the TDI combination.
        """
        return wdm_padding_info(
            self.n_signal,
            self.Nf,
            max_lag_L=self.max_lag_L,
            extra_rows_left=self.guard_rows,
            extra_rows_right=self.guard_rows,
        )

    # -- Times -------------------------------------------------------------

    @property
    def T_total(self) -> float:
        """Padded record duration, in seconds."""
        return self.N * self.dt

    @property
    def T_signal(self) -> float:
        """Unpadded signal duration, in seconds."""
        return self.n_signal * self.dt

    @property
    def signal_times(self) -> np.ndarray:
        """Sample times of the *unpadded* signal, shape=(n_signal,).

        Starts at ``t0 + pad_left * dt``, not ``t0``: the epoch belongs to the
        first padded sample, and the signal begins after the leading guard.
        """
        start = self.t0 + self.padding["pad_left"] * self.dt
        return start + np.arange(self.n_signal, dtype=float) * self.dt

    @property
    def padded_times(self) -> np.ndarray:
        """Sample times of the padded record, shape=(N,)."""
        return self.t0 + np.arange(self.N, dtype=float) * self.dt

    # -- Source ------------------------------------------------------------

    def waveform(self, **overrides):
        r"""Generate the binary these parameters describe, tapered, unpadded.

        A thin call onto ``waveforms.generate_windowed_waveform`` with the
        ``"galactic_binary"`` model. It lives here because the parameters do:
        the point of this class is that the two pipelines cannot end up on
        different sources, and a free function taking a config would let a
        caller assemble the arguments itself.

        Pad afterwards with ``utils.apply_zero_padding`` and
        ``self.padding``. Both codes must then receive *this* array -
        ``conventions.PaddedReplay`` hands ``fastlisaresponse`` the padded
        record rather than letting it generate its own, so the waveform
        conventions cannot differ between them because neither ever sees a
        second version.

        ``t_ref`` is left to the generator, which defaults it to the first
        sample - already the right anchor here, since the grid starts at
        ``signal_times[0]``. Override it only to move the phase origin
        deliberately.

        Parameters
        ----------
        **overrides
            Any ``waveforms.generate_waveform_galactic_binary`` keyword, or
            ``tukey_alpha``, to override for this call.

        Returns
        -------
        h_plus, h_cross : np.ndarray
            Polarisations on ``self.signal_times``. Array shape=(n_signal,).

        """
        alpha = overrides.pop("tukey_alpha", self.tukey_alpha)

        model_kwargs = dict(
            amp=self.amp,
            f=self.f_gb,
            fdot=self.fdot,
            iota=self.iota,
            phi0=self.phi0,
            psi=self.psi,
        )
        model_kwargs.update(overrides)

        _, h_plus, h_cross = generate_windowed_waveform(
            "galactic_binary",
            t0=float(self.signal_times[0]),
            dt=self.dt,
            N=self.n_signal,
            model_kwargs=model_kwargs,
            window_function="tukey" if alpha > 0.0 else None,
            window_kwargs=dict(alpha=alpha),
        )

        return h_plus, h_cross

    # -- Diagnostics -------------------------------------------------------

    @property
    def grid_bytes(self) -> int:
        """One :math:`(N_t, N_f)` float64 grid, in bytes.

        The memory unit of the study: six hold the link responses, and the TDI
        combination must accumulate in place rather than stack all its terms.
        """
        return self.Nt * self.Nf * 8

    # -- Construction ------------------------------------------------------

    def variant(self, **changes) -> "ComparisonConfig":
        """Return a revalidated copy with ``changes`` applied."""
        return replace(self, **changes)

    def orbit_span(self) -> tuple:
        """First and last time the orbit file covers, in seconds.

        Reads the file. The closed-form provider is unbounded and reports an
        infinite span, so callers need no special case.
        """
        if self.orbits_model == "equal_armlength":
            return (-np.inf, np.inf)

        from lisaorbits.file import read_orbit_file

        data = read_orbit_file(self.orbit_file)
        return (float(data.t0), float(data.t0 + data.dt * (data.size - 1)))

    def build_orbits(self):
        """Construct the orbit provider named by ``orbits_model``.

        Imported lazily, so a configuration can be inspected and validated
        without touching ``lisaorbits``.

        The ESA branch goes through ``read_orbit_file`` into
        ``InterpolatedOrbits``, as ``wdm_lisatools_benchmark`` does, which
        preserves the file's absolute epoch. That epoch is around
        :math:`2.07\\times10^9` s, so a default ``t0`` will not fit inside it -
        hence the span check, which fails readably instead of letting the
        solve run off the end of a spline.

        Returns
        -------
        orbits : lisaorbits.Orbits
            A provider exposing ``compute_position(t, sc=[...])``.

        Notes
        -----
        **The comparison runs on the ESA file.** Not for precision - both are
        far inside the budget - but because second-generation TDI exists to
        handle flexing arms, and the analytic model flexes 120 times less.

        The two packages' analytic equal-armlength models are also not the
        same orbit: ``lisaorbits`` works in equatorial coordinates and
        ``lisatools.detector`` in ecliptic, so their positions differ by the
        obliquity, :math:`23.4393^\\circ` or :math:`6\\times10^{10}` m. That is
        description rather than physics - mapping the sky angles between
        frames would recover it - but reading the *same file* avoids the
        rotation entirely, since neither package converts on load. Positions
        then agree to 304 m. ``conventions.flr_wrapper_kwargs`` refuses a
        non-ESA configuration on these grounds.

        ``"equal_armlength"`` stays for the tests: closed
        form, no file, no epoch, unbounded in time. Its arms are rigid but its
        light-travel times are not - the solve is retarded, giving
        :math:`\\ell_{sr} \\approx L(1 + v_s\\cdot\\hat{r}_{sr}/c)`, which
        splits the two directions of a link by :math:`1.1\\times10^{-3}` s and
        drifts each by :math:`3\\times10^{-4}` s over 30 days. First-generation
        TDI does leave a residual here; the ESA orbits vary ~150 times faster,
        which is why the cancellation test belongs on them.
        """
        if self.orbits_model == "equal_armlength":
            from lisaorbits import EqualArmlengthOrbits

            return EqualArmlengthOrbits()

        from lisaorbits import InterpolatedOrbits
        from lisaorbits.file import read_orbit_file

        data = read_orbit_file(self.orbit_file)
        first = float(data.t0)
        last = float(data.t0 + data.dt * (data.size - 1))

        if self.t0 - ORBIT_LEAD_IN < first or self.t0 + self.T_total > last:
            raise ValueError(
                f"The record does not fit inside the orbit file. Needs "
                f"[{self.t0 - ORBIT_LEAD_IN:.6g}, "
                f"{self.t0 + self.T_total:.6g}] s, file covers "
                f"[{first:.6g}, {last:.6g}] s. Use "
                f"ComparisonConfig.for_esa(), which reads t0 from the file."
            )

        times = data.t0 + data.dt * np.arange(data.size)

        return InterpolatedOrbits(times, data.positions, data.velocities)

    @classmethod
    def for_esa(cls,
                orbit_file: str = None,
                *,
                lead_in: float = ORBIT_LEAD_IN,
                **changes) -> "ComparisonConfig":
        """Build a configuration on the ESA trailing orbits.

        The epoch comes from the file, not the caller. This is the supported
        way to move the study onto the real orbits; nothing else changes,
        since the tiling, source and conventions are all independent of the
        provider.

        Parameters
        ----------
        orbit_file : str
            Orbit file path. Defaults to ``EXAMPLES_ORBIT_FILE``. Optional.
        lead_in : float
            Seconds left before ``t0`` for the solve's backwards reach.
            Optional.
        **changes
            Any other field, as for ``variant()``.

        Returns
        -------
        config : ComparisonConfig
            A validated configuration on the ESA orbits.
        """
        from lisaorbits.file import read_orbit_file

        path = EXAMPLES_ORBIT_FILE if orbit_file is None else str(orbit_file)
        data = read_orbit_file(path)

        first = float(data.t0)
        last = float(data.t0 + data.dt * (data.size - 1))

        config = cls(
            orbits_model="esa",
            orbit_file=path,
            t0=first + lead_in,
            **changes,
        )

        if config.t0 + config.T_total > last:
            raise ValueError(
                f"A {config.T_total / 86400.0:.1f} d record does not fit in "
                f"the {(last - first) / 86400.0:.1f} d covered by "
                f"{path}. Shorten N."
            )

        return config

    def build(self):
        """Construct the transform and response for this configuration.

        One call, so the notebook and the reference path cannot end up on
        different grids. The geometry solve runs here; hold the result rather
        than rebuilding.

        Returns
        -------
        wdm : WDM.WDM_transform
            The transform on this tiling.
        response : LISAresponse
            Built on ``wdm`` and the configured orbits, anchored at ``t0``.
        """
        from WDM.code.discrete_wavelet_transform.WDM import WDM_transform

        from ..lisa_response.response import LISAresponse

        wdm = WDM_transform(dt=self.dt, Nf=self.Nf, N=self.N, q=self.q)

        response = LISAresponse(
            wdm,
            self.build_orbits(),
            t0=self.t0,
            max_lag_L=self.max_lag_L,
            num_interp_points=self.num_interp_points,
        )

        return wdm, response

    def summary(self) -> str:
        """Multi-line summary of the grid and its consequences.

        For the notebook's first cell, so the numbers the study turns on are
        stated in the output rather than in prose that can go stale.
        """
        pad = self.padding
        lines = [
            f"cadence          dt = {self.dt:g} s",
            f"bands            Nf = {self.Nf}",
            f"rows             Nt = {self.Nt}",
            f"padded length     N = {self.N} ({self.T_total / 86400.0:.1f} d)",
            f"signal length      = {self.n_signal} "
            f"({self.T_signal / 86400.0:.1f} d)",
            f"tiling           dT = {self.dT:g} s, dF = {self.dF:.6e} Hz",
            f"source        m = {self.m_bin}, f = {self.f_gb * 1e3:.6f} mHz",
            f"guard         {pad['rows_left']} rows per side "
            f"(2x default; L = {self.max_lag_L})",
            f"grid memory      = {self.grid_bytes / 2**20:.1f} MiB per "
            f"(Nt, Nf) grid",
            f"orbits           = {self.orbits_model}",
            f"epoch            t0 = {self.t0:.6g} s",
        ]
        if self.orbits_model == "esa":
            # Basename only. The full path is an absolute one on whoever ran
            # this, and a summary that ends up in a committed notebook output
            # should not carry someone's home directory. ``orbit_file`` still
            # holds it for anything that needs to open the file.
            lines.insert(-1, f"orbit file       = {Path(self.orbit_file).name}")
        return "\n".join(lines)


#: The real run: 485 days at 10 s, one binary at 5.078 mHz in 128 bands, on
#: the ESA trailing orbits. Built at import, so importing this module reads
#: ``EXAMPLES_ORBIT_FILE`` - which is the right failure if it is missing,
#: since the comparison this subpackage exists for cannot run without it.
DEFAULT = ComparisonConfig.for_esa()

#: Thirty days, same cadence and bands and so the same source frequency and
#: conventions. Fix every convention here before running the full record.
SMOKE = ComparisonConfig.for_esa(N=2 ** 18)

#: Closed-form equal-armlength orbits, fifteen days. No file, no epoch to get
#: right, unbounded in time - which makes it the convenient one for tests and
#: for poking at the machinery. Its arms flex 120 times less than the real
#: ones, and its frame is not ``lisatools``', so it is not what the
#: comparison should be run on. See ``ComparisonConfig.build_orbits``.
ANALYTIC = ComparisonConfig(N=2 ** 17)
