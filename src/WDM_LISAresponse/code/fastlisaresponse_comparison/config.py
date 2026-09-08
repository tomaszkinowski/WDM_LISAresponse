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


#: Default epoch, in seconds. Deliberately not zero, for the reason
#: ``tests/conftest.py`` gives: a zero epoch hides floating-point cancellation
#: that only appears at a realistic time.
DEFAULT_T0 = 3.15e7

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
    on the path of every object that exists.

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
        Maximum retained wavelet time-pixel lag :math:`L`. Raised above the
        usual 25 here; see ``Notes``.
    num_interp_points : int
        Interpolation points for the time-delay filter tables. Raised well
        above the package default here; see ``Notes``.
    t0 : float
        Absolute time of the first *padded* sample, in seconds.
    m_bin : int
        WDM frequency bin holding the binary. Bin-centred, so the source
        occupies one row - which is what makes the residual map legible and
        the time-offset measurement in ``conventions`` exact.
    amp, fdot, iota, psi, phi0 : float
        Galactic binary parameters. Free: the study only requires both codes
        receive the same values, and ``utils.wdm_mismatch`` normalises the
        amplitude away entirely.
    theta, phi : float
        Sky position in this package's convention - polar angle from
        :math:`+z`, azimuth in the SSB frame, :math:`\hat{n}` towards the
        source, as ``build_sky_basis`` defines them. ``fastlisaresponse`` uses
        ecliptic longitude and *latitude* instead; ``conventions`` translates.
    orbits_model : str
        One of ``ORBIT_MODELS``. See ``Notes``.
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

    Accumulated over the fourteen shifts of a TDI channel, :math:`L = 25`
    gives :math:`10^{-4}` - level with the local-delay approximation this
    comparison exists to measure - while :math:`L = 51` gives
    :math:`1.5\times10^{-6}`, some seventy times below it. The shift costs
    :math:`O(L)`, so this doubles the runtime, which is the price of keeping
    the operator out of the measurement. Going further keeps paying, but 51
    is already well clear.

    ``num_interp_points`` does essentially nothing, and is 256 only because
    the table costs 204 KiB and the lookup is :math:`O(1)`. At a realistic
    delay the shift error is identical to four figures across 64, 256, 4096
    and 16384 points, at every :math:`L`: lag truncation dominates completely
    and the interpolation is invisible beneath it.

    Beware the :math:`\delta = 0` case, which suggests otherwise. There the
    kernel is exactly diagonal, so lag truncation vanishes and interpolation
    is all that is left - it then converges as :math:`O(h^2)`, four-fold per
    doubling, out to :math:`3\times10^{-11}`. That measurement is real and
    entirely unrepresentative: zero delay is the one point where the dominant
    error is absent. Nothing in this pipeline shifts by zero.

    The guard band is doubled to :math:`P = 2\lceil L/2 \rceil` rows per side,
    which is :math:`\geq L`. The signal passes through two variable time
    shifts - once inside ``one_arm_response`` to reach the retarded phase,
    once in the TDI combination - and :math:`P \geq L` is what makes both
    clean: the first analysed row then reads down to :math:`P - L \geq 0`, so
    no analysed row wraps at all, rather than relying on the wrap landing in
    the opposite pad as ``wdm_padding_info`` alone does.

    Two is the limit, and not because each stage needs its own guard. After
    the second shift the *pad* rows are wrap-contaminated, since their wrapped
    reads now reach a trailing pad that the first shift left non-zero, and a
    third shift's analysed rows would read them. Widening :math:`P` does not
    help: the count is a property of the wrap, not of the pad width. Hence no
    knob - the depth is fixed at two by the pipeline, and the guard that
    serves it is fixed too.

    Nothing needs re-zeroing between the stages. After the first shift the pad
    holds genuine leaked signal at :math:`1.3\times10^{-3}` of peak, and
    clearing it perturbs the analysed region by :math:`3\times10^{-5}` in
    relative L2 - introducing error rather than removing it.

    The padding closes on ``N`` exactly rather than rounding past it, since
    :math:`2\lceil L/2 \rceil` is always even and the parity rounding inside
    ``wdm_padding_info`` never fires. Checked in ``__post_init__``.

    Orbits default to the closed-form provider because it is analytic:
    ``fastlisaresponse`` can be handed the same model exactly, so
    orbit-provider disagreement - otherwise a strong candidate for the noise
    floor of the comparison - is zero rather than something to subtract.

    Equal armlengths do not imply constant light-travel times. The geometry is
    rigid, all three instantaneous separations holding 8.339102380 s to
    :math:`2\times10^{-13}` s, but the solve is retarded, giving
    :math:`\ell_{sr} \approx L(1 + v_s\cdot\hat{r}_{sr}/c)`. At 29.9 km/s that
    term is :math:`\pm5.7\times10^{-4}` s: it reverses under
    :math:`s \leftrightarrow r`, splitting the directions by
    :math:`1.1\times10^{-3}` s, and swings as the triangle revolves, drifting
    each link by :math:`3\times10^{-4}` s over 30 days. So the delays do vary,
    first-generation TDI does leave a laser-noise residual, and the
    cancellation test discriminates here. The ESA orbits vary ~150 times
    faster, which is the reason to move that test to them.
    """

    # Grid
    dt: float = 10.0
    Nf: int = 128
    N: int = 2 ** 22
    q: int = 16

    # Response
    max_lag_L: int = 51
    num_interp_points: int = 256
    t0: float = DEFAULT_T0

    # Source
    m_bin: int = 13
    amp: float = 1.084702251e-22
    fdot: float = 1.0e-16
    iota: float = 1.11820901
    psi: float = 2.3290324
    phi0: float = 4.91128699
    theta: float = 0.59022203
    phi: float = 5.22979888

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
                f"{2 * (self.guard_rows + self.extra_rows)} guard rows of "
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
    def extra_rows(self) -> int:
        r"""Rows added on top of ``guard_rows``, doubling the pad.

        Equal to ``guard_rows``, giving :math:`P = 2\lceil L/2 \rceil \geq L`
        per side. Not a free parameter: the pipeline shifts twice, and
        :math:`P \geq L` is exactly what keeps both stages' analysed rows from
        wrapping. See the class ``Notes``.
        """
        return self.guard_rows

    @property
    def n_signal(self) -> int:
        """Unpadded signal length, in samples.

        Both pads are exact multiples of ``Nf``, so this is a whole number of
        wavelet rows and the signal starts on a row boundary.
        """
        rows = self.guard_rows + self.extra_rows
        return self.N - 2 * rows * self.Nf

    @property
    def padding(self) -> dict:
        """Padding plan from ``utils.wdm_padding_info``.

        Hand ``pad_left``/``pad_right`` to ``utils.apply_zero_padding``, and
        keep the dictionary for ``utils.wdm_cut_info`` afterwards.
        """
        return wdm_padding_info(
            self.n_signal,
            self.Nf,
            max_lag_L=self.max_lag_L,
            extra_rows_left=self.extra_rows,
            extra_rows_right=self.extra_rows,
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

    # -- Diagnostics -------------------------------------------------------

    @property
    def grid_bytes(self) -> int:
        """One :math:`(N_t, N_f)` float64 grid, in bytes.

        The memory unit of the study: six hold the link responses, and the TDI
        combination must accumulate in place rather than stack all its terms.
        """
        return self.Nt * self.Nf * 8

    def time_alignment_tolerance(self, target_mismatch: float = 1e-6) -> float:
        r"""Timing agreement the two codes need for a given mismatch.

        A constant offset is a pure phase error on a monochromatic source, so
        :math:`\mathcal{M} \approx \tfrac{1}{2}(2\pi f \Delta t)^2` and

        .. math::

            |\Delta t| \lesssim \sqrt{2\mathcal{M}} \,/\, 2 \pi f .

        At the default tiling and :math:`10^{-6}` that is 44 ms, or 0.4% of
        one sample - by far the tightest requirement in the study, and why
        ``conventions.measure_time_offset`` measures the offset rather than
        assuming the epochs agree.

        Parameters
        ----------
        target_mismatch : float
            Mismatch to be reached. Optional.

        Returns
        -------
        tolerance : float
            Largest tolerable constant timing offset, in seconds.
        """
        if target_mismatch <= 0.0:
            raise ValueError(
                f"target_mismatch must be positive, got {target_mismatch}."
            )
        return float(np.sqrt(2.0 * target_mismatch)
                     / (2.0 * np.pi * self.f_gb))

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
            f"alignment      dt < {self.time_alignment_tolerance():.4f} s "
            f"for a 1e-6 mismatch",
        ]
        if self.orbits_model == "esa":
            lines.insert(-1, f"orbit file       = {self.orbit_file}")
        return "\n".join(lines)


#: The real run: 485 days at 10 s, one binary at 5.078 mHz in 128 bands.
DEFAULT = ComparisonConfig()

#: Thirty days, same cadence and bands and so the same source frequency and
#: conventions. Fix every convention here before running the full record.
SMOKE = ComparisonConfig(N=2 ** 18)
