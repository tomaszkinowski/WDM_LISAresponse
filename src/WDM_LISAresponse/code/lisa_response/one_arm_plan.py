"""Reusable six-link WDM one-arm response plans for LISA.

The generic variable-delay operator is implemented by ``WDM_GW_wavelets``.
This module adds only the LISA geometry, antenna projections and transfer
amplitudes required to construct six directed one-arm responses.

A :class:`OneArmWDMPlan` is waveform independent.  It may be reused for any
number of plus/cross waveforms on the same WDM grid, orbit model and sky
position.  The maintained execution paths are:

``materialized``
    Form the six unshifted link arrays on the JAX device and apply the prepared
    shifts in one batch.

``materialized_parallel_links``
    Form the same six-link batch, then evaluate contiguous link groups from
    separate Python workers.  This is a CPU scheduling option; it does not
    change the numerical operator.

The former weighted-source/fused-low-memory route was removed because it was
slower than the materialized implementation on the production CPU benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from lisaconstants import c

from WDM.code.time_delay_filters.config import VariableShiftPlanConfig
from WDM.code.time_delay_filters.plans import (
    VariableShiftBatchPlan,
    VariableShiftKernelContext,
)

from .response_sub import (
    _compute_ltt_iterative,
    antenna_pattern_factors,
    all_sky_time_shift_bounds,
    build_directed_link_geometry,
    build_sky_basis,
    evaluate_sky_link_geometry,
    rhat,
)


Link = tuple[int, int]

DEFAULT_DIRECTED_LINKS: tuple[Link, ...] = (
    (2, 1),
    (3, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (1, 2),
)

ONE_ARM_EXECUTION_MODES: tuple[str, ...] = (
    "materialized",
    "materialized_parallel_links",
)


@jax.jit
def _combine_polarizations_device(
    amplitude_plus,
    amplitude_cross,
    w_t_plus,
    w_t_cross,
):
    """Form the unshifted link-major coefficient batch on-device.

    Real WDM coefficients remain real.  Complex arithmetic is used only when
    either input polarization is complex.
    """

    return (
        amplitude_plus * w_t_plus[None, :, :]
        + amplitude_cross * w_t_cross[None, :, :]
    )


def _validate_jax_request(use_jax: bool | None) -> None:
    """Validate the retained compatibility flag.

    The maintained WDM time-shift implementation is JAX-only.  ``None`` and
    ``True`` are accepted so existing call sites need not change immediately;
    ``False`` is rejected rather than silently selecting a removed backend.
    """

    if use_jax is False:
        raise ValueError(
            "The maintained WDM time-shift implementation requires JAX; "
            "use_jax=False is no longer supported."
        )


def _normalise_one_arm_execution_mode(mode: str | None) -> str:
    """Return one of the two maintained one-arm execution modes."""

    if mode is None:
        return "materialized"

    key = str(mode).strip().lower()
    aliases = {
        "materialised": "materialized",
        "parallel": "materialized_parallel_links",
        "parallel_links": "materialized_parallel_links",
        "materialized_parallel": "materialized_parallel_links",
        "materialised_parallel": "materialized_parallel_links",
        "materialised_parallel_links": "materialized_parallel_links",
    }
    key = aliases.get(key, key)

    if key in {"fused", "fused_low_memory", "low_memory", "low-memory"}:
        raise ValueError(
            "The fused_low_memory one-arm route was removed because it was "
            "slower than the materialized production path."
        )

    if key not in ONE_ARM_EXECUTION_MODES:
        allowed = ", ".join(ONE_ARM_EXECUTION_MODES)
        raise ValueError(
            f"Unknown one-arm execution mode {mode!r}; expected one of: "
            f"{allowed}."
        )
    return key


def _normalise_links(links: Iterable[Link] | None) -> tuple[Link, ...]:
    """Validate and canonicalize a directed-link sequence."""

    if links is None:
        return DEFAULT_DIRECTED_LINKS

    normalised = tuple((int(sender), int(receiver)) for sender, receiver in links)
    if not normalised:
        raise ValueError("At least one directed link is required.")
    return normalised


def _normalise_shift_backend(value: str | None) -> str:
    """Map historical backend names onto production or reference."""

    if value is None:
        return "production"

    key = str(value).strip().lower()
    production_aliases = {
        "production",
        "lagfirst_chunked_lagblock",
        "lagblock",
        "lagfirst_lagblock",
        "auto",
    }
    reference_aliases = {
        "reference",
        "legacy",
        "row",
        "lagfirst_row",
        "vmap",
        "lagfirst_chunked",
        "chunked",
    }
    removed_jobblock_aliases = {
        "lagfirst_chunked_lagblock_jobblock",
        "lagblock_jobblock",
        "jobblock",
        "lagfirst_lagblock_jobblock",
    }

    if key in production_aliases:
        return "production"
    if key in reference_aliases:
        return "reference"
    if key in removed_jobblock_aliases:
        raise ValueError(
            "The job-block WDM assembly backend was removed. Use "
            "assembly_backend='production' and tune row_chunk_size and "
            "lag_block_size instead."
        )
    raise ValueError(
        "assembly_backend must resolve to 'production' or 'reference'."
    )


def _shift_options_to_plan_config(
    *,
    L_trunc: int | None,
    variable_shift_kwargs: dict[str, Any] | None,
    use_jax: bool | None,
) -> tuple[VariableShiftPlanConfig, dict[str, Any]]:
    """Convert public shift options into the cleaned reusable-plan API.

    Historical backend names are accepted as migration aliases.  Options for
    removed experimental implementations are rejected with explicit messages.

    Returns
    -------
    config, build_kwargs
        ``config`` contains repeated-application settings. ``build_kwargs``
        contains one-time plan-construction options: ``Nker``, ``safety`` and
        ``kernel_kwargs``.
    """

    _validate_jax_request(use_jax)
    kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)

    nested_use_jax = kwargs.pop("use_jax", None)
    _validate_jax_request(nested_use_jax)

    removed_options = {
        "assembly_vmap": "the vmap assembly route",
        "job_block_size": "the job-block assembly route",
        "jax_pad_last_chunk": "vmap last-chunk padding",
        "tl_tp_interp_backend": "the separate JAX interpolation backend",
    }
    present_removed = [name for name in removed_options if name in kwargs]
    if present_removed:
        details = ", ".join(
            f"{name} ({removed_options[name]})" for name in present_removed
        )
        raise TypeError(
            "Removed variable-shift options were supplied: " + details
        )

    unsupported_profile = {
        "profile_shift_stages",
        "profile_label",
        "return_shift_profile",
    }.intersection(kwargs)
    if unsupported_profile:
        names = ", ".join(sorted(unsupported_profile))
        raise ValueError(
            "Plan construction does not accept per-call profiling options: "
            f"{names}. Use plan.apply(..., return_profile=True)."
        )

    build_kwargs = {
        "Nker": kwargs.pop("Nker", None),
        "safety": kwargs.pop("safety", 1.02),
        "kernel_kwargs": kwargs.pop("kernel_kwargs", None),
        "tl_tp_table_points": kwargs.pop("tl_tp_table_points", 1024),
        "tl_tp_table_pad": kwargs.pop("tl_tp_table_pad", 0.02),
    }

    # Exact Tl/Tp construction remains the compatibility default.  The fast
    # analytic-parity production assembly is selected explicitly.
    config = VariableShiftPlanConfig(
        lag_truncation=L_trunc,
        tl_tp_mode=kwargs.pop("tl_tp_mode", "exact"),
        tl_tp_interp_points=kwargs.pop("tl_tp_interp_points", 64),
        tl_tp_interp_pad=kwargs.pop("tl_tp_interp_pad", 0.0),
        tl_tp_interp_kind=kwargs.pop("tl_tp_interp_kind", "linear"),
        assembly_backend=_normalise_shift_backend(
            kwargs.pop("assembly_backend", "production")
        ),
        assembly_precision=kwargs.pop("assembly_precision", "complex64"),
        row_chunk_size=kwargs.pop("row_chunk_size", 128),
        lag_block_size=kwargs.pop("lag_block_size", 1),
        batch_chunk=kwargs.pop("batch_chunk", 32),
    )

    # These values are controlled by the WDM object and target-mode plan.
    kwargs.pop("Nf", None)
    delta_mode = kwargs.pop("delta_mode", "target")
    if delta_mode != "target":
        raise ValueError("OneArmWDMPlan supports only delta_mode='target'.")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "Unexpected variable-shift options while building OneArmWDMPlan: "
            f"{unexpected}"
        )

    return config, build_kwargs



@dataclass(frozen=True, slots=True)
class OneArmWDMStaticPlan:
    """Sky-independent one-arm WDM geometry and runtime shift context."""

    wdm: Any
    links: tuple[Link, ...]
    freq_vals: np.ndarray
    int_times_vals: np.ndarray
    geometry: Any
    shift_context: VariableShiftKernelContext
    build_seconds: float

    @classmethod
    def build_from_context(
        cls,
        context: dict[str, Any],
        orbits: Any,
        *,
        links: Iterable[Link] | None = None,
        L_trunc: int | None = None,
        variable_shift_kwargs: dict[str, Any] | None = None,
        use_jax: bool | None = None,
    ) -> "OneArmWDMStaticPlan":
        missing = {
            key for key in ("wdm", "freq_vals", "int_times_vals")
            if key not in context
        }
        if missing:
            raise KeyError(
                "OneArmWDMStaticPlan context is missing required keys: "
                + ", ".join(sorted(missing))
            )
        config, build_kwargs = _shift_options_to_plan_config(
            L_trunc=L_trunc,
            variable_shift_kwargs=variable_shift_kwargs,
            use_jax=use_jax,
        )
        return cls.build(
            wdm=context["wdm"],
            freq_vals=context["freq_vals"],
            int_times_vals=context["int_times_vals"],
            orbits=orbits,
            links=links,
            shift_config=config,
            **build_kwargs,
        )

    @classmethod
    def build(
        cls,
        *,
        wdm: Any,
        freq_vals: np.ndarray,
        int_times_vals: np.ndarray,
        orbits: Any,
        links: Iterable[Link] | None = None,
        shift_config: VariableShiftPlanConfig | None = None,
        Nker: int | None = None,
        safety: float = 1.02,
        kernel_kwargs: dict[str, Any] | None = None,
        tl_tp_table_points: int = 1024,
        tl_tp_table_pad: float = 0.02,
    ) -> "OneArmWDMStaticPlan":
        from time import perf_counter

        started = perf_counter()
        links = _normalise_links(links)
        freq_vals = np.asarray(freq_vals, dtype=float)
        int_times_vals = np.asarray(int_times_vals, dtype=float)
        if freq_vals.shape != (int(wdm.Nf),):
            raise ValueError(
                f"Expected freq_vals shape {(int(wdm.Nf),)}, got {freq_vals.shape}."
            )
        if int_times_vals.shape != (int(wdm.Nt),):
            raise ValueError(
                "Expected int_times_vals shape "
                f"{(int(wdm.Nt),)}, got {int_times_vals.shape}."
            )

        geometry = build_directed_link_geometry(
            int_times_vals,
            orbits,
            links,
        )
        shift_min, shift_max = all_sky_time_shift_bounds(geometry)
        # The WDM operator is called with -time_shift.
        delay_min = -shift_max
        delay_max = -shift_min

        config = (
            VariableShiftPlanConfig.production()
            if shift_config is None
            else shift_config
        )
        if config.tl_tp_mode != "interp":
            # Runtime sky updates require a reusable interpolation table.
            config = VariableShiftPlanConfig(
                lag_truncation=config.lag_truncation,
                tl_tp_mode="interp",
                tl_tp_interp_points=config.tl_tp_interp_points,
                tl_tp_interp_pad=config.tl_tp_interp_pad,
                tl_tp_interp_kind=config.tl_tp_interp_kind,
                assembly_backend=config.assembly_backend,
                assembly_precision=config.assembly_precision,
                row_chunk_size=config.row_chunk_size,
                lag_block_size=config.lag_block_size,
                batch_chunk=config.batch_chunk,
            )

        shift_context = VariableShiftKernelContext.build(
            wdm,
            delay_min=delay_min,
            delay_max=delay_max,
            config=config,
            Nf=int(wdm.Nf),
            Nker=Nker,
            safety=safety,
            kernel_kwargs=kernel_kwargs,
            interpolation_points=int(tl_tp_table_points),
            interpolation_pad=float(tl_tp_table_pad),
            interpolation_kind=config.tl_tp_interp_kind,
        )
        return cls(
            wdm=wdm,
            links=links,
            freq_vals=freq_vals,
            int_times_vals=int_times_vals,
            geometry=geometry,
            shift_context=shift_context,
            build_seconds=float(perf_counter() - started),
        )

    def sky_arrays(self, theta: float, phi: float):
        """Return runtime amplitudes and one-arm delay fields."""

        sky = evaluate_sky_link_geometry(
            self.geometry,
            theta,
            phi,
        )
        common = (
            0.5
            * c
            * self.geometry.light_time[:, :, None]
            * np.sinc(
                2.0
                * self.freq_vals[None, None, :]
                * sky.tau[:, :, None]
            )
        )
        amplitude_plus = common * sky.projection_plus[:, :, None]
        amplitude_cross = common * sky.projection_cross[:, :, None]
        if self.shift_context.resolved_assembly_precision == "complex64":
            real_dtype = np.float32
        else:
            real_dtype = np.float64
        return (
            np.asarray(amplitude_plus, dtype=real_dtype),
            np.asarray(amplitude_cross, dtype=real_dtype),
            np.asarray(sky.time_shift, dtype=real_dtype),
        )

    def _validate_waveform_pair(self, w_t_plus, w_t_cross):
        plus = np.asarray(w_t_plus)
        cross = np.asarray(w_t_cross)
        expected = (int(self.wdm.Nt), int(self.wdm.Nf))
        if plus.shape != expected or cross.shape != expected:
            raise ValueError(
                f"Expected plus/cross WDM coefficients {expected}; "
                f"got plus={plus.shape}, cross={cross.shape}."
            )
        return plus, cross

    def apply_batch(
        self,
        w_t_plus,
        w_t_cross,
        theta,
        phi,
        *,
        return_profile: bool = False,
    ):
        """Evaluate one sky location and return a six-link NumPy batch."""

        plus, cross = self._validate_waveform_pair(w_t_plus, w_t_cross)
        amp_plus, amp_cross, time_shift = self.sky_arrays(theta, phi)
        unshifted = (
            amp_plus * plus[None, :, :]
            + amp_cross * cross[None, :, :]
        )
        return self.shift_context.apply(
            unshifted,
            -time_shift,
            return_profile=return_profile,
        )

    def apply_batch_device(
        self,
        w_t_plus,
        w_t_cross,
        theta,
        phi,
        *,
        return_profile: bool = False,
        cache_device_context: bool = True,
    ):
        """Evaluate one sky location while keeping response arrays on-device."""

        plus, cross = self._validate_waveform_pair(w_t_plus, w_t_cross)
        amp_plus, amp_cross, time_shift = self.sky_arrays(theta, phi)
        plus_device, cross_device = (
            jnp.asarray(plus),
            jnp.asarray(cross),
        )
        real_dtype = (
            jnp.float32
            if self.shift_context.resolved_assembly_precision == "complex64"
            else jnp.float64
        )
        unshifted = _combine_polarizations_device(
            jnp.asarray(amp_plus, dtype=real_dtype),
            jnp.asarray(amp_cross, dtype=real_dtype),
            jnp.asarray(plus_device, dtype=real_dtype),
            jnp.asarray(cross_device, dtype=real_dtype),
        )
        return self.shift_context.apply_device(
            unshifted,
            -time_shift,
            return_profile=return_profile,
            cache_device_context=cache_device_context,
        )

    def apply(
        self,
        w_t_plus,
        w_t_cross,
        theta,
        phi,
    ) -> dict[Link, np.ndarray]:
        shifted = self.apply_batch(w_t_plus, w_t_cross, theta, phi)
        return {
            link: shifted[index]
            for index, link in enumerate(self.links)
        }

    def apply_device(
        self,
        w_t_plus,
        w_t_cross,
        theta,
        phi,
        *,
        return_profile: bool = False,
        cache_device_context: bool = True,
    ):
        result = self.apply_batch_device(
            w_t_plus,
            w_t_cross,
            theta,
            phi,
            return_profile=return_profile,
            cache_device_context=cache_device_context,
        )
        if return_profile:
            shifted, profile = result
            return (
                {
                    link: shifted[index]
                    for index, link in enumerate(self.links)
                },
                profile,
            )
        return {
            link: result[index]
            for index, link in enumerate(self.links)
        }

    def clear_device_cache(self) -> None:
        self.shift_context.clear_device_cache()

    @property
    def kernel_memory_bytes(self) -> int:
        return int(self.shift_context.kernel_memory_bytes)


@dataclass(frozen=True, slots=True)
class OneArmWDMPlan:
    """Prepared LISA one-arm amplitudes and variable-delay operators.

    ``amplitude_plus`` and ``amplitude_cross`` contain waveform-independent
    transfer factors with shape ``(num_links, Nt, Nf)``.  ``shift_plan`` stores
    the corresponding generic WDM delay operators.
    """

    wdm: Any
    links: tuple[Link, ...]
    theta: float
    phi: float

    amplitude_plus: np.ndarray
    amplitude_cross: np.ndarray
    time_shift: np.ndarray
    shift_plan: VariableShiftBatchPlan

    _device_cache: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def build_from_context(
        cls,
        context: dict[str, Any],
        orbits: Any,
        theta: float,
        phi: float,
        *,
        links: Iterable[Link] | None = None,
        L_trunc: int | None = None,
        variable_shift_kwargs: dict[str, Any] | None = None,
        use_jax: bool | None = None,
    ) -> "OneArmWDMPlan":
        """Build a waveform-independent plan from WDM grid metadata.

        ``context`` must contain ``wdm``, ``freq_vals`` and
        ``int_times_vals``. Waveform projections are deliberately not read.
        """

        missing = {
            key
            for key in ("wdm", "freq_vals", "int_times_vals")
            if key not in context
        }
        if missing:
            raise KeyError(
                "OneArmWDMPlan context is missing required keys: "
                + ", ".join(sorted(missing))
            )

        config, build_kwargs = _shift_options_to_plan_config(
            L_trunc=L_trunc,
            variable_shift_kwargs=variable_shift_kwargs,
            use_jax=use_jax,
        )

        return cls.build(
            wdm=context["wdm"],
            freq_vals=context["freq_vals"],
            int_times_vals=context["int_times_vals"],
            orbits=orbits,
            theta=theta,
            phi=phi,
            links=links,
            shift_config=config,
            Nker=build_kwargs["Nker"],
            safety=build_kwargs["safety"],
            kernel_kwargs=build_kwargs["kernel_kwargs"],
        )

    @classmethod
    def build(
        cls,
        *,
        wdm: Any,
        freq_vals: np.ndarray,
        int_times_vals: np.ndarray,
        orbits: Any,
        theta: float,
        phi: float,
        links: Iterable[Link] | None = None,
        shift_config: VariableShiftPlanConfig | None = None,
        Nker: int | None = None,
        safety: float = 1.02,
        kernel_kwargs: dict[str, Any] | None = None,
    ) -> "OneArmWDMPlan":
        """Build link geometry, amplitudes and prepared shift operators."""

        links = _normalise_links(links)
        freq_vals = np.asarray(freq_vals, dtype=float)
        int_times_vals = np.asarray(int_times_vals, dtype=float)

        expected_freq_shape = (int(wdm.Nf),)
        expected_time_shape = (int(wdm.Nt),)
        if freq_vals.shape != expected_freq_shape:
            raise ValueError(
                f"Expected freq_vals shape {expected_freq_shape}, "
                f"got {freq_vals.shape}."
            )
        if int_times_vals.shape != expected_time_shape:
            raise ValueError(
                f"Expected int_times_vals shape {expected_time_shape}, "
                f"got {int_times_vals.shape}."
            )

        sky_basis = build_sky_basis(theta, phi)
        Omega = sky_basis.Omega
        n = sky_basis.n

        amplitude_plus_parts: list[np.ndarray] = []
        amplitude_cross_parts: list[np.ndarray] = []
        time_shift_parts: list[np.ndarray] = []

        for sender, receiver in links:
            light_time, receiver_position = _compute_ltt_iterative(
                int_times_vals,
                sender,
                receiver,
                orbits,
                return_receiver_position=True,
            )
            direction = rhat(
                int_times_vals,
                sender,
                receiver,
                orbits,
                tt=light_time,
                x_r_rec=receiver_position,
            )

            omega_dot_direction = direction @ Omega
            tau = 0.5 * light_time * (1.0 - omega_dot_direction)
            projection_plus, projection_cross = antenna_pattern_factors(
                direction,
                sky_basis,
            )

            common = (
                0.5
                * c
                * light_time[:, None]
                * np.sinc(2.0 * freq_vals[None, :] * tau[:, None])
            )
            amplitude_plus_parts.append(common * projection_plus[:, None])
            amplitude_cross_parts.append(common * projection_cross[:, None])
            time_shift_parts.append(receiver_position @ (n / c) - tau)

        amplitude_plus = np.stack(amplitude_plus_parts, axis=0)
        amplitude_cross = np.stack(amplitude_cross_parts, axis=0)
        time_shift = np.stack(time_shift_parts, axis=0)

        shift_config = (
            VariableShiftPlanConfig()
            if shift_config is None
            else shift_config
        )
        shift_plan = VariableShiftBatchPlan.build(
            wdm,
            -time_shift,
            config=shift_config,
            Nf=int(wdm.Nf),
            Nker=Nker,
            safety=safety,
            kernel_kwargs=kernel_kwargs,
        )

        real_dtype = (
            np.float32
            if shift_plan.resolved_assembly_precision == "complex64"
            else np.float64
        )
        return cls(
            wdm=wdm,
            links=links,
            theta=float(theta),
            phi=float(phi),
            amplitude_plus=np.asarray(amplitude_plus, dtype=real_dtype),
            amplitude_cross=np.asarray(amplitude_cross, dtype=real_dtype),
            time_shift=np.asarray(time_shift, dtype=real_dtype),
            shift_plan=shift_plan,
        )

    def _validate_wdm_coefficients(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate host plus/cross coefficient grids without casting them."""

        w_t_plus = np.asarray(w_t_plus)
        w_t_cross = np.asarray(w_t_cross)
        expected = (int(self.wdm.Nt), int(self.wdm.Nf))
        if w_t_plus.shape != expected or w_t_cross.shape != expected:
            raise ValueError(
                "Expected plus/cross WDM coefficients with shape "
                f"{expected}; got plus={w_t_plus.shape}, "
                f"cross={w_t_cross.shape}."
            )
        return w_t_plus, w_t_cross

    def unshifted_coefficients(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> np.ndarray:
        """Construct the host six-link coefficient batch."""

        w_t_plus, w_t_cross = self._validate_wdm_coefficients(
            w_t_plus,
            w_t_cross,
        )
        return (
            self.amplitude_plus * w_t_plus[None, :, :]
            + self.amplitude_cross * w_t_cross[None, :, :]
        )

    def apply_batch(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
        *,
        return_profile: bool = False,
    ):
        """Apply the plan and return a link-major NumPy batch."""

        unshifted = self.unshifted_coefficients(w_t_plus, w_t_cross)
        return self.shift_plan.apply(unshifted, return_profile=return_profile)

    def _device_real_dtype(self):
        """Return the real JAX dtype configured by the shift plan."""

        if self.shift_plan.resolved_assembly_precision == "complex64":
            return jnp.float32
        return jnp.float64

    def _waveform_pair_device(self, w_t_plus, w_t_cross):
        """Validate and cast a waveform pair while preserving real inputs."""

        plus = jnp.asarray(w_t_plus)
        cross = jnp.asarray(w_t_cross)
        expected = (int(self.wdm.Nt), int(self.wdm.Nf))
        if tuple(plus.shape) != expected:
            raise ValueError(
                f"Expected plus WDM coefficients with shape {expected}, "
                f"got {tuple(plus.shape)}."
            )
        if tuple(cross.shape) != expected:
            raise ValueError(
                f"Expected cross WDM coefficients with shape {expected}, "
                f"got {tuple(cross.shape)}."
            )

        requires_complex = (
            jnp.issubdtype(plus.dtype, jnp.complexfloating)
            or jnp.issubdtype(cross.dtype, jnp.complexfloating)
        )
        if self.shift_plan.resolved_assembly_precision == "complex64":
            coefficient_dtype = jnp.complex64 if requires_complex else jnp.float32
        else:
            coefficient_dtype = jnp.complex128 if requires_complex else jnp.float64

        return (
            jnp.asarray(plus, dtype=coefficient_dtype),
            jnp.asarray(cross, dtype=coefficient_dtype),
        )

    def _device_amplitudes(self):
        """Return cached real amplitude arrays on the active JAX device."""

        if "amplitude_plus" not in self._device_cache:
            real_dtype = self._device_real_dtype()
            self._device_cache["amplitude_plus"] = jnp.asarray(
                self.amplitude_plus,
                dtype=real_dtype,
            )
            self._device_cache["amplitude_cross"] = jnp.asarray(
                self.amplitude_cross,
                dtype=real_dtype,
            )
        return (
            self._device_cache["amplitude_plus"],
            self._device_cache["amplitude_cross"],
        )

    def _materialized_device_batch(self, w_t_plus, w_t_cross):
        """Construct the six unshifted link arrays on-device."""

        plus, cross = self._waveform_pair_device(w_t_plus, w_t_cross)
        amplitude_plus, amplitude_cross = self._device_amplitudes()
        return _combine_polarizations_device(
            amplitude_plus,
            amplitude_cross,
            plus,
            cross,
        )

    def apply_batch_device_materialized(
        self,
        w_t_plus,
        w_t_cross,
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply all links through one device-resident production batch."""

        unshifted = self._materialized_device_batch(w_t_plus, w_t_cross)
        return self.shift_plan.apply_device(
            unshifted,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

    def apply_batch_device_materialized_parallel_groups(
        self,
        w_t_plus,
        w_t_cross,
        *,
        group_size: int = 1,
        max_workers: int | None = None,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply contiguous link groups concurrently on the CPU backend."""

        unshifted = self._materialized_device_batch(w_t_plus, w_t_cross)
        return self.shift_plan.apply_device_parallel_groups(
            unshifted,
            group_size=group_size,
            max_workers=max_workers,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

    def apply_batch_device(
        self,
        w_t_plus,
        w_t_cross,
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Compatibility alias for the maintained materialized device path."""

        return self.apply_batch_device_materialized(
            w_t_plus,
            w_t_cross,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

    def apply(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> dict[Link, np.ndarray]:
        """Apply the host path and return the directed-link mapping."""

        shifted = self.apply_batch(w_t_plus, w_t_cross)
        return {
            link: shifted[index]
            for index, link in enumerate(self.links)
        }

    def apply_device(
        self,
        w_t_plus,
        w_t_cross,
        *,
        execution_mode: str = "materialized",
        parallel_links_per_group: int = 1,
        parallel_max_workers: int | None = None,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply the plan and return device-resident arrays by directed link.

        ``materialized_parallel_links`` is an optional CPU scheduling mode.
        The default ``materialized`` route is deterministic and portable.
        """

        mode = _normalise_one_arm_execution_mode(execution_mode)
        if mode == "materialized":
            shifted_result = self.apply_batch_device_materialized(
                w_t_plus,
                w_t_cross,
                return_profile=return_profile,
                cache_device_plan=cache_device_plan,
            )
        else:
            shifted_result = self.apply_batch_device_materialized_parallel_groups(
                w_t_plus,
                w_t_cross,
                group_size=parallel_links_per_group,
                max_workers=parallel_max_workers,
                return_profile=return_profile,
                cache_device_plan=cache_device_plan,
            )

        if return_profile:
            shifted_device, profile = shifted_result
            responses = {
                link: shifted_device[index]
                for index, link in enumerate(self.links)
            }
            profile = dict(profile)
            profile["one_arm_execution_mode"] = mode
            if mode == "materialized_parallel_links":
                profile["one_arm_parallel_links_per_group"] = int(
                    parallel_links_per_group
                )
                profile["one_arm_parallel_max_workers"] = (
                    None
                    if parallel_max_workers is None
                    else int(parallel_max_workers)
                )
            return responses, profile

        shifted_device = shifted_result
        return {
            link: shifted_device[index]
            for index, link in enumerate(self.links)
        }

    def _validate_context_grid(self, context: dict[str, Any]) -> None:
        """Ensure a waveform context uses the plan's WDM grid."""

        candidate = context.get("wdm")
        if candidate is self.wdm:
            return
        if candidate is None or (
            int(candidate.Nt) != int(self.wdm.Nt)
            or int(candidate.Nf) != int(self.wdm.Nf)
            or not np.isclose(float(candidate.dF), float(self.wdm.dF))
        ):
            raise ValueError(
                "The context WDM grid does not match this OneArmWDMPlan."
            )

    def apply_context(self, context: dict[str, Any]) -> dict[Link, np.ndarray]:
        """Apply the host path to waveform projections stored in a context."""

        self._validate_context_grid(context)
        return self.apply(context["w_t_plus"], context["w_t_cross"])

    def apply_context_device(
        self,
        context: dict[str, Any],
        *,
        execution_mode: str = "materialized",
        parallel_links_per_group: int = 1,
        parallel_max_workers: int | None = None,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply a waveform context while keeping link arrays on-device."""

        self._validate_context_grid(context)
        return self.apply_device(
            context["w_t_plus"],
            context["w_t_cross"],
            execution_mode=execution_mode,
            parallel_links_per_group=parallel_links_per_group,
            parallel_max_workers=parallel_max_workers,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

    def clear_device_cache(self) -> None:
        """Release cached amplitude and shift-plan arrays from the device."""

        self._device_cache.clear()
        self.shift_plan.clear_device_cache()
