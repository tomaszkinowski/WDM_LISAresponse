"""Reusable six-link WDM one-arm response plans for LISA.

The generic variable-shift operator lives in ``WDM_GW_wavelets``.  This module
contains only the LISA-specific quantities needed to construct its delay fields
and unshifted one-arm response coefficients.

A :class:`OneArmWDMPlan` depends on the WDM grid, LISA orbit, sky position and
shift configuration, but not on the waveform coefficients.  It can therefore
be reused for any number of waveforms sampled on the same WDM grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from lisaconstants import c

from WDM.code.time_delay_filters.config import VariableShiftPlanConfig
from WDM.code.time_delay_filters.plans import VariableShiftBatchPlan

from .response_sub import (
    _compute_ltt_iterative,
    antenna_pattern_factors,
    build_sky_basis,
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


def _normalise_links(links: Iterable[Link] | None) -> tuple[Link, ...]:
    if links is None:
        return DEFAULT_DIRECTED_LINKS

    normalised = tuple((int(sender), int(receiver)) for sender, receiver in links)
    if not normalised:
        raise ValueError("At least one directed link is required.")
    return normalised


def _legacy_shift_options_to_plan_config(
    *,
    L_trunc: int | None,
    variable_shift_kwargs: dict[str, Any] | None,
    use_jax: bool | None,
) -> tuple[VariableShiftPlanConfig, dict[str, Any]]:
    """Translate the existing shifter kwargs into a reusable plan config.

    This adapter keeps the first integration backwards compatible with the
    current ``variable_shift_kwargs`` interface.  It can be removed once the
    LISA package exposes ``VariableShiftPlanConfig`` directly.

    Returns
    -------
    config, build_kwargs
        ``build_kwargs`` contains the plan-construction arguments that are not
        fields of :class:`VariableShiftPlanConfig` (``Nker``, ``safety`` and
        ``kernel_kwargs``).
    """

    kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)

    # Profiling belongs to plan application rather than plan construction.  The
    # current one-arm wrapper does not expose it, so reject it explicitly rather
    # than silently changing behaviour.
    unsupported = {
        "profile_shift_stages",
        "profile_label",
        "return_shift_profile",
    }.intersection(kwargs)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            "The reusable one-arm plan does not accept profiling options through "
            f"variable_shift_kwargs: {names}. Profile plan.build_seconds and "
            "plan.apply(..., return_profile=True) instead."
        )

    build_kwargs = {
        "Nker": kwargs.pop("Nker", None),
        "safety": kwargs.pop("safety", 1.02),
        "kernel_kwargs": kwargs.pop("kernel_kwargs", None),
    }

    # Preserve the defaults selected by the current one-arm code path.
    assembly_backend = kwargs.pop("assembly_backend", None)
    assembly_vmap = kwargs.pop("assembly_vmap", None)
    if assembly_backend is None and assembly_vmap is None:
        assembly_backend = "lagfirst_chunked"

    config = VariableShiftPlanConfig(
        lag_truncation=L_trunc,
        tl_tp_mode=kwargs.pop("tl_tp_mode", "exact"),
        tl_tp_interp_points=kwargs.pop("tl_tp_interp_points", 64),
        tl_tp_interp_pad=kwargs.pop("tl_tp_interp_pad", 0.0),
        tl_tp_interp_kind=kwargs.pop("tl_tp_interp_kind", "linear"),
        tl_tp_interp_backend=kwargs.pop("tl_tp_interp_backend", "numpy"),
        assembly_backend=assembly_backend,
        assembly_precision=kwargs.pop("assembly_precision", "complex64"),
        row_chunk_size=kwargs.pop("row_chunk_size", 128),
        lag_block_size=kwargs.pop("lag_block_size", 1),
        job_block_size=kwargs.pop("job_block_size", 1),
        assembly_vmap=assembly_vmap,
        batch_chunk=kwargs.pop("batch_chunk", 32),
        jax_pad_last_chunk=kwargs.pop("jax_pad_last_chunk", False),
        use_jax=kwargs.pop("use_jax", use_jax),
    )

    # These parameters are controlled by the WDM object/context rather than by
    # the plan.  Accepting matching legacy values is useful during migration.
    kwargs.pop("Nf", None)
    kwargs.pop("delta_mode", None)

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "Unexpected variable-shift options while building OneArmWDMPlan: "
            f"{unexpected}"
        )

    return config, build_kwargs


@dataclass(frozen=True, slots=True)
class OneArmWDMPlan:
    """Prepared LISA one-arm response and six variable-delay operators.

    ``amplitude_plus`` and ``amplitude_cross`` contain the waveform-independent
    transfer factors for each directed link.  ``shift_plan`` stores the
    corresponding delay-dependent WDM operators.
    """

    wdm: Any
    links: tuple[Link, ...]
    theta: float
    phi: float

    amplitude_plus: np.ndarray
    amplitude_cross: np.ndarray
    time_shift: np.ndarray
    shift_plan: VariableShiftBatchPlan

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
        """Build a waveform-independent one-arm plan from WDM grid metadata.

        The context must contain ``wdm``, ``freq_vals`` and ``int_times_vals``.
        The waveform projections ``w_t_plus`` and ``w_t_cross`` are deliberately
        not read during construction.
        """

        missing = {
            key for key in ("wdm", "freq_vals", "int_times_vals") if key not in context
        }
        if missing:
            raise KeyError(
                "OneArmWDMPlan context is missing required keys: "
                + ", ".join(sorted(missing))
            )

        config, build_kwargs = _legacy_shift_options_to_plan_config(
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
        theta: float,
        phi: float,
        links: Iterable[Link] | None = None,
        shift_config: VariableShiftPlanConfig | None = None,
        Nker: int | None = None,
        safety: float = 1.02,
        kernel_kwargs: dict[str, Any] | None = None,
    ) -> "OneArmWDMPlan":
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

            sinc_matrix = np.sinc(
                2.0 * freq_vals[None, :] * tau[:, None]
            )
            common = 0.5 * c * light_time[:, None] * sinc_matrix

            amplitude_plus_parts.append(common * projection_plus[:, None])
            amplitude_cross_parts.append(common * projection_cross[:, None])
            time_shift_parts.append(receiver_position @ (n / c) - tau)

        amplitude_plus = np.stack(amplitude_plus_parts, axis=0)
        amplitude_cross = np.stack(amplitude_cross_parts, axis=0)
        time_shift = np.stack(time_shift_parts, axis=0)

        shift_config = (
            VariableShiftPlanConfig() if shift_config is None else shift_config
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

        return cls(
            wdm=wdm,
            links=links,
            theta=float(theta),
            phi=float(phi),
            amplitude_plus=amplitude_plus,
            amplitude_cross=amplitude_cross,
            time_shift=time_shift,
            shift_plan=shift_plan,
        )

    def _validate_wdm_coefficients(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        w_t_plus = np.asarray(w_t_plus)
        w_t_cross = np.asarray(w_t_cross)
        expected = (int(self.wdm.Nt), int(self.wdm.Nf))

        if w_t_plus.shape != expected or w_t_cross.shape != expected:
            raise ValueError(
                "Expected plus/cross WDM coefficients with shape "
                f"{expected}; got plus={w_t_plus.shape}, cross={w_t_cross.shape}."
            )
        return w_t_plus, w_t_cross

    def unshifted_coefficients(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> np.ndarray:
        """Build the unshifted six-link coefficient batch."""

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
        """Apply the prepared one-arm operators and return link-major arrays."""

        unshifted = self.unshifted_coefficients(w_t_plus, w_t_cross)
        return self.shift_plan.apply(unshifted, return_profile=return_profile)

    def apply_batch_device(
        self,
        w_t_plus,
        w_t_cross,
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply all one-arm operators and return a link-major JAX array.

        The current first implementation deliberately retains the validated
        NumPy construction of the unshifted link coefficients. Only the shift
        output remains on-device. A later optimisation can fuse polarization
        modulation with the shift kernel.
        """

        unshifted = self.unshifted_coefficients(
            w_t_plus,
            w_t_cross,
        )

        return self.shift_plan.apply_device(
            unshifted,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

    def apply(
        self,
        w_t_plus: np.ndarray,
        w_t_cross: np.ndarray,
    ) -> dict[Link, np.ndarray]:
        """Apply the plan and return the existing directed-link mapping."""

        shifted = self.apply_batch(w_t_plus, w_t_cross)
        return {
            link: shifted[index]
            for index, link in enumerate(self.links)
        }
    
    def apply_device(
        self,
        w_t_plus,
        w_t_cross,
    ):
        """Apply the plan and return a JAX-array dictionary by directed link."""

        shifted_device = self.apply_batch_device(
            w_t_plus,
            w_t_cross,
        )

        return {
            link: shifted_device[index]
            for index, link in enumerate(self.links)
        }

    def apply_context(self, context: dict[str, Any]) -> dict[Link, np.ndarray]:
        """Apply the plan to waveform projections stored in a response context."""

        if context.get("wdm") is not self.wdm:
            candidate = context.get("wdm")
            if candidate is None or (
                int(candidate.Nt) != int(self.wdm.Nt)
                or int(candidate.Nf) != int(self.wdm.Nf)
                or not np.isclose(float(candidate.dF), float(self.wdm.dF))
            ):
                raise ValueError(
                    "The context WDM grid does not match this OneArmWDMPlan."
                )

        return self.apply(context["w_t_plus"], context["w_t_cross"])
    
    def apply_context_device(
        self,
        context: dict[str, Any],
    ):
        """Apply the plan to a context and retain link arrays on-device."""

        if context.get("wdm") is not self.wdm:
            candidate = context.get("wdm")
            if candidate is None or (
                int(candidate.Nt) != int(self.wdm.Nt)
                or int(candidate.Nf) != int(self.wdm.Nf)
                or not np.isclose(
                    float(candidate.dF),
                    float(self.wdm.dF),
                )
            ):
                raise ValueError(
                    "The context WDM grid does not match this "
                    "OneArmWDMPlan."
                )

        return self.apply_device(
            context["w_t_plus"],
            context["w_t_cross"],
        )