"""Reusable second-generation Michelson WDM TDI plans.

This module is LISA-specific: it maps the static Gen-2 Michelson delay-chain
library onto the generic :class:`VariableShiftBatchPlan` implemented by
``WDM_GW_wavelets``.

The prepared plan depends on the WDM grid, recursive TDI delay fields and
shift configuration, but not on waveform coefficients. It can therefore be
built in the A0 stage and applied repeatedly in the waveform/TDI stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from WDM.code.time_delay_filters.plans import VariableShiftBatchPlan

from .one_arm_plan import _shift_options_to_plan_config


Link = tuple[int, int]
DelayChain = tuple[Link, ...]
JobTemplate = tuple[str, Link, DelayChain]


@dataclass(frozen=True, slots=True)
class Gen2MichelsonWDMPlan:
    """Prepared 42-job WDM operator for Gen-2 Michelson ``X, Y, Z``.

    The generic shift plan stores all delay-dependent WDM kernels.  This
    wrapper stores only the LISA-specific mapping from each shifted term to its
    input directed link, output TDI channel and sign.
    """

    wdm: Any
    job_names: tuple[str, ...]
    job_links: tuple[Link, ...]
    output_channels: np.ndarray
    signs: np.ndarray
    base_positive_links: tuple[Link, Link, Link]
    base_negative_links: tuple[Link, Link, Link]
    shift_plan: VariableShiftBatchPlan
    build_seconds: float

    @classmethod
    def build(
        cls,
        *,
        wdm: Any,
        shifts_X: Mapping[DelayChain, np.ndarray],
        shifts_Y: Mapping[DelayChain, np.ndarray],
        shifts_Z: Mapping[DelayChain, np.ndarray],
        job_templates: tuple[
            Iterable[JobTemplate],
            Iterable[JobTemplate],
            Iterable[JobTemplate],
        ],
        L_trunc: int | None,
        variable_shift_kwargs: dict[str, Any] | None = None,
        use_jax: bool | None = None,
    ) -> "Gen2MichelsonWDMPlan":
        """Build the prepared 42-job second-generation Michelson operator.

        Parameters
        ----------
        wdm : object
            WDM transform. Each delay field must have length ``wdm.Nt``.
        shifts_X, shifts_Y, shifts_Z : mapping[DelayChain, ndarray]
            Recursive TDI delay libraries. Each value has shape ``(Nt,)`` and is in
            seconds. Chain keys must match those referenced by ``job_templates``.
        job_templates : tuple[iterable[JobTemplate], iterable[JobTemplate], iterable[JobTemplate]]
            Templates for X, Y and Z. Every channel must provide exactly 14 tuples
            ``(name, source_link, delay_chain)``. The first seven shifted terms receive
            sign ``+1`` and the final seven sign ``-1``.
        L_trunc : int or None
            Maximum retained WDM time lag.
        variable_shift_kwargs : dict[str, Any] or None, optional
            Prepared-plan filter and assembly options.
        use_jax : bool or None, optional
            Compatibility flag; the maintained implementation is JAX-only.

        Returns
        -------
        Gen2MichelsonWDMPlan
            Immutable LISA-specific bookkeeping plus a generic
            ``VariableShiftBatchPlan`` containing all 42 prepared delay operators.

        Raises
        ------
        ValueError
            If a channel does not contain 14 templates or the stacked delay matrix is
            not shaped ``(42, Nt)``.
        KeyError
            If a template references a delay chain absent from its channel library.

        Notes
        -----
        The plan is waveform- and sky-independent. Reuse it for every set of six
        one-arm responses on the same orbit and WDM grid.
        """
        started = perf_counter()

        config, build_kwargs = _shift_options_to_plan_config(
            L_trunc=L_trunc,
            variable_shift_kwargs=variable_shift_kwargs,
            use_jax=use_jax,
        )

        # The prepared Gen-2 TDI operator has fixed, orbit-dependent delays.
        # Runtime-sky Tl/Tp interpolation-table options belong only to the
        # one-arm response path, so forward only arguments accepted by
        # VariableShiftBatchPlan.build().
        tdi_plan_build_kwargs = {
            key: build_kwargs[key]
            for key in (
                "Nker",
                "safety",
                "kernel_kwargs",
            )
            if key in build_kwargs
        }

        channel_names = ("X", "Y", "Z")
        shift_libraries = (shifts_X, shifts_Y, shifts_Z)

        names: list[str] = []
        links: list[Link] = []
        channels: list[int] = []
        signs: list[float] = []
        delays: list[np.ndarray] = []

        for channel_index, (channel_name, shift_library, templates) in enumerate(
            zip(channel_names, shift_libraries, job_templates)
        ):
            templates = tuple(templates)
            if len(templates) != 14:
                raise ValueError(
                    f"Expected 14 shifted terms for {channel_name}, got {len(templates)}."
                )

            for term_index, (name, link, chain) in enumerate(templates):
                if chain not in shift_library:
                    raise KeyError(
                        f"Missing delay chain {chain!r} for channel {channel_name}."
                    )
                names.append(f"{channel_name}:{name}")
                links.append((int(link[0]), int(link[1])))
                channels.append(channel_index)
                signs.append(1.0 if term_index < 7 else -1.0)
                delays.append(np.asarray(shift_library[chain], dtype=float))

        delay_matrix = np.stack(delays, axis=0)
        if delay_matrix.shape != (42, int(wdm.Nt)):
            raise ValueError(
                "Unexpected Gen-2 Michelson delay shape: "
                f"expected {(42, int(wdm.Nt))}, got {delay_matrix.shape}."
            )

        shift_plan = VariableShiftBatchPlan.build(
            wdm,
            delay_matrix,
            config=config,
            Nf=int(wdm.Nf),
            **tdi_plan_build_kwargs,
        )

        # Undelayed terms are the leading +link and -link for each cyclic
        # Michelson channel.
        base_positive_links: tuple[Link, Link, Link] = (
            (2, 1),
            (3, 2),
            (1, 3),
        )
        base_negative_links: tuple[Link, Link, Link] = (
            (3, 1),
            (1, 2),
            (2, 3),
        )

        return cls(
            wdm=wdm,
            job_names=tuple(names),
            job_links=tuple(links),
            output_channels=np.asarray(channels, dtype=np.int8),
            signs=np.asarray(signs, dtype=np.float32),
            base_positive_links=base_positive_links,
            base_negative_links=base_negative_links,
            shift_plan=shift_plan,
            build_seconds=float(perf_counter() - started),
        )

    def _source_layout(
        self,
        link_w: Mapping[Link, np.ndarray],
    ) -> tuple[tuple[Link, ...], np.ndarray, np.ndarray]:
        """Return six source links, per-job source indices and source batch."""

        ordered_links: list[Link] = []
        seen: set[Link] = set()
        for link in (
            *self.job_links,
            *self.base_positive_links,
            *self.base_negative_links,
        ):
            if link not in seen:
                seen.add(link)
                ordered_links.append(link)

        source_links = tuple(ordered_links)
        missing = [link for link in source_links if link not in link_w]
        if missing:
            raise KeyError(f"Missing directed-link WDM arrays: {missing}")

        source_batch = np.stack(
            [np.asarray(link_w[link]) for link in source_links],
            axis=0,
        )
        expected_source_shape = (
            len(source_links),
            int(self.wdm.Nt),
            int(self.wdm.Nf),
        )
        if source_batch.shape != expected_source_shape:
            raise ValueError(
                f"Expected source coefficient batch {expected_source_shape}, "
                f"got {source_batch.shape}."
            )

        source_lookup = {link: index for index, link in enumerate(source_links)}
        source_indices = np.asarray(
            [source_lookup[link] for link in self.job_links],
            dtype=np.int32,
        )
        return source_links, source_indices, source_batch

    def _base_from_sources(
        self,
        source_links: tuple[Link, ...],
        source_batch: np.ndarray,
    ) -> np.ndarray:
        """Construct the three undelayed Michelson base contributions."""

        source_lookup = {link: index for index, link in enumerate(source_links)}
        return np.stack(
            [
                source_batch[source_lookup[pos]]
                - source_batch[source_lookup[neg]]
                for pos, neg in zip(
                    self.base_positive_links,
                    self.base_negative_links,
                )
            ],
            axis=0,
        )

    def input_batch(self, link_w: Mapping[Link, np.ndarray]) -> np.ndarray:
        """Gather the 42 input coefficient arrays from six directed links."""

        missing = sorted(set(self.job_links).difference(link_w))
        if missing:
            raise KeyError(f"Missing directed-link WDM arrays: {missing}")

        coefficients = np.stack(
            [np.asarray(link_w[link]) for link in self.job_links],
            axis=0,
        )
        expected = (42, int(self.wdm.Nt), int(self.wdm.Nf))
        if coefficients.shape != expected:
            raise ValueError(
                f"Expected gathered TDI coefficient batch {expected}, "
                f"got {coefficients.shape}."
            )
        return coefficients

    def _assemble(
        self,
        link_w: Mapping[Link, np.ndarray],
        shifted: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifted = np.asarray(shifted)
        expected = (42, int(self.wdm.Nt), int(self.wdm.Nf))
        if shifted.shape != expected:
            raise ValueError(
                f"Expected shifted TDI batch {expected}, got {shifted.shape}."
            )

        base = np.stack(
            [
                np.asarray(link_w[pos]) - np.asarray(link_w[neg])
                for pos, neg in zip(
                    self.base_positive_links,
                    self.base_negative_links,
                )
            ],
            axis=0,
        )

        out_dtype = np.result_type(base.dtype, shifted.dtype)
        outputs = base.astype(out_dtype, copy=True)

        for channel in range(3):
            mask = self.output_channels == channel
            outputs[channel] += np.tensordot(
                self.signs[mask].astype(out_dtype, copy=False),
                shifted[mask],
                axes=(0, 0),
            )

        return outputs[0], outputs[1], outputs[2]

    def apply_materialized(
        self,
        link_w: Mapping[Link, np.ndarray],
        *,
        return_profile: bool = False,
    ):
        """Reference path that materialises all 42 shifted job outputs."""

        inputs = self.input_batch(link_w)

        if return_profile:
            shifted, profile = self.shift_plan.apply(
                inputs,
                return_profile=True,
            )
            result = self._assemble(link_w, shifted)
            profile = dict(profile)
            profile.update(
                {
                    "tdi_execution_mode": "materialized",
                    "tdi_plan_build_s": float(self.build_seconds),
                    "tdi_kernel_memory_bytes": int(
                        self.shift_plan.kernel_memory_bytes
                    ),
                }
            )
            return result, profile

        shifted = self.shift_plan.apply(inputs)
        return self._assemble(link_w, shifted)

    def apply(
        self,
        link_w: Mapping[Link, np.ndarray],
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply 42 shifts and accumulate directly into ``X, Y, Z``.

        The grouped path transfers only the six distinct directed-link arrays,
        gathers the required job inputs on-device and returns only the three
        final TDI channels. A materialized fallback is retained as a numerical
        reference and for lightweight test doubles.
        """

        grouped_apply = getattr(
            self.shift_plan,
            "apply_indexed_and_accumulate",
            None,
        )
        if grouped_apply is None:
            return self.apply_materialized(
                link_w,
                return_profile=return_profile,
            )

        source_links, source_indices, source_batch = self._source_layout(link_w)
        base = self._base_from_sources(source_links, source_batch)

        grouped = grouped_apply(
            source_batch,
            source_indices,
            self.output_channels,
            self.signs,
            num_outputs=3,
            base=base,
            return_profile=return_profile,
            cache_device_plan=cache_device_plan,
        )

        if return_profile:
            outputs, profile = grouped
            profile = dict(profile)
            profile.update(
                {
                    "tdi_execution_mode": "indexed_accumulate",
                    "tdi_plan_build_s": float(self.build_seconds),
                    "tdi_kernel_memory_bytes": int(
                        self.shift_plan.kernel_memory_bytes
                    ),
                }
            )
            return (outputs[0], outputs[1], outputs[2]), profile

        return grouped[0], grouped[1], grouped[2]

    @property
    def kernel_memory_bytes(self) -> int:
        """Persistent memory occupied by the generic WDM shift kernels."""

        return int(self.shift_plan.kernel_memory_bytes)
