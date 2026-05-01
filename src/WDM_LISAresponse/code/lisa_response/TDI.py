#code to generate TDI signals in TD and WDM domains 
import time
import numpy as np
from WDM_LISAresponse.code.lisa_response.response import (
    one_arm_response_td,
    precompute_one_arm_wdm_dT_links,
    precompute_one_arm_wdm_dT_links_from_context,
    _build_wdm_dT_context_from_wdm,
    _split_variable_shift_kwargs,
)
from WDM.code.time_delay_filters.time_shift_fast import (
    wdm_time_shift_fixed_batch,
    wdm_time_shift_variable_batch,
)
from WDM_LISAresponse.code.lisa_response.TDI_sub import fft_time_shift, interp_time_shift, precompute_gen2_michelson_shifts, precompute_gen2_sagnac_shifts


def _resolve_one_arm_cache(one_arm_cache):
    """Return an opt-in cache for one-arm WDM responses.

    By default (``None``), no persistent cross-call cache is used.
    Pass a mutable dict to reuse one-arm responses across repeated calls.
    """
    return one_arm_cache


def _resolve_variable_shift_batch_kwargs(
    variable_shift_kwargs,
    default_batch_chunk=32,
):
    """Build kwargs for ``wdm_time_shift_variable_batch`` with stable defaults.

    ``batch_chunk`` can be overridden by the caller through
    ``variable_shift_kwargs``. When omitted, the previous Gen-2 behavior
    (chunk size 32) is preserved. The default assembly backend uses the
    lag-first chunked kernels unless overridden.

    Parameters
    ----------
    variable_shift_kwargs : dict or None
        User-provided variable shift options, including backend selection.
    default_batch_chunk : int or None
        Default ``batch_chunk`` when not explicitly provided.
    """
    shift_kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)
    # ``use_jax`` is passed explicitly at call sites; drop legacy duplicate key.
    shift_kwargs.pop("use_jax", None)
    if "assembly_backend" not in shift_kwargs and "assembly_vmap" not in shift_kwargs:
        shift_kwargs["assembly_backend"] = "lagfirst_chunked"
        shift_kwargs["assembly_precision"] = "complex64"
        shift_kwargs["row_chunk_size"] = 128
        shift_kwargs["tl_tp_mode"] = "exact"
    batch_chunk = shift_kwargs.pop("batch_chunk", default_batch_chunk)
    if batch_chunk is not None:
        shift_kwargs["batch_chunk"] = batch_chunk
    return shift_kwargs


def _finalize_variable_shift_batch_kwargs(shift_kwargs, job_count):
    """Finalize batch kwargs once the total variable-shift job count is known.
    """
    out = {} if shift_kwargs is None else dict(shift_kwargs)
    n_jobs = max(1, int(job_count))

    batch_chunk = out.get("batch_chunk", None)

    if batch_chunk is None:
        out.pop("batch_chunk", None)
    else:
        out["batch_chunk"] = int(min(max(1, int(batch_chunk)), n_jobs))

    return out


def _rot_sc(i, k):
    """Rotate a spacecraft index by ``k`` steps in {1,2,3} cyclic order."""
    return ((i - 1 + k) % 3) + 1


def _rot_link(link, k):
    """Rotate a directed link ``(s, r)`` by ``k`` spacecraft steps."""
    s, r = link
    return (_rot_sc(s, k), _rot_sc(r, k))


def _rot_chain(chain, k):
    """Rotate each directed link in a delay chain by ``k`` steps."""
    return tuple(_rot_link(lk, k) for lk in chain)


def _build_gen2_sagnac_job_templates():
    """Build static Gen-2 Sagnac shift job templates for alpha/beta/gamma."""
    # Base chains for alpha.
    cA_21 = ((2, 1),)
    cA_21_32 = ((2, 1), (3, 2))
    cA_21_32_13 = ((2, 1), (3, 2), (1, 3))
    cA_21_32_13_31 = ((2, 1), (3, 2), (1, 3), (3, 1))
    cA_21_32_13_31_23 = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3))

    cA_31 = ((3, 1),)
    cA_31_23 = ((3, 1), (2, 3))
    cA_31_23_12 = ((3, 1), (2, 3), (1, 2))
    cA_31_23_12_21 = ((3, 1), (2, 3), (1, 2), (2, 1))
    cA_31_23_12_21_32 = ((3, 1), (2, 3), (1, 2), (2, 1), (3, 2))

    cB_21 = _rot_chain(cA_21, 1)
    cB_21_32 = _rot_chain(cA_21_32, 1)
    cB_21_32_13 = _rot_chain(cA_21_32_13, 1)
    cB_21_32_13_31 = _rot_chain(cA_21_32_13_31, 1)
    cB_21_32_13_31_23 = _rot_chain(cA_21_32_13_31_23, 1)

    cB_31 = _rot_chain(cA_31, 1)
    cB_31_23 = _rot_chain(cA_31_23, 1)
    cB_31_23_12 = _rot_chain(cA_31_23_12, 1)
    cB_31_23_12_21 = _rot_chain(cA_31_23_12_21, 1)
    cB_31_23_12_21_32 = _rot_chain(cA_31_23_12_21_32, 1)

    cG_21 = _rot_chain(cA_21, 2)
    cG_21_32 = _rot_chain(cA_21_32, 2)
    cG_21_32_13 = _rot_chain(cA_21_32_13, 2)
    cG_21_32_13_31 = _rot_chain(cA_21_32_13_31, 2)
    cG_21_32_13_31_23 = _rot_chain(cA_21_32_13_31_23, 2)

    cG_31 = _rot_chain(cA_31, 2)
    cG_31_23 = _rot_chain(cA_31_23, 2)
    cG_31_23_12 = _rot_chain(cA_31_23_12, 2)
    cG_31_23_12_21 = _rot_chain(cA_31_23_12_21, 2)
    cG_31_23_12_21_32 = _rot_chain(cA_31_23_12_21_32, 2)

    alpha_jobs = [
        ("w32_c21", (3, 2), cA_21),
        ("w13_c21_32", (1, 3), cA_21_32),
        ("w31_c21_32_13", (3, 1), cA_21_32_13),
        ("w23_c21_32_13_31", (2, 3), cA_21_32_13_31),
        ("w12_c21_32_13_31_23", (1, 2), cA_21_32_13_31_23),
        ("w23_c31", (2, 3), cA_31),
        ("w12_c31_23", (1, 2), cA_31_23),
        ("w21_c31_23_12", (2, 1), cA_31_23_12),
        ("w32_c31_23_12_21", (3, 2), cA_31_23_12_21),
        ("w13_c31_23_12_21_32", (1, 3), cA_31_23_12_21_32),
    ]
    beta_jobs = [
        ("w13b_c21", (1, 3), cB_21),
        ("w21b_c21_32", (2, 1), cB_21_32),
        ("w12b_c21_32_13", (1, 2), cB_21_32_13),
        ("w31b_c21_32_13_31", (3, 1), cB_21_32_13_31),
        ("w23b_c21_32_13_31_23", (2, 3), cB_21_32_13_31_23),
        ("w31b_c31", (3, 1), cB_31),
        ("w23b_c31_23", (2, 3), cB_31_23),
        ("w32b_c31_23_12", (3, 2), cB_31_23_12),
        ("w13b_c31_23_12_21", (1, 3), cB_31_23_12_21),
        ("w21b_c31_23_12_21_32", (2, 1), cB_31_23_12_21_32),
    ]
    gamma_jobs = [
        ("w21g_c21", (2, 1), cG_21),
        ("w32g_c21_32", (3, 2), cG_21_32),
        ("w23g_c21_32_13", (2, 3), cG_21_32_13),
        ("w12g_c21_32_13_31", (1, 2), cG_21_32_13_31),
        ("w31g_c21_32_13_31_23", (3, 1), cG_21_32_13_31_23),
        ("w12g_c31", (1, 2), cG_31),
        ("w31g_c31_23", (3, 1), cG_31_23),
        ("w13g_c31_23_12", (1, 3), cG_31_23_12),
        ("w21g_c31_23_12_21", (2, 1), cG_31_23_12_21),
        ("w32g_c31_23_12_21_32", (3, 2), cG_31_23_12_21_32),
    ]

    return alpha_jobs, beta_jobs, gamma_jobs


_GEN2_SAGNAC_JOB_TEMPLATES = _build_gen2_sagnac_job_templates()


def _build_gen2_michelson_job_templates():
    """Build static Gen-2 Michelson shift job templates for X/Y/Z."""
    # Base X-arm chains.
    cX_21 = ((2, 1),)
    cX_21_12 = ((2, 1), (1, 2))
    cX_21_12_31 = ((2, 1), (1, 2), (3, 1))
    cX_21_12_31_13 = ((2, 1), (1, 2), (3, 1), (1, 3))
    cX_21_12_31_13_31 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1))
    cX_21_12_31_13_31_13 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3))
    cX_21_12_31_13_31_13_21 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3), (2, 1))

    cX_31 = ((3, 1),)
    cX_31_13 = ((3, 1), (1, 3))
    cX_31_13_21 = ((3, 1), (1, 3), (2, 1))
    cX_31_13_21_12 = ((3, 1), (1, 3), (2, 1), (1, 2))
    cX_31_13_21_12_21 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1))
    cX_31_13_21_12_21_12 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2))
    cX_31_13_21_12_21_12_31 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2), (3, 1))

    x_jobs = [
        ("w12_cX_21", (1, 2), cX_21),
        ("w31_cX_21_12", (3, 1), cX_21_12),
        ("w13_cX_21_12_31", (1, 3), cX_21_12_31),
        ("w31_cX_21_12_31_13", (3, 1), cX_21_12_31_13),
        ("w13_cX_21_12_31_13_31", (1, 3), cX_21_12_31_13_31),
        ("w21_cX_21_12_31_13_31_13", (2, 1), cX_21_12_31_13_31_13),
        ("w12_cX_21_12_31_13_31_13_21", (1, 2), cX_21_12_31_13_31_13_21),
        ("w13_cX_31", (1, 3), cX_31),
        ("w21_cX_31_13", (2, 1), cX_31_13),
        ("w12_cX_31_13_21", (1, 2), cX_31_13_21),
        ("w21_cX_31_13_21_12", (2, 1), cX_31_13_21_12),
        ("w12_cX_31_13_21_12_21", (1, 2), cX_31_13_21_12_21),
        ("w31_cX_31_13_21_12_21_12", (3, 1), cX_31_13_21_12_21_12),
        ("w13_cX_31_13_21_12_21_12_31", (1, 3), cX_31_13_21_12_21_12_31),
    ]

    yz_skeleton = [
        ("t1", (1, 2), cX_21),
        ("t2", (3, 1), cX_21_12),
        ("t3", (1, 3), cX_21_12_31),
        ("t4", (3, 1), cX_21_12_31_13),
        ("t5", (1, 3), cX_21_12_31_13_31),
        ("t6", (2, 1), cX_21_12_31_13_31_13),
        ("t7", (1, 2), cX_21_12_31_13_31_13_21),
        ("n1", (1, 3), cX_31),
        ("n2", (2, 1), cX_31_13),
        ("n3", (1, 2), cX_31_13_21),
        ("n4", (2, 1), cX_31_13_21_12),
        ("n5", (1, 2), cX_31_13_21_12_21),
        ("n6", (3, 1), cX_31_13_21_12_21_12),
        ("n7", (1, 3), cX_31_13_21_12_21_12_31),
    ]

    def rot_jobs(k):
        return [(name, _rot_link(link, k), _rot_chain(chain, k)) for name, link, chain in yz_skeleton]

    y_jobs = rot_jobs(1)
    z_jobs = rot_jobs(2)
    return x_jobs, y_jobs, z_jobs


_GEN2_MICHELSON_JOB_TEMPLATES = _build_gen2_michelson_job_templates()


def _build_gen2_michelson_job_records(link_w, shifts_X, shifts_Y, shifts_Z):
    """Resolve the static Gen-2 Michelson templates into concrete shift jobs."""
    x_jobs_template, y_jobs_template, z_jobs_template = _GEN2_MICHELSON_JOB_TEMPLATES

    def build_jobs(prefix, shift_dict, jobs):
        return [(f"{prefix}:{name}", link_w[link], shift_dict[chain]) for name, link, chain in jobs]

    x_jobs = build_jobs("X", shifts_X, x_jobs_template)
    y_jobs = build_jobs("Y", shifts_Y, y_jobs_template)
    z_jobs = build_jobs("Z", shifts_Z, z_jobs_template)
    all_jobs = x_jobs + y_jobs + z_jobs
    return x_jobs, y_jobs, z_jobs, all_jobs


def _build_gen2_michelson_accumulator_job_specs(link_w, shifts_X, shifts_Y, shifts_Z):
    """Build dense base terms and signed shift jobs for accumulator assembly."""
    x_jobs_template, y_jobs_template, z_jobs_template = _GEN2_MICHELSON_JOB_TEMPLATES

    w21 = link_w[(2, 1)]
    w31 = link_w[(3, 1)]
    y_y231 = link_w[_rot_link((2, 1), 1)]
    y_y32p1 = link_w[_rot_link((3, 1), 1)]
    z_y231 = link_w[_rot_link((2, 1), 2)]
    z_y32p1 = link_w[_rot_link((3, 1), 2)]

    base_X = w21 - w31
    base_Y = y_y231 - y_y32p1
    base_Z = z_y231 - z_y32p1

    def build_acc_jobs(channel, shift_dict, jobs):
        acc_jobs = []
        for index, (name, link, chain) in enumerate(jobs):
            sign = 1.0 if index < 7 else -1.0
            acc_jobs.append((channel, sign, f"{channel}:{name}", link_w[link], shift_dict[chain]))
        return acc_jobs

    acc_jobs = build_acc_jobs("X", shifts_X, x_jobs_template)
    acc_jobs.extend(build_acc_jobs("Y", shifts_Y, y_jobs_template))
    acc_jobs.extend(build_acc_jobs("Z", shifts_Z, z_jobs_template))

    return base_X, base_Y, base_Z, acc_jobs


def _assemble_gen2_michelson_from_shifted_map(link_w, shifted_map):
    """Assemble the Gen-2 Michelson X/Y/Z combinations from shifted jobs."""
    w21 = link_w[(2, 1)]
    w32 = link_w[(3, 2)]
    w13 = link_w[(1, 3)]
    w31 = link_w[(3, 1)]
    w23 = link_w[(2, 3)]
    w12 = link_w[(1, 2)]

    y_y231 = link_w[_rot_link((2, 1), 1)]
    y_y32p1 = link_w[_rot_link((3, 1), 1)]
    z_y231 = link_w[_rot_link((2, 1), 2)]
    z_y32p1 = link_w[_rot_link((3, 1), 2)]

    X1 = (
        w21
        + shifted_map["X:w12_cX_21"] + shifted_map["X:w31_cX_21_12"] + shifted_map["X:w13_cX_21_12_31"]
        + shifted_map["X:w31_cX_21_12_31_13"] + shifted_map["X:w13_cX_21_12_31_13_31"]
        + shifted_map["X:w21_cX_21_12_31_13_31_13"] + shifted_map["X:w12_cX_21_12_31_13_31_13_21"]
        - w31
        - shifted_map["X:w13_cX_31"] - shifted_map["X:w21_cX_31_13"] - shifted_map["X:w12_cX_31_13_21"]
        - shifted_map["X:w21_cX_31_13_21_12"] - shifted_map["X:w12_cX_31_13_21_12_21"]
        - shifted_map["X:w31_cX_31_13_21_12_21_12"] - shifted_map["X:w13_cX_31_13_21_12_21_12_31"]
    )

    Y1 = (
        y_y231
        + shifted_map["Y:t1"] + shifted_map["Y:t2"] + shifted_map["Y:t3"] + shifted_map["Y:t4"]
        + shifted_map["Y:t5"] + shifted_map["Y:t6"] + shifted_map["Y:t7"]
        - y_y32p1
        - shifted_map["Y:n1"] - shifted_map["Y:n2"] - shifted_map["Y:n3"] - shifted_map["Y:n4"]
        - shifted_map["Y:n5"] - shifted_map["Y:n6"] - shifted_map["Y:n7"]
    )

    Z1 = (
        z_y231
        + shifted_map["Z:t1"] + shifted_map["Z:t2"] + shifted_map["Z:t3"] + shifted_map["Z:t4"]
        + shifted_map["Z:t5"] + shifted_map["Z:t6"] + shifted_map["Z:t7"]
        - z_y32p1
        - shifted_map["Z:n1"] - shifted_map["Z:n2"] - shifted_map["Z:n3"] - shifted_map["Z:n4"]
        - shifted_map["Z:n5"] - shifted_map["Z:n6"] - shifted_map["Z:n7"]
    )

    return X1, Y1, Z1


def _gen2_michelson_wdm_accumulate_downstream(
    wdm,
    link_w,
    x_jobs,
    y_jobs,
    z_jobs,
    all_jobs,
    *,
    shifts_X,
    shifts_Y,
    shifts_Z,
    precomp_Nf,
    batch_L_trunc,
    resolved_use_jax,
    batch_shift_kwargs,
    accumulator_chunk=1,
    # profiling passthrough
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
):
    """Assemble Gen-2 Michelson outputs by accumulating shifted jobs in chunks.

    When ``profile_shift_stages`` is True, forward profiling flags into the
    underlying batch shifter and aggregate per-chunk profiles. If
    ``return_shift_profile`` is True, return ``((X,Y,Z), profile)``.
    """
    base_X, base_Y, base_Z, acc_jobs = _build_gen2_michelson_accumulator_job_specs(
        link_w,
        shifts_X,
        shifts_Y,
        shifts_Z,
    )

    if len(acc_jobs) != len(all_jobs):
        raise RuntimeError(
            f"Accumulator job count mismatch: expected {len(all_jobs)}, built {len(acc_jobs)}."
        )

    X = np.array(base_X, copy=True)
    Y = np.array(base_Y, copy=True)
    Z = np.array(base_Z, copy=True)

    chunk_size = max(1, int(accumulator_chunk))
    shift_kwargs = dict(batch_shift_kwargs)
    shift_kwargs.pop("batch_chunk", None)

    prof_chunk_profiles = []

    for start in range(0, len(acc_jobs), chunk_size):
        chunk = acc_jobs[start : start + chunk_size]
        shifted_result = wdm_time_shift_variable_batch(
            wdm,
            [(w_input, t_shift) for _, _, _, w_input, t_shift in chunk],
            Nf=precomp_Nf,
            L_trunc=batch_L_trunc,
            use_jax=resolved_use_jax,
            batch_chunk=min(chunk_size, len(chunk)),
            profile_shift_stages=profile_shift_stages,
            profile_label=profile_label,
            **shift_kwargs,
        )

        if profile_shift_stages:
            try:
                shifted_chunk, chunk_profile = shifted_result
                prof_chunk_profiles.append(chunk_profile)
            except Exception:
                shifted_chunk = shifted_result
        else:
            shifted_chunk = shifted_result

        if len(shifted_chunk) != len(chunk):
            raise RuntimeError(
                f"Expected {len(chunk)} shifted arrays, got {len(shifted_chunk)}."
            )

        for (channel, sign, _, _, _), shifted in zip(chunk, shifted_chunk):
            shifted_arr = np.asarray(shifted)
            if channel == "X":
                X += sign * shifted_arr
            elif channel == "Y":
                Y += sign * shifted_arr
            elif channel == "Z":
                Z += sign * shifted_arr
            else:
                raise ValueError(f"Unknown Gen-2 Michelson channel: {channel}")

    if profile_shift_stages and return_shift_profile:
        try:
            agg = {
                "n_jobs": len(acc_jobs),
                "batch_chunk": accumulator_chunk,
                "total_s": sum(float(p.get("total_s", 0.0)) for p in prof_chunk_profiles) if prof_chunk_profiles else 0.0,
                "tl_tp_prepare_s": sum(float(p.get("tl_tp_prepare_s", 0.0)) for p in prof_chunk_profiles) if prof_chunk_profiles else 0.0,
                "jax_assembly_s": sum(float(p.get("jax_assembly_s", 0.0)) for p in prof_chunk_profiles) if prof_chunk_profiles else 0.0,
                "postprocess_s": sum(float(p.get("postprocess_s", 0.0)) for p in prof_chunk_profiles) if prof_chunk_profiles else 0.0,
                "other_s": sum(float(p.get("other_s", 0.0)) for p in prof_chunk_profiles) if prof_chunk_profiles else 0.0,
                "label": profile_label,
            }
        except Exception:
            agg = {"n_jobs": len(acc_jobs), "label": profile_label}
        return (X, Y, Z), agg

    return X, Y, Z


def precompute_gen2_michelson_wdm_inputs(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    L_trunc=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
    one_arm_cache=None,
    shifts_cache=None,
):
    """Backward-compatible staged Gen-2 Michelson precompute wrapper.

    This wrapper now builds:
    A0 geometry/static precompute -> A1 waveform precompute.
    """
    geom_precomp = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=Nf,
        L_trunc=L_trunc,
        q=q,
        d=4,
        use_jax=use_jax,
        variable_shift_kwargs=variable_shift_kwargs,
        shifts_cache=shifts_cache,
    )
    return precompute_gen2_michelson_wdm_waveform(
        geom_precomp,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=variable_shift_kwargs,
        one_arm_cache=one_arm_cache,
    )


def precompute_gen2_michelson_wdm_geometry(
    times,
    theta,
    phi,
    orbits,
    *,
    Nf,
    L_trunc=None,
    q=None,
    d=4,
    use_jax=None,
    variable_shift_kwargs=None,
    shifts_cache=None,
):
    """A0 precompute: build geometry/static Gen-2 Michelson state.

    This stage is waveform-independent and reusable across many waveforms on
    the same time grid, sky location, orbit object, and WDM settings.
    """
    from WDM_LISAresponse.code.lisa_response.response import _build_wdm_dT_context

    times = np.asarray(times, dtype=float)
    one_arm_shift_kwargs, tdi_batch_shift_kwargs = _split_variable_shift_kwargs(variable_shift_kwargs)

    # Build only WDM/grid metadata using dummy waveforms (no one-arm responses).
    dummy = np.zeros(times.shape[0], dtype=float)
    wdm_context = _build_wdm_dT_context(
        times,
        dummy,
        dummy,
        Nf=Nf,
        d=d,
        q=q,
        calc_m0=True,
    )
    wdm = wdm_context["wdm"]
    times_p = float(times[0]) + np.arange(wdm.Nt, dtype=float) * float(wdm.dT)

    cache_key = ("michelson_wdm", id(orbits), int(wdm.Nt), float(times[0]), float(wdm.dT), int(Nf))
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_X, shifts_Y, shifts_Z = shifts_cache[cache_key]
    else:
        shifts_X, shifts_Y, shifts_Z = precompute_gen2_michelson_shifts(times_p, orbits)
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_X, shifts_Y, shifts_Z)

    return {
        "kind": "gen2_michelson_wdm_geometry",
        "times": times,
        "theta": float(theta),
        "phi": float(phi),
        "orbits": orbits,
        "wdm": wdm,
        "times_p": times_p,
        "shifts_X": shifts_X,
        "shifts_Y": shifts_Y,
        "shifts_Z": shifts_Z,
        "job_templates": _GEN2_MICHELSON_JOB_TEMPLATES,
        "one_arm_shift_kwargs": one_arm_shift_kwargs,
        "tdi_batch_shift_kwargs": tdi_batch_shift_kwargs,
        "Nf": int(Nf),
        "L_trunc": None if L_trunc is None else int(L_trunc),
        "q": q,
        "d": int(d),
        "use_jax": use_jax,
    }


def precompute_gen2_michelson_wdm_waveform(
    geom_precomp,
    waveform_plus,
    waveform_cross,
    *,
    variable_shift_kwargs=None,
    one_arm_cache=None,
):
    """A1 precompute: build waveform-specific one-arm links and TDI jobs."""
    times = geom_precomp["times"]
    theta = geom_precomp["theta"]
    phi = geom_precomp["phi"]
    orbits = geom_precomp["orbits"]
    Nf = int(geom_precomp["Nf"])
    L_trunc = geom_precomp.get("L_trunc")
    use_jax = geom_precomp.get("use_jax")
    q = geom_precomp.get("q")

    one_arm_shift_kwargs = dict(geom_precomp.get("one_arm_shift_kwargs", {}))
    tdi_batch_shift_kwargs = dict(geom_precomp.get("tdi_batch_shift_kwargs", {}))

    if variable_shift_kwargs is not None:
        one_arm_override, tdi_override = _split_variable_shift_kwargs(variable_shift_kwargs)
        one_arm_shift_kwargs.update(one_arm_override)
        tdi_batch_shift_kwargs.update(tdi_override)

    t_ctx0 = time.perf_counter()
    wave_context = _build_wdm_dT_context_from_wdm(
        times,
        waveform_plus,
        waveform_cross,
        geom_precomp["wdm"],
        m_offset=0.0,
    )
    waveform_context_build_s = time.perf_counter() - t_ctx0

    cache = _resolve_one_arm_cache(one_arm_cache)
    t_links0 = time.perf_counter()
    wdm, link_w = precompute_one_arm_wdm_dT_links_from_context(
        wave_context,
        orbits,
        theta,
        phi,
        L_trunc=L_trunc,
        cache=cache,
        variable_shift_kwargs=one_arm_shift_kwargs,
        use_jax=use_jax,
    )
    six_link_compute_s = time.perf_counter() - t_links0

    t_jobs0 = time.perf_counter()
    x_jobs, y_jobs, z_jobs, all_jobs = _build_gen2_michelson_job_records(
        link_w,
        geom_precomp["shifts_X"],
        geom_precomp["shifts_Y"],
        geom_precomp["shifts_Z"],
    )
    job_build_s = time.perf_counter() - t_jobs0

    return {
        "kind": "gen2_michelson_wdm_waveform",
        "geometry_precomp": geom_precomp,
        "wdm": wdm,
        "times_p": geom_precomp["times_p"],
        "link_w": link_w,
        "shifts_X": geom_precomp["shifts_X"],
        "shifts_Y": geom_precomp["shifts_Y"],
        "shifts_Z": geom_precomp["shifts_Z"],
        "x_jobs": x_jobs,
        "y_jobs": y_jobs,
        "z_jobs": z_jobs,
        "all_jobs": all_jobs,
        "job_templates": geom_precomp["job_templates"],
        "one_arm_shift_kwargs": one_arm_shift_kwargs,
        "tdi_batch_shift_kwargs": tdi_batch_shift_kwargs,
        "Nf": Nf,
        "L_trunc": L_trunc,
        "use_jax": use_jax,
        "q": q,
        "waveform_context_build_s": float(waveform_context_build_s),
        "six_link_compute_s": float(six_link_compute_s),
        "job_build_s": float(job_build_s),
    }


def gen2_michelson_wdm_from_waveform_precomputed(
    wave_precomp,
    *,
    variable_shift_kwargs=None,
    L_trunc=None,
    Nf=None,
    use_jax=None,
    tdi_assembly_mode="batch_map",
    accumulator_chunk=1,
    # profiling options (opt-in)
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
):
    """B stage: run downstream TDI batch shifts from waveform precompute."""
    if "link_w" not in wave_precomp:
        raise ValueError("wave_precomp is missing link_w; expected waveform-level precompute payload.")

    wdm = wave_precomp["wdm"]
    link_w = wave_precomp["link_w"]
    all_jobs = wave_precomp.get("all_jobs")
    if all_jobs is None:
        all_jobs = wave_precomp["x_jobs"] + wave_precomp["y_jobs"] + wave_precomp["z_jobs"]

    precomp_Nf = int(wave_precomp.get("Nf", wdm.Nf))
    if Nf is not None and int(Nf) != precomp_Nf:
        raise ValueError(f"Nf must match the precomputed WDM grid; expected {precomp_Nf}, got {Nf}.")

    precomp_L_trunc = wave_precomp.get("L_trunc", None)
    if L_trunc is not None:
        if precomp_L_trunc is None:
            raise ValueError("L_trunc cannot be changed after precomputing staged Gen-2 shifts.")
        if int(L_trunc) != int(precomp_L_trunc):
            raise ValueError(
                f"L_trunc must match the precomputed shifts; expected {precomp_L_trunc}, got {L_trunc}."
            )

    resolved_use_jax = wave_precomp.get("use_jax", use_jax)
    if use_jax is not None:
        resolved_use_jax = use_jax

    batch_shift_kwargs = dict(wave_precomp.get("tdi_batch_shift_kwargs", {}))
    override_shift_kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)
    override_shift_kwargs.pop("use_jax", None)
    batch_shift_kwargs.update(override_shift_kwargs)

    batch_shift_kwargs = _finalize_variable_shift_batch_kwargs(batch_shift_kwargs, len(all_jobs))
    batch_L_trunc = L_trunc if L_trunc is not None else precomp_L_trunc

    if tdi_assembly_mode == "batch_map":
        shifted_result = wdm_time_shift_variable_batch(
            wdm,
            [(w, t_shift) for _, w, t_shift in all_jobs],
            Nf=precomp_Nf,
            L_trunc=batch_L_trunc,
            use_jax=resolved_use_jax,
            profile_shift_stages=profile_shift_stages,
            profile_label=profile_label,
            **batch_shift_kwargs,
        )

        if profile_shift_stages:
            all_shifted, shift_profile = shifted_result
        else:
            all_shifted = shifted_result

        shifted_map = {name: arr for (name, _, _), arr in zip(all_jobs, all_shifted)}
        assembled = _assemble_gen2_michelson_from_shifted_map(link_w, shifted_map)
        if profile_shift_stages and return_shift_profile:
            return assembled, shift_profile
        return assembled

    if tdi_assembly_mode == "accumulate":
        return _gen2_michelson_wdm_accumulate_downstream(
            wdm,
            link_w,
            wave_precomp["x_jobs"],
            wave_precomp["y_jobs"],
            wave_precomp["z_jobs"],
            all_jobs,
            shifts_X=wave_precomp["shifts_X"],
            shifts_Y=wave_precomp["shifts_Y"],
            shifts_Z=wave_precomp["shifts_Z"],
            precomp_Nf=precomp_Nf,
            batch_L_trunc=batch_L_trunc,
            resolved_use_jax=resolved_use_jax,
            batch_shift_kwargs=batch_shift_kwargs,
            accumulator_chunk=accumulator_chunk,
            profile_shift_stages=profile_shift_stages,
            profile_label=profile_label,
            return_shift_profile=return_shift_profile,
        )

    raise ValueError(
        f"Unknown tdi_assembly_mode={tdi_assembly_mode!r}; expected 'batch_map' or 'accumulate'."
    )


def gen2_michelson_wdm_from_precomputed(
    precomp,
    variable_shift_kwargs=None,
    L_trunc=None,
    use_jax=None,
    Nf=None,
    tdi_assembly_mode="batch_map",
    accumulator_chunk=1,
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
):
    """Backward-compatible alias for waveform-level staged execution.

    Accepts both old combined staged payloads and new waveform-level payloads.
    """
    return gen2_michelson_wdm_from_waveform_precomputed(
        precomp,
        variable_shift_kwargs=variable_shift_kwargs,
        L_trunc=L_trunc,
        Nf=Nf,
        use_jax=use_jax,
        tdi_assembly_mode=tdi_assembly_mode,
        accumulator_chunk=accumulator_chunk,
        profile_shift_stages=profile_shift_stages,
        profile_label=profile_label,
        return_shift_profile=return_shift_profile,
    )

def gen1_sagnac_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    Lbar,
    return_zeta=False,
    L_trunc=None,
    one_arm_cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
):
    """
    Implement the generation 1 version of the sagnac-type variables in the wdm domain.

    Parameters
    ----------
    times : ndarray 
        array of times for the waveforms 
    waveform_plus : ndarray
        plus polarization of the wdm waveform 
    waveform_cross : ndarray 
        cross polarization of the wdm waveform
    theta : float 
        polar coordinate of the incoming signal 
    phi : float 
        azimuthal coordinate of the incoming signal 
    orbits : LISA orbits class objects
        computes the positions of all the SC
    Nf : int
        number of frequency cells in wdm domain
    Lbar : dict 
        dictionary encoding the fixed lengths between satelites 
    return_zeta : bool 
        return the 4th symmetric variable zeta 
    L_trunc : int or None
        Optional lag truncation forwarded to the fixed-delay WDM shifter.
    one_arm_cache : dict or None
        Optional cache for shared one-arm WDM responses.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to the one-arm variable-delay shift
        used when constructing the directed link responses.
    q : int or None, optional
        WDM truncation parameter for one-arm response construction. When
        ``None``, the maximal allowed q is used (same behavior as one-arm
        response helpers and Gen-2 APIs).
    
    
    

    Returns
    -------
    alpha : ndarray
        alpha sagnac-like wdm coefficients Nt x Nf
    beta : ndarray
        beta sagnac-like wdm coefficients Nt x Nf
    gamma : ndarray
        gamma sagnac-like wdm coefficients Nt x Nf

    """

    cache = _resolve_one_arm_cache(one_arm_cache)
    wdm, link_w = precompute_one_arm_wdm_dT_links(
        times,
        orbits,
        theta,
        phi,
        waveform_plus,
        waveform_cross,
        Nf=Nf,
        L_trunc=L_trunc,
        cache=cache,
        variable_shift_kwargs=variable_shift_kwargs,
        use_jax=use_jax,
        q=q,
    )

    w21 = link_w[(2, 1)]
    w32 = link_w[(3, 2)]
    w13 = link_w[(1, 3)]
    w31 = link_w[(3, 1)]
    w23 = link_w[(2, 3)]
    w12 = link_w[(1, 2)]

    jobs = [
        ("w32_L21", w32, Lbar[(2, 1)]),
        ("w13_L32_L21", w13, Lbar[(2, 1)] + Lbar[(3, 2)]),
        ("w23_L13", w23, Lbar[(1, 3)]),
        ("w12_L32_L13", w12, Lbar[(3, 2)] + Lbar[(1, 3)]),
        ("w13_L32", w13, Lbar[(3, 2)]),
        ("w21_L13_L32", w21, Lbar[(3, 2)] + Lbar[(1, 3)]),
        ("w31_L21", w31, Lbar[(2, 1)]),
        ("w23_L21_L13", w23, Lbar[(2, 1)] + Lbar[(1, 3)]),
        ("w21_L13", w21, Lbar[(1, 3)]),
        ("w32_L21_L13", w32, Lbar[(2, 1)] + Lbar[(1, 3)]),
        ("w12_L32", w12, Lbar[(3, 2)]),
        ("w31_L32_L21", w31, Lbar[(3, 2)] + Lbar[(2, 1)]),
    ]
    shifted = wdm_time_shift_fixed_batch(
        wdm,
        [(w, delta) for _, w, delta in jobs],
        Nf=Nf,
        L_trunc=L_trunc,
        use_jax=use_jax,
    )
    shifted_map = {name: arr for (name, _, _), arr in zip(jobs, shifted)}

    w32_L21 = shifted_map["w32_L21"]
    w13_L32_L21 = shifted_map["w13_L32_L21"]
    w23_L13 = shifted_map["w23_L13"]
    w12_L32_L13 = shifted_map["w12_L32_L13"]
    w13_L32 = shifted_map["w13_L32"]
    w21_L13_L32 = shifted_map["w21_L13_L32"]
    w31_L21 = shifted_map["w31_L21"]
    w23_L21_L13 = shifted_map["w23_L21_L13"]
    w21_L13 = shifted_map["w21_L13"]
    w32_L21_L13 = shifted_map["w32_L21_L13"]
    w12_L32 = shifted_map["w12_L32"]
    w31_L32_L21 = shifted_map["w31_L32_L21"]

    alpha = w21 + w32_L21 + w13_L32_L21 - w31 - w23_L13 - w12_L32_L13
    beta = w32 + w13_L32 + w21_L13_L32 - w12 - w31_L21 - w23_L21_L13
    gamma = w13 + w21_L13 + w32_L21_L13 - w23 - w12_L32 - w31_L32_L21

    if return_zeta == True: 
        zeta_jobs = [
            (w21, Lbar[(3, 2)]),
            (w31, Lbar[(3, 2)]),
            (w32, Lbar[(1, 3)]),
            (w12, Lbar[(1, 3)]),
            (w13, Lbar[(2, 1)]),
            (w23, Lbar[(2, 1)]),
        ]
        w21_L32, w31_L32, w32_L13, w12_L13, w13_L21, w23_L21 = wdm_time_shift_fixed_batch(
            wdm,
            zeta_jobs,
            Nf=Nf,
            L_trunc=L_trunc,
            use_jax=use_jax,
        )

        zeta = w21_L32 - w31_L32 + w32_L13 - w12_L13 + w13_L21 - w23_L21


        return alpha, beta, gamma, zeta 

    else:
        return alpha, beta, gamma 
    

def gen1_sagnac_td(times, waveform_plus, waveform_cross, theta, phi, orbits, Lbar, return_zeta = False):
    """
    Implement the generation 1 version of the sagnac-type variables in the time domain.

    Parameters
    ----------
    times : ndarray 
        array of times for the waveforms 
    waveform_plus : ndarray
        plus polarization of the waveform 
    waveform_cross : ndarray 
        cross polarization of the waveform
    theta : float 
        polar coordinate of the incoming signal 
    phi : float 
        azimuthal coordinate of the incoming signal 
    orbits : LISA orbits class objects
        computes the positions of all the SC
    Lbar : dict 
        dictionary encoding the fixed lengths between satelites 
    return_zeta : bool 
        return the 4th symmetric variable zeta 
    

    Returns
    -------
    alpha : ndarray
        alpha sagnac-type variable of length len(times)
    beta : ndarray
        beta sagnac-type variable of length len(times)
    gamma : ndarray
        gamma sagnac-type variable of length len(times)

    """

    #generate the unshifted terms 

    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)

    dt = times[1] - times[0]

    #generate shifted terms 

    #for alpha
    dl_32_L21 = fft_time_shift(dl_32, dt, Lbar[(2,1)])
    dl_13_L32_L21 = fft_time_shift(dl_13, dt, Lbar[(2,1)] + Lbar[(3,2)])
    dl_23_L13 = fft_time_shift(dl_23, dt, Lbar[(1,3)])
    dl_12_L32_L13 = fft_time_shift(dl_12, dt, Lbar[(3,2)] + Lbar[(1,3)])
    #for beta
    dl_13_L32 = fft_time_shift(dl_13, dt, Lbar[(3,2)])
    dl_21_L13_L32 = fft_time_shift(dl_21, dt, Lbar[(1,3)] + Lbar[(3,2)])
    dl_31_L21 = fft_time_shift(dl_31, dt, Lbar[(2,1)])
    dl_23_L21_L13 = fft_time_shift(dl_23, dt, Lbar[(2,1)] + Lbar[(1,3)])
    #for gamma 
    dl_21_L13 = fft_time_shift(dl_21, dt, Lbar[(1,3)])
    dl_32_L21_L13 = fft_time_shift(dl_32, dt, Lbar[(2,1)] + Lbar[(1,3)])
    dl_12_L32 = fft_time_shift(dl_12, dt, Lbar[(3,2)])
    dl_31_L32_L21 = fft_time_shift(dl_31, dt, Lbar[(3,2)] + Lbar[(2,1)])

    alpha = dl_21 + dl_32_L21 + dl_13_L32_L21 - dl_31 - dl_23_L13 - dl_12_L32_L13
    beta = dl_32 + dl_13_L32 + dl_21_L13_L32 - dl_12 - dl_31_L21 - dl_23_L21_L13
    gamma = dl_13 + dl_21_L13 + dl_32_L21_L13 - dl_23 - dl_12_L32 - dl_31_L32_L21

    if return_zeta == True: 
        #for zeta 
        dl_21_L32 = fft_time_shift(dl_21, dt, Lbar[(3,2)])
        dl_31_L32 = fft_time_shift(dl_31, dt, Lbar[(3,2)])
        dl_32_L13 = fft_time_shift(dl_32, dt, Lbar[(1,3)])
        dl_12_L13 = fft_time_shift(dl_12, dt, Lbar[(1,3)])
        dl_13_L21 = fft_time_shift(dl_13, dt, Lbar[(2,1)])
        dl_23_L21 = fft_time_shift(dl_23, dt, Lbar[(2,1)])

        zeta = dl_21_L32 - dl_31_L32 + dl_32_L13 - dl_12_L13 + dl_13_L21 - dl_23_L21


        return alpha, beta, gamma, zeta 

    else:
        return alpha, beta, gamma
    
def gen1_michelson_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    Lbar,
    L_trunc=None,
    one_arm_cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
):
    """
    Implement the generation 1 Michelson-type variables (X, Y, Z) in the WDM domain.

    X is from Eq. (3.23):
      X = y231 + y13′2,3 + y32′1,33 + y123,233
        - y32′1 - y123,2 - y231,22 - y13′2,322

    Y and Z are obtained by cyclic permutations of the indices.

    Parameters
    ----------
    times : ndarray 
        array of times for the waveforms 
    waveform_plus : ndarray
        plus polarization of the wdm waveform 
    waveform_cross : ndarray 
        cross polarization of the wdm waveform
    theta : float 
        polar coordinate of the incoming signal 
    phi : float 
        azimuthal coordinate of the incoming signal 
    orbits : LISA orbits class objects
        computes the positions of all the SC
    Nf : int
        number of frequency cells in wdm domain
    Lbar : dict 
        dictionary encoding the fixed lengths between satelites 
    L_trunc : int or None
        Optional lag truncation forwarded to the fixed-delay WDM shifter.
    one_arm_cache : dict or None
        Optional cache for shared one-arm WDM responses.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to the one-arm variable-delay shift
        used when constructing the directed link responses.
    q : int or None, optional
        WDM truncation parameter for one-arm response construction. When
        ``None``, the maximal allowed q is used (same behavior as one-arm
        response helpers and Gen-2 APIs).

    Returns
    -------
    X, Y, Z : ndarray
        Nt x Nf WDM coefficients
    """

    cache = _resolve_one_arm_cache(one_arm_cache)
    wdm, link_w = precompute_one_arm_wdm_dT_links(
        times,
        orbits,
        theta,
        phi,
        waveform_plus,
        waveform_cross,
        Nf=Nf,
        L_trunc=L_trunc,
        cache=cache,
        variable_shift_kwargs=variable_shift_kwargs,
        use_jax=use_jax,
        q=q,
    )

    w21 = link_w[(2, 1)]
    w32 = link_w[(3, 2)]
    w13 = link_w[(1, 3)]
    w31 = link_w[(3, 1)]
    w23 = link_w[(2, 3)]
    w12 = link_w[(1, 2)]

    jobs = [
        ("w12_L21", w12, Lbar[(2, 1)]),
        ("w31_L21_L21", w31, Lbar[(2, 1)] + Lbar[(2, 1)]),
        ("w13_L13_L21_L21", w13, Lbar[(1, 3)] + Lbar[(2, 1)] + Lbar[(2, 1)]),
        ("w13_L13", w13, Lbar[(1, 3)]),
        ("w21_L13_L13", w21, Lbar[(1, 3)] + Lbar[(1, 3)]),
        ("w12_L21_L13_L13", w12, Lbar[(2, 1)] + Lbar[(1, 3)] + Lbar[(1, 3)]),
        ("w23_L32", w23, Lbar[(3, 2)]),
        ("w12_L32_L32", w12, Lbar[(3, 2)] + Lbar[(3, 2)]),
        ("w21_L21_L32_L32", w21, Lbar[(2, 1)] + Lbar[(3, 2)] + Lbar[(3, 2)]),
        ("w21_L21", w21, Lbar[(2, 1)]),
        ("w32_L21_L21", w32, Lbar[(2, 1)] + Lbar[(2, 1)]),
        ("w23_L32_L21_L21", w23, Lbar[(3, 2)] + Lbar[(2, 1)] + Lbar[(2, 1)]),
        ("w31_L13", w31, Lbar[(1, 3)]),
        ("w23_L13_L13", w23, Lbar[(1, 3)] + Lbar[(1, 3)]),
        ("w32_L32_L13_L13", w32, Lbar[(3, 2)] + Lbar[(1, 3)] + Lbar[(1, 3)]),
        ("w32_L32", w32, Lbar[(3, 2)]),
        ("w13_L32_L32", w13, Lbar[(3, 2)] + Lbar[(3, 2)]),
        ("w31_L13_L32_L32", w31, Lbar[(1, 3)] + Lbar[(3, 2)] + Lbar[(3, 2)]),
    ]
    shifted = wdm_time_shift_fixed_batch(
        wdm,
        [(w, delta) for _, w, delta in jobs],
        Nf=Nf,
        L_trunc=L_trunc,
        use_jax=use_jax,
    )
    shifted_map = {name: arr for (name, _, _), arr in zip(jobs, shifted)}

    w12_L21 = shifted_map["w12_L21"]
    w31_L21_L21 = shifted_map["w31_L21_L21"]
    w13_L13_L21_L21 = shifted_map["w13_L13_L21_L21"]
    w13_L13 = shifted_map["w13_L13"]
    w21_L13_L13 = shifted_map["w21_L13_L13"]
    w12_L21_L13_L13 = shifted_map["w12_L21_L13_L13"]
    w23_L32 = shifted_map["w23_L32"]
    w12_L32_L32 = shifted_map["w12_L32_L32"]
    w21_L21_L32_L32 = shifted_map["w21_L21_L32_L32"]
    w21_L21 = shifted_map["w21_L21"]
    w32_L21_L21 = shifted_map["w32_L21_L21"]
    w23_L32_L21_L21 = shifted_map["w23_L32_L21_L21"]
    w31_L13 = shifted_map["w31_L13"]
    w23_L13_L13 = shifted_map["w23_L13_L13"]
    w32_L32_L13_L13 = shifted_map["w32_L32_L13_L13"]
    w32_L32 = shifted_map["w32_L32"]
    w13_L32_L32 = shifted_map["w13_L32_L32"]
    w31_L13_L32_L32 = shifted_map["w31_L13_L32_L32"]

    X = (w21 + w12_L21 + w31_L21_L21 + w13_L13_L21_L21) - (w31 + w13_L13 + w21_L13_L13 + w12_L21_L13_L13)

    Y = (w32 + w23_L32 + w12_L32_L32 + w21_L21_L32_L32) - (w12 + w21_L21 + w32_L21_L21 + w23_L32_L21_L21)

    Z = (w13 + w31_L13 + w23_L13_L13 + w32_L32_L13_L13) - (w23 + w32_L32 + w13_L32_L32 + w31_L13_L32_L32)

    return X, Y, Z

def gen1_michelson_td(times, waveform_plus, waveform_cross, theta, phi, orbits, Lbar):
    """
    Implement the generation 1 Michelson-type variables (X, Y, Z) in the time domain.

    X is from Eq. (3.23):
      X = y231 + y13′2,3 + y32′1,33 + y123,233
        - y32′1 - y123,2 - y231,22 - y13′2,322

    Y and Z are obtained by cyclic permutations of the indices.

    Parameters
    ----------
    times : ndarray
        array of times for the waveforms
    waveform_plus : ndarray
        plus polarization of the waveform (time domain)
    waveform_cross : ndarray
        cross polarization of the waveform (time domain)
    theta : float
        polar coordinate of the incoming signal
    phi : float
        azimuthal coordinate of the incoming signal
    orbits : LISA orbits class objects
        computes the positions of all the SC
    Lbar : dict 
        dictionary encoding the fixed lengths between satelites 

    Returns
    -------
    X, Y, Z : ndarray
        time-domain Michelson variables, length len(times)
    """

    # generate the unshifted terms
    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)

    dt = times[1] - times[0]

    # for X
    dl_12_L21 = fft_time_shift(dl_12, dt, Lbar[(2, 1)])
    dl_31_L21_L21 = fft_time_shift(dl_31, dt, Lbar[(2, 1)] + Lbar[(2, 1)])
    dl_13_L13_L21_L21 = fft_time_shift(dl_13, dt, Lbar[(1, 3)] + Lbar[(2, 1)] + Lbar[(2, 1)])
    dl_13_L13 = fft_time_shift(dl_13, dt, Lbar[(1, 3)])
    dl_21_L13_L13 = fft_time_shift(dl_21, dt, Lbar[(1, 3)] + Lbar[(1, 3)])
    dl_12_L21_L13_L13 = fft_time_shift(dl_12, dt, Lbar[(2, 1)] + Lbar[(1, 3)] + Lbar[(1, 3)])

    X = (dl_21 + dl_12_L21 + dl_31_L21_L21 + dl_13_L13_L21_L21) - (
        dl_31 + dl_13_L13 + dl_21_L13_L13 + dl_12_L21_L13_L13
    )

    # for Y
    dl_23_L32 = fft_time_shift(dl_23, dt, Lbar[(3, 2)])
    dl_12_L32_L32 = fft_time_shift(dl_12, dt, Lbar[(3, 2)] + Lbar[(3, 2)])
    dl_21_L21_L32_L32 = fft_time_shift(dl_21, dt, Lbar[(2, 1)] + Lbar[(3, 2)] + Lbar[(3, 2)])
    dl_21_L21 = fft_time_shift(dl_21, dt, Lbar[(2, 1)])
    dl_32_L21_L21 = fft_time_shift(dl_32, dt, Lbar[(2, 1)] + Lbar[(2, 1)])
    dl_23_L32_L21_L21 = fft_time_shift(dl_23, dt, Lbar[(3, 2)] + Lbar[(2, 1)] + Lbar[(2, 1)])

    Y = (dl_32 + dl_23_L32 + dl_12_L32_L32 + dl_21_L21_L32_L32) - (
        dl_12 + dl_21_L21 + dl_32_L21_L21 + dl_23_L32_L21_L21
    )

    # for Z
    dl_31_L13 = fft_time_shift(dl_31, dt, Lbar[(1, 3)])
    dl_23_L13_L13 = fft_time_shift(dl_23, dt, Lbar[(1, 3)] + Lbar[(1, 3)])
    dl_32_L32_L13_L13 = fft_time_shift(dl_32, dt, Lbar[(3, 2)] + Lbar[(1, 3)] + Lbar[(1, 3)])
    dl_32_L32 = fft_time_shift(dl_32, dt, Lbar[(3, 2)])
    dl_13_L32_L32 = fft_time_shift(dl_13, dt, Lbar[(3, 2)] + Lbar[(3, 2)])
    dl_31_L13_L32_L32 = fft_time_shift(dl_31, dt, Lbar[(1, 3)] + Lbar[(3, 2)] + Lbar[(3, 2)])

    Z = (dl_13 + dl_31_L13 + dl_23_L13_L13 + dl_32_L32_L13_L13) - (
        dl_23 + dl_32_L32 + dl_13_L32_L32 + dl_31_L13_L32_L32
    )

    return X, Y, Z

def gen2_sagnac_td(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    order=7,
    mode="zero",
    num_int_points=1000,
    shifts_cache=None,
):
    """
    Gen-2 Sagnac-type variables (alpha1, beta1, gamma1) in TIME DOMAIN,
    with built-in precomputation of gen2 shifts.

    Parameters
    ----------
    shifts_cache : dict or None
        Optional mutable dict used to memoize computed shift libraries.
        Keyed by ("sagnac", id(orbits), len(times), times[0], times[1]-times[0]).
    """

    times = np.asarray(times, dtype=float)
    dt = times[1] - times[0]

    # -----------------------
    # built-in shift precompute (on TD grid)
    # -----------------------
    cache_key = ("sagnac", id(orbits), len(times), float(times[0]), float(dt))
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_alpha, shifts_beta, shifts_gamma = shifts_cache[cache_key]
    else:
        shifts_alpha, shifts_beta, shifts_gamma = precompute_gen2_sagnac_shifts(times, orbits)
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_alpha, shifts_beta, shifts_gamma)

    # -----------------------
    # unshifted one-arm responses (directed links)
    # -----------------------
    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)

    def TS(x, shift_dict, chain):
        return interp_time_shift(x, dt, shift_dict[chain], order=order, mode=mode)

    # --- cyclic permutation helper ---
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    # =====================================================================
    # Base chains for ALPHA (must match your alpha_patterns in precompute)
    # =====================================================================
    cA_21                  = ((2, 1),)
    cA_21_32               = ((2, 1), (3, 2))
    cA_21_32_13            = ((2, 1), (3, 2), (1, 3))
    cA_21_32_13_31         = ((2, 1), (3, 2), (1, 3), (3, 1))
    cA_21_32_13_31_23      = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3))

    cA_31                  = ((3, 1),)
    cA_31_23               = ((3, 1), (2, 3))
    cA_31_23_12            = ((3, 1), (2, 3), (1, 2))
    cA_31_23_12_21         = ((3, 1), (2, 3), (1, 2), (2, 1))
    cA_31_23_12_21_32      = ((3, 1), (2, 3), (1, 2), (2, 1), (3, 2))

    cB_21                  = rot_chain(cA_21, 1)
    cB_21_32               = rot_chain(cA_21_32, 1)
    cB_21_32_13            = rot_chain(cA_21_32_13, 1)
    cB_21_32_13_31         = rot_chain(cA_21_32_13_31, 1)
    cB_21_32_13_31_23      = rot_chain(cA_21_32_13_31_23, 1)

    cB_31                  = rot_chain(cA_31, 1)
    cB_31_23               = rot_chain(cA_31_23, 1)
    cB_31_23_12            = rot_chain(cA_31_23_12, 1)
    cB_31_23_12_21         = rot_chain(cA_31_23_12_21, 1)
    cB_31_23_12_21_32      = rot_chain(cA_31_23_12_21_32, 1)

    cG_21                  = rot_chain(cA_21, 2)
    cG_21_32               = rot_chain(cA_21_32, 2)
    cG_21_32_13            = rot_chain(cA_21_32_13, 2)
    cG_21_32_13_31         = rot_chain(cA_21_32_13_31, 2)
    cG_21_32_13_31_23      = rot_chain(cA_21_32_13_31_23, 2)

    cG_31                  = rot_chain(cA_31, 2)
    cG_31_23               = rot_chain(cA_31_23, 2)
    cG_31_23_12            = rot_chain(cA_31_23_12, 2)
    cG_31_23_12_21         = rot_chain(cA_31_23_12_21, 2)
    cG_31_23_12_21_32      = rot_chain(cA_31_23_12_21_32, 2)

    # =====================================================================
    # ALPHA1
    # =====================================================================
    dl_32_c21                  = TS(dl_32, shifts_alpha, cA_21)
    dl_13_c21_32               = TS(dl_13, shifts_alpha, cA_21_32)
    dl_31_c21_32_13            = TS(dl_31, shifts_alpha, cA_21_32_13)
    dl_23_c21_32_13_31         = TS(dl_23, shifts_alpha, cA_21_32_13_31)
    dl_12_c21_32_13_31_23      = TS(dl_12, shifts_alpha, cA_21_32_13_31_23)

    dl_23_c31                  = TS(dl_23, shifts_alpha, cA_31)
    dl_12_c31_23               = TS(dl_12, shifts_alpha, cA_31_23)
    dl_21_c31_23_12            = TS(dl_21, shifts_alpha, cA_31_23_12)
    dl_32_c31_23_12_21         = TS(dl_32, shifts_alpha, cA_31_23_12_21)
    dl_13_c31_23_12_21_32      = TS(dl_13, shifts_alpha, cA_31_23_12_21_32)

    alpha1 = (
        dl_21
        + dl_32_c21
        + dl_13_c21_32
        + dl_31_c21_32_13
        + dl_23_c21_32_13_31
        + dl_12_c21_32_13_31_23
        - dl_31
        - dl_23_c31
        - dl_12_c31_23
        - dl_21_c31_23_12
        - dl_32_c31_23_12_21
        - dl_13_c31_23_12_21_32
    )

    # =====================================================================
    # BETA1
    # =====================================================================
    dl_13b_c21                 = TS(dl_13, shifts_beta, cB_21)
    dl_21b_c21_32              = TS(dl_21, shifts_beta, cB_21_32)
    dl_12b_c21_32_13           = TS(dl_12, shifts_beta, cB_21_32_13)
    dl_31b_c21_32_13_31        = TS(dl_31, shifts_beta, cB_21_32_13_31)
    dl_23b_c21_32_13_31_23     = TS(dl_23, shifts_beta, cB_21_32_13_31_23)

    dl_31b_c31                 = TS(dl_31, shifts_beta, cB_31)
    dl_23b_c31_23              = TS(dl_23, shifts_beta, cB_31_23)
    dl_32b_c31_23_12           = TS(dl_32, shifts_beta, cB_31_23_12)
    dl_13b_c31_23_12_21        = TS(dl_13, shifts_beta, cB_31_23_12_21)
    dl_21b_c31_23_12_21_32     = TS(dl_21, shifts_beta, cB_31_23_12_21_32)

    beta1 = (
        dl_32
        + dl_13b_c21
        + dl_21b_c21_32
        + dl_12b_c21_32_13
        + dl_31b_c21_32_13_31
        + dl_23b_c21_32_13_31_23
        - dl_12
        - dl_31b_c31
        - dl_23b_c31_23
        - dl_32b_c31_23_12
        - dl_13b_c31_23_12_21
        - dl_21b_c31_23_12_21_32
    )

    # =====================================================================
    # GAMMA1
    # =====================================================================
    dl_21g_c21                 = TS(dl_21, shifts_gamma, cG_21)
    dl_32g_c21_32              = TS(dl_32, shifts_gamma, cG_21_32)
    dl_23g_c21_32_13           = TS(dl_23, shifts_gamma, cG_21_32_13)
    dl_12g_c21_32_13_31        = TS(dl_12, shifts_gamma, cG_21_32_13_31)
    dl_31g_c21_32_13_31_23     = TS(dl_31, shifts_gamma, cG_21_32_13_31_23)

    dl_12g_c31                 = TS(dl_12, shifts_gamma, cG_31)
    dl_31g_c31_23              = TS(dl_31, shifts_gamma, cG_31_23)
    dl_13g_c31_23_12           = TS(dl_13, shifts_gamma, cG_31_23_12)
    dl_21g_c31_23_12_21        = TS(dl_21, shifts_gamma, cG_31_23_12_21)
    dl_32g_c31_23_12_21_32     = TS(dl_32, shifts_gamma, cG_31_23_12_21_32)

    gamma1 = (
        dl_13
        + dl_21g_c21
        + dl_32g_c21_32
        + dl_23g_c21_32_13
        + dl_12g_c21_32_13_31
        + dl_31g_c21_32_13_31_23
        - dl_23
        - dl_12g_c31
        - dl_31g_c31_23
        - dl_13g_c31_23_12
        - dl_21g_c31_23_12_21
        - dl_32g_c31_23_12_21_32
    )

    return alpha1, beta1, gamma1


def gen2_sagnac_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    shifts_cache=None,
    L_trunc=None,
    one_arm_cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
):
    """
    Gen-2 Sagnac-type variables (alpha1, beta1, gamma1) in WDM DOMAIN,
    with built-in construction of the WDM time grid and gen2 shifts.

    IMPORTANT:
    - Shifts are computed on times_p = times[0] + arange(wdm.Nt)*wdm.dT
    - One-arm responses are also evaluated on times_p (so shapes match).

    Parameters
    ----------
    times : ndarray
        Input time grid for the padded waveform.
    waveform_plus, waveform_cross : ndarray
        Time-domain GW polarizations sampled on ``times``.
    theta, phi : float
        Sky-location angles in radians.
    orbits : object
        Orbit provider used for one-arm response and delay calculations.
    Nf : int
        Number of WDM frequency bins.
    shifts_cache : dict or None
        Optional mutable dict used to memoize computed shift libraries.
        Keyed by ("sagnac_wdm", id(orbits), wdm.Nt, times[0], wdm.dT, Nf).
    L_trunc : int or None
        Optional lag truncation forwarded to one-arm response construction and
        subsequent variable-delay WDM shifts.
    one_arm_cache : dict or None
        Optional cache for shared one-arm WDM responses.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to
        ``wdm_time_shift_variable_batch`` (e.g. ``tl_tp_mode='interp'``,
        ``tl_tp_interp_kind='linear'|'cubic'``, ``tl_tp_interp_points=...``).
    q : int or None, optional
        WDM truncation parameter for one-arm response construction. When
        ``None``, the maximal allowed q is used (same behavior as one-arm
        response helpers).

    Returns
    -------
    tuple of ndarray
        ``(alpha1, beta1, gamma1)`` WDM coefficient arrays.
    """

    times = np.asarray(times, dtype=float)
    one_arm_cache = _resolve_one_arm_cache(one_arm_cache)
    shift_kwargs = _resolve_variable_shift_batch_kwargs(
        variable_shift_kwargs,
        default_batch_chunk=None,
    )
    wdm, link_w = precompute_one_arm_wdm_dT_links(
        times,
        orbits,
        theta,
        phi,
        waveform_plus,
        waveform_cross,
        Nf=Nf,
        L_trunc=L_trunc,
        cache=one_arm_cache,
        variable_shift_kwargs=shift_kwargs,
        use_jax=use_jax,
        q=q,
    )

    # Build WDM time grid internally
    times_p = float(times[0]) + np.arange(wdm.Nt, dtype=float) * float(wdm.dT)

    # -----------------------
    # built-in shift precompute (on WDM grid)
    # -----------------------
    cache_key = ("sagnac_wdm", id(orbits), int(wdm.Nt), float(times[0]), float(wdm.dT), int(Nf))
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_alpha, shifts_beta, shifts_gamma = shifts_cache[cache_key]
    else:
        shifts_alpha, shifts_beta, shifts_gamma = precompute_gen2_sagnac_shifts(times_p, orbits)
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_alpha, shifts_beta, shifts_gamma)

    w21 = link_w[(2, 1)]
    w32 = link_w[(3, 2)]
    w13 = link_w[(1, 3)]
    w31 = link_w[(3, 1)]
    w23 = link_w[(2, 3)]
    w12 = link_w[(1, 2)]

    alpha_jobs_template, beta_jobs_template, gamma_jobs_template = _GEN2_SAGNAC_JOB_TEMPLATES

    def build_jobs(prefix, shift_dict, jobs):
        return [(f"{prefix}:{name}", link_w[link], shift_dict[chain]) for name, link, chain in jobs]

    all_jobs = []
    all_jobs.extend(build_jobs("A", shifts_alpha, alpha_jobs_template))
    all_jobs.extend(build_jobs("B", shifts_beta, beta_jobs_template))
    all_jobs.extend(build_jobs("G", shifts_gamma, gamma_jobs_template))

    shift_kwargs = _finalize_variable_shift_batch_kwargs(shift_kwargs, len(all_jobs))

    all_shifted = wdm_time_shift_variable_batch(
        wdm,
        [(w, t_shift) for _, w, t_shift in all_jobs],
        Nf=Nf,
        L_trunc=L_trunc,
        use_jax=use_jax,
        **shift_kwargs,
    )
    shifted_map = {name: arr for (name, _, _), arr in zip(all_jobs, all_shifted)}

    w32_c21 = shifted_map["A:w32_c21"]
    w13_c21_32 = shifted_map["A:w13_c21_32"]
    w31_c21_32_13 = shifted_map["A:w31_c21_32_13"]
    w23_c21_32_13_31 = shifted_map["A:w23_c21_32_13_31"]
    w12_c21_32_13_31_23 = shifted_map["A:w12_c21_32_13_31_23"]
    w23_c31 = shifted_map["A:w23_c31"]
    w12_c31_23 = shifted_map["A:w12_c31_23"]
    w21_c31_23_12 = shifted_map["A:w21_c31_23_12"]
    w32_c31_23_12_21 = shifted_map["A:w32_c31_23_12_21"]
    w13_c31_23_12_21_32 = shifted_map["A:w13_c31_23_12_21_32"]

    alpha1 = (
        w21
        + w32_c21
        + w13_c21_32
        + w31_c21_32_13
        + w23_c21_32_13_31
        + w12_c21_32_13_31_23
        - w31
        - w23_c31
        - w12_c31_23
        - w21_c31_23_12
        - w32_c31_23_12_21
        - w13_c31_23_12_21_32
    )

    w13b_c21 = shifted_map["B:w13b_c21"]
    w21b_c21_32 = shifted_map["B:w21b_c21_32"]
    w12b_c21_32_13 = shifted_map["B:w12b_c21_32_13"]
    w31b_c21_32_13_31 = shifted_map["B:w31b_c21_32_13_31"]
    w23b_c21_32_13_31_23 = shifted_map["B:w23b_c21_32_13_31_23"]
    w31b_c31 = shifted_map["B:w31b_c31"]
    w23b_c31_23 = shifted_map["B:w23b_c31_23"]
    w32b_c31_23_12 = shifted_map["B:w32b_c31_23_12"]
    w13b_c31_23_12_21 = shifted_map["B:w13b_c31_23_12_21"]
    w21b_c31_23_12_21_32 = shifted_map["B:w21b_c31_23_12_21_32"]

    beta1 = (
        w32
        + w13b_c21
        + w21b_c21_32
        + w12b_c21_32_13
        + w31b_c21_32_13_31
        + w23b_c21_32_13_31_23
        - w12
        - w31b_c31
        - w23b_c31_23
        - w32b_c31_23_12
        - w13b_c31_23_12_21
        - w21b_c31_23_12_21_32
    )

    w21g_c21 = shifted_map["G:w21g_c21"]
    w32g_c21_32 = shifted_map["G:w32g_c21_32"]
    w23g_c21_32_13 = shifted_map["G:w23g_c21_32_13"]
    w12g_c21_32_13_31 = shifted_map["G:w12g_c21_32_13_31"]
    w31g_c21_32_13_31_23 = shifted_map["G:w31g_c21_32_13_31_23"]
    w12g_c31 = shifted_map["G:w12g_c31"]
    w31g_c31_23 = shifted_map["G:w31g_c31_23"]
    w13g_c31_23_12 = shifted_map["G:w13g_c31_23_12"]
    w21g_c31_23_12_21 = shifted_map["G:w21g_c31_23_12_21"]
    w32g_c31_23_12_21_32 = shifted_map["G:w32g_c31_23_12_21_32"]

    gamma1 = (
        w13
        + w21g_c21
        + w32g_c21_32
        + w23g_c21_32_13
        + w12g_c21_32_13_31
        + w31g_c21_32_13_31_23
        - w23
        - w12g_c31
        - w31g_c31_23
        - w13g_c31_23_12
        - w21g_c31_23_12_21
        - w32g_c31_23_12_21_32
    )

    return alpha1, beta1, gamma1

def gen2_michelson_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    shifts_cache=None,
    L_trunc=None,
    one_arm_cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
    tdi_assembly_mode="batch_map",
    accumulator_chunk=1,
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
):
    """
    Gen-2 Michelson-type variables (X1, Y1, Z1) in the WDM domain,
    with built-in construction of the WDM time grid and gen2 shifts.

    IMPORTANT:
    - Shifts are computed on times_p = times[0] + arange(wdm.Nt)*wdm.dT
    - One-arm responses are also evaluated on times_p (so shapes match).

    Parameters
    ----------
    times : ndarray
        Input time grid for the padded waveform.
    waveform_plus, waveform_cross : ndarray
        Time-domain GW polarizations sampled on ``times``.
    theta, phi : float
        Sky-location angles in radians.
    orbits : object
        Orbit provider used for one-arm response and delay calculations.
    Nf : int
        Number of WDM frequency bins.
    shifts_cache : dict or None
        Optional mutable dict used to memoize computed shift libraries.
        Keyed by ("michelson_wdm", id(orbits), wdm.Nt, times[0], wdm.dT, Nf).
    L_trunc : int or None
        Optional lag truncation forwarded to one-arm response construction and
        subsequent variable-delay WDM shifts.
    one_arm_cache : dict or None
        Optional cache for shared one-arm WDM responses.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to
        ``wdm_time_shift_variable_batch`` (e.g. ``tl_tp_mode='interp'``,
        ``tl_tp_interp_kind='linear'|'cubic'``, ``tl_tp_interp_points=...``).
    q : int or None, optional
        WDM truncation parameter for one-arm response construction. When
        ``None``, the maximal allowed q is used (same behavior as one-arm
        response helpers).

    Returns
    -------
    tuple of ndarray
        ``(X1, Y1, Z1)`` WDM coefficient arrays.
    """
    times = np.asarray(times, dtype=float)
    precomp = precompute_gen2_michelson_wdm_inputs(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        Nf,
        shifts_cache=shifts_cache,
        L_trunc=L_trunc,
        one_arm_cache=one_arm_cache,
        variable_shift_kwargs=variable_shift_kwargs,
        use_jax=use_jax,
        q=q,
    )
    return gen2_michelson_wdm_from_precomputed(
        precomp,
        variable_shift_kwargs=variable_shift_kwargs,
        L_trunc=L_trunc,
        use_jax=use_jax,
        Nf=Nf,
        tdi_assembly_mode=tdi_assembly_mode,
        accumulator_chunk=accumulator_chunk,
        profile_shift_stages=profile_shift_stages,
        profile_label=profile_label,
        return_shift_profile=return_shift_profile,
    )

def gen2_michelson_td(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    order=7,
    mode="zero",
    num_int_points=1000,
    shifts_cache=None,
):
    """
    Gen-2 Michelson-type variables (X1, Y1, Z1) in TIME DOMAIN
    with built-in precomputation of gen2 shifts.

    Parameters
    ----------
    shifts_cache : dict or None
        Optional mutable dict used to memoize computed shift libraries.
        Keyed by ("michelson", id(orbits), len(times), times[0], times[1]-times[0]).
    """

    times = np.asarray(times, dtype=float)
    dt = times[1] - times[0]

    cache_key = ("michelson", id(orbits), len(times), float(times[0]), float(dt))
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_X, shifts_Y, shifts_Z = shifts_cache[cache_key]
    else:
        shifts_X, shifts_Y, shifts_Z = precompute_gen2_michelson_shifts(times, orbits)
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_X, shifts_Y, shifts_Z)

    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)

    def TS(x, shift_dict, chain):
        return interp_time_shift(x, dt, shift_dict[chain], order=order, mode=mode)

    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    cX_21 = ((2, 1),)
    cX_21_12 = ((2, 1), (1, 2))
    cX_21_12_31 = ((2, 1), (1, 2), (3, 1))
    cX_21_12_31_13 = ((2, 1), (1, 2), (3, 1), (1, 3))
    cX_21_12_31_13_31 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1))
    cX_21_12_31_13_31_13 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3))
    cX_21_12_31_13_31_13_21 = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3), (2, 1))

    cX_31 = ((3, 1),)
    cX_31_13 = ((3, 1), (1, 3))
    cX_31_13_21 = ((3, 1), (1, 3), (2, 1))
    cX_31_13_21_12 = ((3, 1), (1, 3), (2, 1), (1, 2))
    cX_31_13_21_12_21 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1))
    cX_31_13_21_12_21_12 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2))
    cX_31_13_21_12_21_12_31 = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2), (3, 1))

    cY_21 = rot_chain(cX_21, 1);  cZ_21 = rot_chain(cX_21, 2)
    cY_21_12 = rot_chain(cX_21_12, 1);  cZ_21_12 = rot_chain(cX_21_12, 2)
    cY_21_12_31 = rot_chain(cX_21_12_31, 1);  cZ_21_12_31 = rot_chain(cX_21_12_31, 2)
    cY_21_12_31_13 = rot_chain(cX_21_12_31_13, 1);  cZ_21_12_31_13 = rot_chain(cX_21_12_31_13, 2)
    cY_21_12_31_13_31 = rot_chain(cX_21_12_31_13_31, 1);  cZ_21_12_31_13_31 = rot_chain(cX_21_12_31_13_31, 2)
    cY_21_12_31_13_31_13 = rot_chain(cX_21_12_31_13_31_13, 1);  cZ_21_12_31_13_31_13 = rot_chain(cX_21_12_31_13_31_13, 2)
    cY_21_12_31_13_31_13_21 = rot_chain(cX_21_12_31_13_31_13_21, 1);  cZ_21_12_31_13_31_13_21 = rot_chain(cX_21_12_31_13_31_13_21, 2)

    cY_31 = rot_chain(cX_31, 1);  cZ_31 = rot_chain(cX_31, 2)
    cY_31_13 = rot_chain(cX_31_13, 1);  cZ_31_13 = rot_chain(cX_31_13, 2)
    cY_31_13_21 = rot_chain(cX_31_13_21, 1);  cZ_31_13_21 = rot_chain(cX_31_13_21, 2)
    cY_31_13_21_12 = rot_chain(cX_31_13_21_12, 1);  cZ_31_13_21_12 = rot_chain(cX_31_13_21_12, 2)
    cY_31_13_21_12_21 = rot_chain(cX_31_13_21_12_21, 1);  cZ_31_13_21_12_21 = rot_chain(cX_31_13_21_12_21, 2)
    cY_31_13_21_12_21_12 = rot_chain(cX_31_13_21_12_21_12, 1);  cZ_31_13_21_12_21_12 = rot_chain(cX_31_13_21_12_21_12, 2)
    cY_31_13_21_12_21_12_31 = rot_chain(cX_31_13_21_12_21_12_31, 1);  cZ_31_13_21_12_21_12_31 = rot_chain(cX_31_13_21_12_21_12_31, 2)

    dl_12_cX_21 = TS(dl_12, shifts_X, cX_21)
    dl_31_cX_21_12 = TS(dl_31, shifts_X, cX_21_12)
    dl_13_cX_21_12_31 = TS(dl_13, shifts_X, cX_21_12_31)
    dl_31_cX_21_12_31_13 = TS(dl_31, shifts_X, cX_21_12_31_13)
    dl_13_cX_21_12_31_13_31 = TS(dl_13, shifts_X, cX_21_12_31_13_31)
    dl_21_cX_21_12_31_13_31_13 = TS(dl_21, shifts_X, cX_21_12_31_13_31_13)
    dl_12_cX_21_12_31_13_31_13_21 = TS(dl_12, shifts_X, cX_21_12_31_13_31_13_21)

    dl_13_cX_31 = TS(dl_13, shifts_X, cX_31)
    dl_21_cX_31_13 = TS(dl_21, shifts_X, cX_31_13)
    dl_12_cX_31_13_21 = TS(dl_12, shifts_X, cX_31_13_21)
    dl_21_cX_31_13_21_12 = TS(dl_21, shifts_X, cX_31_13_21_12)
    dl_12_cX_31_13_21_12_21 = TS(dl_12, shifts_X, cX_31_13_21_12_21)
    dl_31_cX_31_13_21_12_21_12 = TS(dl_31, shifts_X, cX_31_13_21_12_21_12)
    dl_13_cX_31_13_21_12_21_12_31 = TS(dl_13, shifts_X, cX_31_13_21_12_21_12_31)

    X1 = (
        dl_21
        + dl_12_cX_21
        + dl_31_cX_21_12
        + dl_13_cX_21_12_31
        + dl_31_cX_21_12_31_13
        + dl_13_cX_21_12_31_13_31
        + dl_21_cX_21_12_31_13_31_13
        + dl_12_cX_21_12_31_13_31_13_21
        - dl_31
        - dl_13_cX_31
        - dl_21_cX_31_13
        - dl_12_cX_31_13_21
        - dl_21_cX_31_13_21_12
        - dl_12_cX_31_13_21_12_21
        - dl_31_cX_31_13_21_12_21_12
        - dl_13_cX_31_13_21_12_21_12_31
    )

    DL = {(2,1): dl_21, (3,2): dl_32, (1,3): dl_13, (3,1): dl_31, (2,3): dl_23, (1,2): dl_12}

    def eval_from_template(shift_dict, c21, c21_12, c21_12_31, c21_12_31_13, c21_12_31_13_31, c21_12_31_13_31_13, c21_12_31_13_31_13_21,
                           c31, c31_13, c31_13_21, c31_13_21_12, c31_13_21_12_21, c31_13_21_12_21_12, c31_13_21_12_21_12_31, k):

        def R(link):
            return rot_link(link, k)

        y231  = DL[R((2,1))]
        y13p2 = DL[R((1,2))]
        y32p1 = DL[R((3,1))]
        y123  = DL[R((1,3))]

        t1 = TS(y13p2, shift_dict, c21)
        t2 = TS(y32p1, shift_dict, c21_12)
        t3 = TS(y123,  shift_dict, c21_12_31)
        t4 = TS(y32p1, shift_dict, c21_12_31_13)
        t5 = TS(y123,  shift_dict, c21_12_31_13_31)
        t6 = TS(y231,  shift_dict, c21_12_31_13_31_13)
        t7 = TS(y13p2, shift_dict, c21_12_31_13_31_13_21)

        n1 = TS(y123,  shift_dict, c31)
        n2 = TS(y231,  shift_dict, c31_13)
        n3 = TS(y13p2, shift_dict, c31_13_21)
        n4 = TS(y231,  shift_dict, c31_13_21_12)
        n5 = TS(y13p2, shift_dict, c31_13_21_12_21)
        n6 = TS(y32p1, shift_dict, c31_13_21_12_21_12)
        n7 = TS(y123,  shift_dict, c31_13_21_12_21_12_31)

        return (y231 + t1+t2+t3+t4+t5+t6+t7) - (y32p1 + n1+n2+n3+n4+n5+n6+n7)

    Y1 = eval_from_template(
        shifts_Y,
        cY_21, cY_21_12, cY_21_12_31, cY_21_12_31_13, cY_21_12_31_13_31, cY_21_12_31_13_31_13, cY_21_12_31_13_31_13_21,
        cY_31, cY_31_13, cY_31_13_21, cY_31_13_21_12, cY_31_13_21_12_21, cY_31_13_21_12_21_12, cY_31_13_21_12_21_12_31,
        k=1
    )

    Z1 = eval_from_template(
        shifts_Z,
        cZ_21, cZ_21_12, cZ_21_12_31, cZ_21_12_31_13, cZ_21_12_31_13_31, cZ_21_12_31_13_31_13, cZ_21_12_31_13_31_13_21,
        cZ_31, cZ_31_13, cZ_31_13_21, cZ_31_13_21_12, cZ_31_13_21_12_21, cZ_31_13_21_12_21_12, cZ_31_13_21_12_21_12_31,
        k=2
    )

    return X1, Y1, Z1