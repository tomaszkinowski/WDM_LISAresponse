"""Time-domain and WDM Time-Delay Interferometry response assembly.

The prepared Gen-2 Michelson plan is the maintained production WDM path.
Direct batch-map and accumulator implementations are retained as numerical
reference routes for staged validation.
"""
import time
import numpy as np
from WDM_LISAresponse.code.lisa_response.response_with_plan import (
    one_arm_response_td,
    one_arm_response_td_chunked,
    precompute_one_arm_wdm_dT_links,
    precompute_one_arm_wdm_dT_links_from_context,
    apply_one_arm_wdm_runtime_from_context,
    _build_wdm_dT_context_from_wdm,
    _build_wdm_dT_grid,
    _split_variable_shift_kwargs,
)
from WDM_LISAresponse.code.lisa_response.one_arm_plan import (
    OneArmWDMPlan,
    OneArmWDMStaticPlan,
    _normalise_one_arm_execution_mode,
    _normalise_shift_backend,
    _validate_jax_request,
)
from WDM_LISAresponse.code.lisa_response.gen2_michelson_plan import Gen2MichelsonWDMPlan
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
    """Build cleaned kwargs for direct variable-shift batch calls.

    Historical backend names are normalized to ``production`` or
    ``reference``. Options belonging to removed job-block, vmap and separate
    interpolation implementations are rejected explicitly.
    """

    shift_kwargs = (
        {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)
    )
    _validate_jax_request(shift_kwargs.pop("use_jax", None))

    removed_options = {
        "assembly_vmap",
        "job_block_size",
        "jax_pad_last_chunk",
        "tl_tp_interp_backend",
    }
    present_removed = sorted(removed_options.intersection(shift_kwargs))
    if present_removed:
        raise TypeError(
            "Removed variable-shift options were supplied: "
            + ", ".join(present_removed)
        )

    shift_kwargs["assembly_backend"] = _normalise_shift_backend(
        shift_kwargs.get("assembly_backend", "production")
    )
    shift_kwargs.setdefault("assembly_precision", "complex64")
    shift_kwargs.setdefault("row_chunk_size", 128)
    shift_kwargs.setdefault("lag_block_size", 1)
    shift_kwargs.setdefault("tl_tp_mode", "exact")

    batch_chunk = shift_kwargs.pop("batch_chunk", default_batch_chunk)
    if batch_chunk is not None:
        shift_kwargs["batch_chunk"] = int(batch_chunk)
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




def precompute_gen2_michelson_wdm_static(
    times,
    orbits,
    *,
    Nf,
    L_trunc=None,
    q=None,
    d=4,
    A_frac=0.25,
    calc_m0=True,
    use_jax=None,
    variable_shift_kwargs=None,
    shifts_cache=None,
):
    """Build sky- and waveform-independent Gen-2 Michelson WDM state."""

    times = np.asarray(times, dtype=float)
    one_arm_shift_kwargs, tdi_batch_shift_kwargs = _split_variable_shift_kwargs(
        variable_shift_kwargs
    )

    t_grid0 = time.perf_counter()
    wdm_context = _build_wdm_dT_grid(
        times,
        Nf=Nf,
        d=d,
        A_frac=A_frac,
        q=q,
        calc_m0=calc_m0,
    )
    wdm_grid_build_s = time.perf_counter() - t_grid0
    wdm = wdm_context["wdm"]
    times_p = wdm_context["int_times_vals"]

    t_one_arm0 = time.perf_counter()
    one_arm_static_plan = OneArmWDMStaticPlan.build_from_context(
        wdm_context,
        orbits,
        L_trunc=L_trunc,
        variable_shift_kwargs=one_arm_shift_kwargs,
        use_jax=use_jax,
    )
    one_arm_static_build_s = time.perf_counter() - t_one_arm0

    cache_key = (
        "michelson_wdm",
        id(orbits),
        int(wdm.Nt),
        float(times[0]),
        float(wdm.dT),
        int(Nf),
    )
    t_tdi_delay0 = time.perf_counter()
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_X, shifts_Y, shifts_Z = shifts_cache[cache_key]
        tdi_delays_from_cache = True
    else:
        shifts_X, shifts_Y, shifts_Z = precompute_gen2_michelson_shifts(
            times_p,
            orbits,
        )
        tdi_delays_from_cache = False
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_X, shifts_Y, shifts_Z)
    tdi_delay_build_s = time.perf_counter() - t_tdi_delay0

    t_tdi_plan0 = time.perf_counter()
    tdi_plan = Gen2MichelsonWDMPlan.build(
        wdm=wdm,
        shifts_X=shifts_X,
        shifts_Y=shifts_Y,
        shifts_Z=shifts_Z,
        job_templates=_GEN2_MICHELSON_JOB_TEMPLATES,
        L_trunc=L_trunc,
        variable_shift_kwargs=tdi_batch_shift_kwargs,
        use_jax=use_jax,
    )
    tdi_plan_build_s = time.perf_counter() - t_tdi_plan0

    return {
        "kind": "gen2_michelson_wdm_static",
        "times": times,
        "orbits": orbits,
        "wdm": wdm,
        "wdm_grid_context": wdm_context,
        "times_p": times_p,
        "one_arm_static_plan": one_arm_static_plan,
        "tdi_plan": tdi_plan,
        "shifts_X": shifts_X,
        "shifts_Y": shifts_Y,
        "shifts_Z": shifts_Z,
        "job_templates": _GEN2_MICHELSON_JOB_TEMPLATES,
        "one_arm_shift_kwargs": one_arm_shift_kwargs,
        "tdi_batch_shift_kwargs": tdi_batch_shift_kwargs,
        "Nf": int(Nf),
        "L_trunc": None if L_trunc is None else int(L_trunc),
        "calc_m0": bool(calc_m0),
        "q": q,
        "d": int(d),
        "A_frac": float(A_frac),
        "use_jax": True,
        "wdm_grid_build_s": float(wdm_grid_build_s),
        "one_arm_static_build_s": float(one_arm_static_build_s),
        "one_arm_interpolation_table_memory_bytes": int(
            one_arm_static_plan.kernel_memory_bytes
        ),
        "tdi_delay_build_s": float(tdi_delay_build_s),
        "tdi_plan_build_s": float(tdi_plan_build_s),
        "tdi_plan_kernel_memory_bytes": int(tdi_plan.kernel_memory_bytes),
        "tdi_delays_from_cache": bool(tdi_delays_from_cache),
    }


def precompute_gen2_michelson_wdm_waveform_runtime(
    static_precomp,
    waveform_plus,
    waveform_cross,
    *,
    parallel_wdm_transforms=False,
):
    """Build only waveform WDM coefficients on a static response grid."""

    if static_precomp.get("kind") != "gen2_michelson_wdm_static":
        raise ValueError(
            "static_precomp must come from "
            "precompute_gen2_michelson_wdm_static()."
        )
    started = time.perf_counter()
    wave_context = _build_wdm_dT_context_from_wdm(
        static_precomp["times"],
        waveform_plus,
        waveform_cross,
        static_precomp["wdm"],
        m_offset=0.0,
        parallel_wdm_transforms=parallel_wdm_transforms,
    )
    return {
        "kind": "gen2_michelson_wdm_waveform_runtime",
        "static_precomp": static_precomp,
        "wdm": static_precomp["wdm"],
        "wave_context": wave_context,
        "parallel_wdm_transforms": bool(parallel_wdm_transforms),
        "waveform_context_build_s": float(time.perf_counter() - started),
    }


def gen2_michelson_wdm_from_static_waveform(
    static_precomp,
    waveform_precomp,
    theta,
    phi,
    *,
    return_profile=False,
):
    """Evaluate sky-dependent one-arm response and static prepared TDI."""

    if waveform_precomp.get("static_precomp") is not static_precomp:
        raise ValueError(
            "waveform_precomp was not built from this static_precomp object."
        )
    one_arm_static_plan = static_precomp["one_arm_static_plan"]
    wave_context = waveform_precomp["wave_context"]

    started = time.perf_counter()
    _, one_arm_result = apply_one_arm_wdm_runtime_from_context(
        wave_context,
        one_arm_static_plan,
        theta,
        phi,
        return_device=False,
        return_profile=return_profile,
    )
    if return_profile:
        link_w, one_arm_profile = one_arm_result
    else:
        link_w = one_arm_result

    tdi_started = time.perf_counter()
    if return_profile:
        outputs, tdi_profile = static_precomp["tdi_plan"].apply(
            link_w,
            return_profile=True,
        )
        return outputs, {
            "theta": float(theta),
            "phi": float(phi),
            "one_arm": one_arm_profile,
            "tdi": tdi_profile,
            "tdi_apply_s": float(time.perf_counter() - tdi_started),
            "total_runtime_s": float(time.perf_counter() - started),
        }

    outputs = static_precomp["tdi_plan"].apply(link_w)
    return outputs


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
    d=4,
    A_frac=0.25,
    one_arm_cache=None,
    shifts_cache=None,
    calc_m0=True,
    return_device_links=False,
    one_arm_execution_mode="materialized",
    one_arm_parallel_links_per_group=1,
    one_arm_parallel_max_workers=None,
    parallel_wdm_transforms=False,
):
    """Backward-compatible staged Gen-2 Michelson precompute wrapper.

    This wrapper builds the reusable A0 geometry/static state and then the A1
    waveform state. The A1 stage may use concurrent plus/cross WDM transforms
    and concurrent groups of one-arm link shifts.
    """
    geom_precomp = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=Nf,
        L_trunc=L_trunc,
        q=q,
        d=d,
        A_frac=A_frac,
        calc_m0=calc_m0,
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
        return_device_links=return_device_links,
        one_arm_execution_mode=one_arm_execution_mode,
        one_arm_parallel_links_per_group=(
            one_arm_parallel_links_per_group
        ),
        one_arm_parallel_max_workers=one_arm_parallel_max_workers,
        parallel_wdm_transforms=parallel_wdm_transforms,
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
    A_frac=0.25,
    calc_m0=True,
    use_jax=None,
    variable_shift_kwargs=None,
    shifts_cache=None,
):
    """A0 precompute: build geometry/static Gen-2 Michelson state.

    This stage is waveform-independent and reusable across many waveforms on
    the same time grid, sky location, orbit object, and WDM settings.
    """
    times = np.asarray(times, dtype=float)
    one_arm_shift_kwargs, tdi_batch_shift_kwargs = _split_variable_shift_kwargs(variable_shift_kwargs)

    # Build WDM/grid metadata directly.  This is the A0 grid stage and must
    # not project dummy waveforms or call ``wdm.dwt``.
    t_grid0 = time.perf_counter()
    wdm_context = _build_wdm_dT_grid(
        times,
        Nf=Nf,
        d=d,
        A_frac=A_frac,
        q=q,
        calc_m0=calc_m0,
    )
    wdm_grid_build_s = time.perf_counter() - t_grid0

    wdm = wdm_context["wdm"]
    times_p = wdm_context["int_times_vals"]

    # The one-arm geometry, transfer factors and variable-shift kernels depend
    # on the orbit, sky position and WDM grid, but not on the waveform.  Build
    # them in A0 so A1 only performs the two real waveform DWTs and applies the
    # prepared plan.
    t_one_arm0 = time.perf_counter()
    one_arm_plan = OneArmWDMPlan.build_from_context(
        wdm_context,
        orbits,
        theta,
        phi,
        L_trunc=L_trunc,
        variable_shift_kwargs=one_arm_shift_kwargs,
        use_jax=use_jax,
    )
    one_arm_plan_build_s = time.perf_counter() - t_one_arm0

    cache_key = ("michelson_wdm", id(orbits), int(wdm.Nt), float(times[0]), float(wdm.dT), int(Nf))
    t_tdi_delay0 = time.perf_counter()
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_X, shifts_Y, shifts_Z = shifts_cache[cache_key]
        tdi_delays_from_cache = True
    else:
        shifts_X, shifts_Y, shifts_Z = precompute_gen2_michelson_shifts(times_p, orbits)
        tdi_delays_from_cache = False
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_X, shifts_Y, shifts_Z)
    tdi_delay_build_s = time.perf_counter() - t_tdi_delay0

    # Build the complete 42-job WDM TDI operator in A0.  The recursive delay
    # fields and all Tl/Tp kernels are waveform independent.
    t_tdi_plan0 = time.perf_counter()
    tdi_plan = Gen2MichelsonWDMPlan.build(
        wdm=wdm,
        shifts_X=shifts_X,
        shifts_Y=shifts_Y,
        shifts_Z=shifts_Z,
        job_templates=_GEN2_MICHELSON_JOB_TEMPLATES,
        L_trunc=L_trunc,
        variable_shift_kwargs=tdi_batch_shift_kwargs,
        use_jax=use_jax,
    )
    tdi_plan_build_s = time.perf_counter() - t_tdi_plan0

    return {
        "kind": "gen2_michelson_wdm_geometry",
        "times": times,
        "theta": float(theta),
        "phi": float(phi),
        "orbits": orbits,
        "wdm": wdm,
        "wdm_grid_context": wdm_context,
        "times_p": times_p,
        "shifts_X": shifts_X,
        "shifts_Y": shifts_Y,
        "shifts_Z": shifts_Z,
        "job_templates": _GEN2_MICHELSON_JOB_TEMPLATES,
        "one_arm_plan": one_arm_plan,
        "tdi_plan": tdi_plan,
        "one_arm_shift_kwargs": one_arm_shift_kwargs,
        "tdi_batch_shift_kwargs": tdi_batch_shift_kwargs,
        "Nf": int(Nf),
        "L_trunc": None if L_trunc is None else int(L_trunc),
        "calc_m0": bool(calc_m0),
        "q": q,
        "d": int(d),
        "A_frac": float(A_frac),
        "use_jax": True,
        "wdm_grid_build_s": float(wdm_grid_build_s),
        "one_arm_plan_build_s": float(one_arm_plan_build_s),
        "tdi_delay_build_s": float(tdi_delay_build_s),
        "tdi_plan_build_s": float(tdi_plan_build_s),
        "tdi_plan_kernel_memory_bytes": int(tdi_plan.kernel_memory_bytes),
        "tdi_delays_from_cache": bool(tdi_delays_from_cache),
    }



def precompute_gen2_michelson_wdm_waveform(
    geom_precomp,
    waveform_plus,
    waveform_cross,
    *,
    variable_shift_kwargs=None,
    one_arm_cache=None,
    return_device_links=False,
    one_arm_execution_mode="materialized",
    one_arm_parallel_links_per_group=1,
    one_arm_parallel_max_workers=None,
    parallel_wdm_transforms=False,
):
    """A1 precompute: transform one waveform and apply six one-arm plans.

    The prepared TDI plan from A0 is retained for the optimized
    ``tdi_assembly_mode="prepared_plan"`` path. Runtime TDI overrides remain
    available for the legacy ``batch_map`` and ``accumulate`` reference paths.

    ``parallel_wdm_transforms=True`` evaluates the independent plus/cross DWTs
    concurrently. ``one_arm_execution_mode='materialized_parallel_links'``
    evaluates contiguous one-arm link groups concurrently with the unchanged
    production shift kernel.
    """
    resolved_one_arm_execution_mode = _normalise_one_arm_execution_mode(
        one_arm_execution_mode
    )

    parallel_links_per_group = int(
        one_arm_parallel_links_per_group
    )
    if parallel_links_per_group < 1:
        raise ValueError(
            "one_arm_parallel_links_per_group must be >= 1."
        )
    parallel_max_workers = (
        None
        if one_arm_parallel_max_workers is None
        else int(one_arm_parallel_max_workers)
    )
    if parallel_max_workers is not None and parallel_max_workers < 1:
        raise ValueError(
            "one_arm_parallel_max_workers must be >= 1 or None."
        )

    times = geom_precomp["times"]
    theta = geom_precomp["theta"]
    phi = geom_precomp["phi"]
    orbits = geom_precomp["orbits"]
    Nf = int(geom_precomp["Nf"])
    L_trunc = geom_precomp.get("L_trunc")
    d = int(geom_precomp.get("d", 4))
    A_frac = float(geom_precomp.get("A_frac", 0.25))
    use_jax = geom_precomp.get("use_jax")
    q = geom_precomp.get("q")
    calc_m0 = bool(geom_precomp.get("calc_m0", True))

    one_arm_shift_kwargs = dict(
        geom_precomp.get("one_arm_shift_kwargs", {})
    )
    prepared_tdi_shift_kwargs = dict(
        geom_precomp.get("tdi_batch_shift_kwargs", {})
    )
    runtime_tdi_shift_kwargs = dict(prepared_tdi_shift_kwargs)

    rebuild_one_arm_plan = False
    if variable_shift_kwargs is not None:
        one_arm_override, tdi_override = _split_variable_shift_kwargs(
            variable_shift_kwargs
        )

        updated_one_arm_kwargs = dict(one_arm_shift_kwargs)
        updated_one_arm_kwargs.update(one_arm_override)
        rebuild_one_arm_plan = (
            updated_one_arm_kwargs != one_arm_shift_kwargs
        )
        one_arm_shift_kwargs = updated_one_arm_kwargs

        runtime_tdi_shift_kwargs.update(tdi_override)

    runtime_tdi_differs_from_plan = (
        runtime_tdi_shift_kwargs != prepared_tdi_shift_kwargs
    )

    t_ctx0 = time.perf_counter()
    wave_context = _build_wdm_dT_context_from_wdm(
        times,
        waveform_plus,
        waveform_cross,
        geom_precomp["wdm"],
        m_offset=0.0,
        parallel_wdm_transforms=parallel_wdm_transforms,
    )
    waveform_context_build_s = time.perf_counter() - t_ctx0

    one_arm_plan = geom_precomp.get("one_arm_plan")
    if one_arm_plan is None or rebuild_one_arm_plan:
        one_arm_plan = OneArmWDMPlan.build_from_context(
            wave_context,
            orbits,
            theta,
            phi,
            L_trunc=L_trunc,
            variable_shift_kwargs=one_arm_shift_kwargs,
            use_jax=use_jax,
        )

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
        one_arm_plan=one_arm_plan,
        return_device=return_device_links,
        one_arm_execution_mode=resolved_one_arm_execution_mode,
        one_arm_parallel_links_per_group=parallel_links_per_group,
        one_arm_parallel_max_workers=parallel_max_workers,
    )
    six_link_compute_s = time.perf_counter() - t_links0

    tdi_plan = geom_precomp.get("tdi_plan")
    if tdi_plan is None:
        raise ValueError(
            "geom_precomp is missing the prepared Gen-2 Michelson TDI plan."
        )

    t_jobs0 = time.perf_counter()
    x_jobs, y_jobs, z_jobs, all_jobs = (
        _build_gen2_michelson_job_records(
            link_w,
            geom_precomp["shifts_X"],
            geom_precomp["shifts_Y"],
            geom_precomp["shifts_Z"],
        )
    )
    job_build_s = time.perf_counter() - t_jobs0

    return {
        "kind": "gen2_michelson_wdm_waveform",
        "geometry_precomp": geom_precomp,
        "wdm": wdm,
        "times_p": geom_precomp["times_p"],
        "link_w": link_w,
        "link_w_on_device": bool(return_device_links),
        "one_arm_execution_mode": (
            resolved_one_arm_execution_mode
            if return_device_links
            else "host_materialized"
        ),
        "one_arm_parallel_links_per_group": (
            parallel_links_per_group
            if return_device_links
            and resolved_one_arm_execution_mode
            == "materialized_parallel_links"
            else None
        ),
        "one_arm_parallel_max_workers": (
            parallel_max_workers
            if return_device_links
            and resolved_one_arm_execution_mode
            == "materialized_parallel_links"
            else None
        ),
        "parallel_wdm_transforms": bool(parallel_wdm_transforms),
        "one_arm_plan": one_arm_plan,
        "tdi_plan": tdi_plan,
        "shifts_X": geom_precomp["shifts_X"],
        "shifts_Y": geom_precomp["shifts_Y"],
        "shifts_Z": geom_precomp["shifts_Z"],
        "x_jobs": x_jobs,
        "y_jobs": y_jobs,
        "z_jobs": z_jobs,
        "all_jobs": all_jobs,
        "job_templates": geom_precomp["job_templates"],
        "one_arm_shift_kwargs": one_arm_shift_kwargs,
        "tdi_batch_shift_kwargs": prepared_tdi_shift_kwargs,
        "tdi_runtime_shift_kwargs": runtime_tdi_shift_kwargs,
        "tdi_runtime_differs_from_plan": bool(
            runtime_tdi_differs_from_plan
        ),
        "Nf": Nf,
        "L_trunc": L_trunc,
        "d": d,
        "A_frac": A_frac,
        "use_jax": True,
        "q": q,
        "calc_m0": calc_m0,
        "waveform_context_build_s": float(
            waveform_context_build_s
        ),
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
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
):
    """B stage: construct Gen-2 Michelson ``X, Y, Z``.

    ``prepared_plan`` is the production path: it applies the immutable
    42-job operator built in A0. ``batch_map`` and ``accumulate`` are retained
    as reference paths and rebuild delay kernels during this stage.
    """
    if "link_w" not in wave_precomp:
        raise ValueError(
            "wave_precomp is missing link_w; expected waveform-level "
            "precompute payload."
        )

    wdm = wave_precomp["wdm"]
    link_w = wave_precomp["link_w"]

    precomp_Nf = int(wave_precomp.get("Nf", wdm.Nf))
    if Nf is not None and int(Nf) != precomp_Nf:
        raise ValueError(
            f"Nf must match the prepared WDM grid; expected {precomp_Nf}, "
            f"got {Nf}."
        )

    precomp_L_trunc = wave_precomp.get("L_trunc", None)
    if L_trunc is not None:
        if precomp_L_trunc is None:
            raise ValueError(
                "L_trunc cannot be changed after preparing the staged "
                "Gen-2 response."
            )
        if int(L_trunc) != int(precomp_L_trunc):
            raise ValueError(
                "L_trunc must match the prepared shifts; expected "
                f"{precomp_L_trunc}, got {L_trunc}."
            )

    if tdi_assembly_mode == "prepared_plan":
        tdi_plan = wave_precomp.get("tdi_plan")
        if tdi_plan is None:
            geometry = wave_precomp.get("geometry_precomp", {})
            tdi_plan = geometry.get("tdi_plan")
        if tdi_plan is None:
            raise ValueError(
                "No prepared Gen-2 Michelson TDI plan was found in "
                "wave_precomp."
            )

        _validate_jax_request(use_jax)

        if wave_precomp.get("tdi_runtime_differs_from_plan", False):
            raise ValueError(
                "A1 contains runtime TDI overrides that differ from the "
                "prepared A0 plan. Use tdi_assembly_mode='batch_map' or "
                "'accumulate' for backend experiments, or rebuild A0 with "
                "those settings before using 'prepared_plan'."
            )

        if variable_shift_kwargs is not None:
            _, tdi_override = _split_variable_shift_kwargs(
                variable_shift_kwargs
            )
            stored_tdi_kwargs = dict(
                wave_precomp.get("tdi_batch_shift_kwargs", {})
            )
            updated_tdi_kwargs = dict(stored_tdi_kwargs)
            updated_tdi_kwargs.update(tdi_override)
            if updated_tdi_kwargs != stored_tdi_kwargs:
                raise ValueError(
                    "variable_shift_kwargs are part of the prepared TDI "
                    "plan; rebuild A0 to change them, or use a reference "
                    "assembly mode for comparisons."
                )

        if profile_shift_stages:
            assembled, shift_profile = tdi_plan.apply(
                link_w,
                return_profile=True,
            )
            shift_profile = dict(shift_profile)
            if profile_label is not None:
                shift_profile["label"] = profile_label
            if return_shift_profile:
                return assembled, shift_profile
            return assembled

        return tdi_plan.apply(link_w)

    # ------------------------------------------------------------------
    # Reference/testing paths.  These intentionally preserve the former
    # runtime-configurable behavior so backend-equivalence tests remain useful.
    # ------------------------------------------------------------------
    all_jobs = wave_precomp.get("all_jobs")
    if all_jobs is None:
        x_jobs, y_jobs, z_jobs, all_jobs = (
            _build_gen2_michelson_job_records(
                link_w,
                wave_precomp["shifts_X"],
                wave_precomp["shifts_Y"],
                wave_precomp["shifts_Z"],
            )
        )
    else:
        x_jobs = wave_precomp.get("x_jobs")
        y_jobs = wave_precomp.get("y_jobs")
        z_jobs = wave_precomp.get("z_jobs")

    _validate_jax_request(use_jax)

    batch_shift_kwargs = dict(
        wave_precomp.get(
            "tdi_runtime_shift_kwargs",
            wave_precomp.get("tdi_batch_shift_kwargs", {}),
        )
    )
    if variable_shift_kwargs is not None:
        _, tdi_override = _split_variable_shift_kwargs(
            variable_shift_kwargs
        )
        batch_shift_kwargs.update(tdi_override)

    batch_shift_kwargs = _finalize_variable_shift_batch_kwargs(
        batch_shift_kwargs,
        len(all_jobs),
    )
    batch_L_trunc = (
        L_trunc if L_trunc is not None else precomp_L_trunc
    )

    if tdi_assembly_mode == "batch_map":
        shifted_result = wdm_time_shift_variable_batch(
            wdm,
            [(w, t_shift) for _, w, t_shift in all_jobs],
            Nf=precomp_Nf,
            L_trunc=batch_L_trunc,
            profile_shift_stages=profile_shift_stages,
            profile_label=profile_label,
            **batch_shift_kwargs,
        )

        if profile_shift_stages:
            all_shifted, shift_profile = shifted_result
        else:
            all_shifted = shifted_result

        shifted_map = {
            name: arr
            for (name, _, _), arr in zip(all_jobs, all_shifted)
        }
        assembled = _assemble_gen2_michelson_from_shifted_map(
            link_w,
            shifted_map,
        )
        if profile_shift_stages and return_shift_profile:
            return assembled, shift_profile
        return assembled

    if tdi_assembly_mode == "accumulate":
        if x_jobs is None or y_jobs is None or z_jobs is None:
            x_jobs, y_jobs, z_jobs, all_jobs = (
                _build_gen2_michelson_job_records(
                    link_w,
                    wave_precomp["shifts_X"],
                    wave_precomp["shifts_Y"],
                    wave_precomp["shifts_Z"],
                )
            )

        return _gen2_michelson_wdm_accumulate_downstream(
            wdm,
            link_w,
            x_jobs,
            y_jobs,
            z_jobs,
            all_jobs,
            shifts_X=wave_precomp["shifts_X"],
            shifts_Y=wave_precomp["shifts_Y"],
            shifts_Z=wave_precomp["shifts_Z"],
            precomp_Nf=precomp_Nf,
            batch_L_trunc=batch_L_trunc,
            batch_shift_kwargs=batch_shift_kwargs,
            accumulator_chunk=accumulator_chunk,
            profile_shift_stages=profile_shift_stages,
            profile_label=profile_label,
            return_shift_profile=return_shift_profile,
        )

    raise ValueError(
        f"Unknown tdi_assembly_mode={tdi_assembly_mode!r}; expected "
        "'batch_map' or 'accumulate', or the new 'prepared_plan'."
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
    calc_m0=True,
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

def precompute_gen2_michelson_td_geometry(
    times,
    theta,
    phi,
    orbits,
    *,
    shifts_cache=None,
):
    """Precompute reusable Gen-2 Michelson TD geometry and shifts."""
    times = np.asarray(times, dtype=float)
    dt = float(times[1] - times[0])

    cache_key = ("michelson", id(orbits), len(times), float(times[0]), float(dt))
    if shifts_cache is not None and cache_key in shifts_cache:
        shifts_X, shifts_Y, shifts_Z = shifts_cache[cache_key]
    else:
        shifts_X, shifts_Y, shifts_Z = precompute_gen2_michelson_shifts(times, orbits)
        if shifts_cache is not None:
            shifts_cache[cache_key] = (shifts_X, shifts_Y, shifts_Z)

    return {
        "kind": "gen2_michelson_td_geometry",
        "times": times,
        "dt": dt,
        "theta": float(theta),
        "phi": float(phi),
        "orbits": orbits,
        "shifts_X": shifts_X,
        "shifts_Y": shifts_Y,
        "shifts_Z": shifts_Z,
    }


def precompute_one_arm_td_links(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    *,
    links=((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2)),
    num_int_points=10,
    one_arm_cache=None,
    chunk_size=None,
):
    """Precompute directed one-arm TD responses, optionally using chunking."""
    times = np.asarray(times, dtype=float)
    waveform_plus = np.asarray(waveform_plus)
    waveform_cross = np.asarray(waveform_cross)
    dt = float(times[1] - times[0])
    cache = {} if one_arm_cache is None else one_arm_cache

    out = {}
    response_fn = one_arm_response_td_chunked if chunk_size is not None else one_arm_response_td
    for link in links:
        s, r = link
        cache_key = (
            "td_one_arm",
            id(orbits),
            len(times),
            float(times[0]),
            float(dt),
            float(theta),
            float(phi),
            tuple(link),
            int(num_int_points),
            id(waveform_plus),
            id(waveform_cross),
            None if chunk_size is None else int(chunk_size),
        )
        if cache_key in cache:
            out[link] = cache[cache_key]
            continue

        if chunk_size is None:
            dl = response_fn(times, s, r, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
        else:
            dl = response_fn(
                times,
                s,
                r,
                orbits,
                theta,
                phi,
                waveform_plus,
                waveform_cross,
                num_int_points=num_int_points,
                chunk_size=chunk_size,
            )
        cache[cache_key] = dl
        out[link] = dl

    return out


def precompute_gen2_michelson_td_waveform(
    geom_precomp,
    waveform_plus,
    waveform_cross,
    *,
    num_int_points=10,
    one_arm_cache=None,
    chunk_size=None,
):
    """Precompute waveform-specific TD one-arm links for Gen-2 Michelson."""
    links = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))
    link_precomp = precompute_one_arm_td_links(
        geom_precomp["times"],
        waveform_plus,
        waveform_cross,
        geom_precomp["theta"],
        geom_precomp["phi"],
        geom_precomp["orbits"],
        links=links,
        num_int_points=num_int_points,
        one_arm_cache=one_arm_cache,
        chunk_size=chunk_size,
    )

    return {
        "kind": "gen2_michelson_td_waveform",
        "geometry_precomp": geom_precomp,
        "times": geom_precomp["times"],
        "dt": geom_precomp["dt"],
        "theta": geom_precomp["theta"],
        "phi": geom_precomp["phi"],
        "orbits": geom_precomp["orbits"],
        "num_int_points": int(num_int_points),
        "chunk_size": chunk_size,
        "dl_21": link_precomp[(2, 1)],
        "dl_32": link_precomp[(3, 2)],
        "dl_13": link_precomp[(1, 3)],
        "dl_31": link_precomp[(3, 1)],
        "dl_23": link_precomp[(2, 3)],
        "dl_12": link_precomp[(1, 2)],
    }


def gen2_michelson_td_from_precomputed(precomp, *, order=7, mode="zero"):
    """Assemble Gen-2 Michelson TD variables from staged precomputed payload."""
    times = np.asarray(precomp["times"], dtype=float)
    dt = float(precomp["dt"])
    geom_precomp = precomp["geometry_precomp"]
    shifts_X = geom_precomp["shifts_X"]
    shifts_Y = geom_precomp["shifts_Y"]
    shifts_Z = geom_precomp["shifts_Z"]

    dl_21 = precomp["dl_21"]
    dl_32 = precomp["dl_32"]
    dl_13 = precomp["dl_13"]
    dl_31 = precomp["dl_31"]
    dl_23 = precomp["dl_23"]
    dl_12 = precomp["dl_12"]

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

    X1 = (
        dl_21
        + TS(dl_12, shifts_X, cX_21)
        + TS(dl_31, shifts_X, cX_21_12)
        + TS(dl_13, shifts_X, cX_21_12_31)
        + TS(dl_31, shifts_X, cX_21_12_31_13)
        + TS(dl_13, shifts_X, cX_21_12_31_13_31)
        + TS(dl_21, shifts_X, cX_21_12_31_13_31_13)
        + TS(dl_12, shifts_X, cX_21_12_31_13_31_13_21)
        - dl_31
        - TS(dl_13, shifts_X, cX_31)
        - TS(dl_21, shifts_X, cX_31_13)
        - TS(dl_12, shifts_X, cX_31_13_21)
        - TS(dl_21, shifts_X, cX_31_13_21_12)
        - TS(dl_12, shifts_X, cX_31_13_21_12_21)
        - TS(dl_31, shifts_X, cX_31_13_21_12_21_12)
        - TS(dl_13, shifts_X, cX_31_13_21_12_21_12_31)
    )

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
    d=4,
    A_frac=0.25,
    calc_m0=True,
    tdi_assembly_mode="prepared_plan",
    accumulator_chunk=1,
    profile_shift_stages=False,
    profile_label=None,
    return_shift_profile=False,
    return_device_links=False,
    one_arm_execution_mode="materialized",
    one_arm_parallel_links_per_group=1,
    one_arm_parallel_max_workers=None,
    parallel_wdm_transforms=False,
):
    """Construct Gen-2 Michelson channels with runtime sky assembly.

    The direct API now builds a sky-independent static object, transforms the
    waveform, and supplies ``theta``/``phi`` only to the runtime response
    stage. Legacy scheduling parameters are accepted for compatibility but the
    runtime static-plan path currently uses one materialized six-link batch.
    """

    requested_backend = (
        "production"
        if variable_shift_kwargs is None
        else str(variable_shift_kwargs.get("assembly_backend", "production")).lower()
    )
    requested_tl_tp_mode = (
        "exact"
        if variable_shift_kwargs is None
        else str(variable_shift_kwargs.get("tl_tp_mode", "exact")).lower()
    )
    use_legacy_fixed_plan = (
        requested_backend in {
            "reference", "legacy", "row", "vmap", "chunked", "lagfirst_row"
        }
        or requested_tl_tp_mode == "exact"
    )

    if tdi_assembly_mode != "prepared_plan" or use_legacy_fixed_plan:
        # Preserve the existing configurable reference routes.
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
            d=d,
            A_frac=A_frac,
            calc_m0=calc_m0,
            return_device_links=return_device_links,
            one_arm_execution_mode=one_arm_execution_mode,
            one_arm_parallel_links_per_group=one_arm_parallel_links_per_group,
            one_arm_parallel_max_workers=one_arm_parallel_max_workers,
            parallel_wdm_transforms=parallel_wdm_transforms,
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

    static = precompute_gen2_michelson_wdm_static(
        times,
        orbits,
        Nf=Nf,
        L_trunc=L_trunc,
        q=q,
        d=d,
        A_frac=A_frac,
        calc_m0=calc_m0,
        use_jax=use_jax,
        variable_shift_kwargs=variable_shift_kwargs,
        shifts_cache=shifts_cache,
    )
    waveform = precompute_gen2_michelson_wdm_waveform_runtime(
        static,
        waveform_plus,
        waveform_cross,
        parallel_wdm_transforms=parallel_wdm_transforms,
    )
    return gen2_michelson_wdm_from_static_waveform(
        static,
        waveform,
        theta,
        phi,
        return_profile=bool(profile_shift_stages and return_shift_profile),
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
    chunk_size=None,
    one_arm_cache=None,
    return_precomp=False,
):
    """Backward-compatible wrapper for staged Gen-2 Michelson TD assembly."""

    geom = precompute_gen2_michelson_td_geometry(
        times,
        theta,
        phi,
        orbits,
        shifts_cache=shifts_cache,
    )
    precomp = precompute_gen2_michelson_td_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        num_int_points=num_int_points,
        one_arm_cache=one_arm_cache,
        chunk_size=chunk_size,
    )
    result = gen2_michelson_td_from_precomputed(precomp, order=order, mode=mode)
    if return_precomp:
        return result, precomp
    return result