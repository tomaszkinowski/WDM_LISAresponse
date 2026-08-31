"""
Utility functions for WDM_LISAresponse.

Everything here is support machinery rather than physics: WDM tiling
arithmetic, signal comparison, argument resolution for the response API,
generic interpolation on a uniform grid, and the memory budgeting that decides
how the response blocks its work. The response model itself - the link
geometry, the sky basis and the one-arm kernel - lives in
``lisa_response.response_sub``.
"""
import jax
import jax.numpy as jnp
import numpy as np


def wdm_tail_padding_info(n_samples, Nf):
    """Compute right-tail padding needed to satisfy WDM tiling constraints."""
    n_samples_i = int(n_samples)
    Nf_i = int(Nf)

    if n_samples_i < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples_i}.")
    if Nf_i < 1:
        raise ValueError(f"Nf must be >= 1, got {Nf_i}.")

    import WDM

    block = 2 * Nf_i
    target_length = int(WDM.utils.next_multiple(n_samples_i, block))
    pad_right = int(target_length - n_samples_i)

    return {
        "input_length": n_samples_i,
        "target_length": target_length,
        "pad_right": pad_right,
        "Nf": Nf_i,
        "Nt": int(target_length // Nf_i),
    }


def next_multiple(value, base):
    """Compatibility wrapper around ``WDM.utils.next_multiple``."""
    value_i = int(value)
    base_i = int(base)
    if value_i < 0:
        raise ValueError(f"value must be non-negative, got {value_i}.")
    if base_i <= 0:
        raise ValueError(f"base must be positive, got {base_i}.")
    import WDM
    return int(WDM.utils.next_multiple(value_i, base_i))


def compute_wdm_cut_info(
    original_length,
    Nf,
    dt,
    *,
    alpha=0.0,
    tukey_factor=0.0,
    window_pad_left=None,
    window_pad_right=None,
    include_wdm_tail_padding=True,
    include_taper_margin=False,
    wdm_dT=None,
    wdm_Nt=None,
    drop_m0=True,
    drop_freq_edges = False,
):
    """Build WDM-domain cut metadata for mismatch calculations."""
    original_length_i = int(original_length)
    Nf_i = int(Nf)
    window_pad_left_i = None if window_pad_left is None else int(window_pad_left)
    window_pad_right_i = None if window_pad_right is None else int(window_pad_right)
    dt_f = float(dt)
    alpha_f = float(alpha)
    tukey_factor_f = float(tukey_factor)

    if original_length_i < 1:
        raise ValueError(f"original_length must be >= 1, got {original_length_i}.")
    if Nf_i < 1:
        raise ValueError(f"Nf must be >= 1, got {Nf_i}.")
    if dt_f <= 0.0:
        raise ValueError(f"dt must be positive, got {dt_f}.")
    if not (0.0 <= alpha_f <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha_f}.")
    if tukey_factor_f < 0.0:
        raise ValueError(f"tukey_factor must be non-negative, got {tukey_factor_f}.")

    inferred_window_pad = int(np.ceil(tukey_factor_f * alpha_f * 0.5 * original_length_i))
    if window_pad_left_i is None:
        window_pad_left_i = inferred_window_pad
    if window_pad_right_i is None:
        window_pad_right_i = inferred_window_pad

    if window_pad_left_i < 0 or window_pad_right_i < 0:
        raise ValueError("window_pad_left/window_pad_right must be non-negative.")

    taper_margin_samples = int(np.ceil(0.5 * alpha_f * original_length_i)) if include_taper_margin else 0
    total_left_samples = int(taper_margin_samples + window_pad_left_i)
    total_right_samples_no_tail = int(taper_margin_samples + window_pad_right_i)

    post_window_length = int(original_length_i + window_pad_left_i + window_pad_right_i)
    tail_info = wdm_tail_padding_info(post_window_length, Nf_i)
    tail_pad_right = int(tail_info["pad_right"])
    total_right_samples = int(
        total_right_samples_no_tail + (tail_pad_right if include_wdm_tail_padding else 0)
    )

    wdm_dT_f = float(Nf_i * dt_f) if wdm_dT is None else float(wdm_dT)
    if wdm_dT_f <= 0.0:
        raise ValueError(f"wdm_dT must be positive, got {wdm_dT_f}.")

    left_cut = int(np.ceil(total_left_samples * dt_f / wdm_dT_f))
    right_cut = int(np.ceil(total_right_samples * dt_f / wdm_dT_f))

    Nt = int(tail_info["Nt"]) if wdm_Nt is None else int(wdm_Nt)
    if Nt < 1:
        raise ValueError(f"wdm_Nt must be >= 1, got {Nt}.")

    t_start = max(0, left_cut)
    t_stop = Nt - max(0, right_cut)
    if t_stop < t_start:
        t_stop = t_start

    if drop_freq_edges:
        freq_slice = slice(1, -1)
    elif drop_m0:
        freq_slice = slice(1, None)
    else:
        freq_slice = slice(None)

    return {
        "original_length": original_length_i,
        "Nf": Nf_i,
        "dt": dt_f,
        "wdm_dT": wdm_dT_f,
        "wdm_Nt": Nt,
        "alpha": alpha_f,
        "tukey_factor": tukey_factor_f,
        "window_pad_left": window_pad_left_i,
        "window_pad_right": window_pad_right_i,
        "window_padding_inferred_per_side": inferred_window_pad,
        "taper_margin_samples_per_side": taper_margin_samples,
        "tail_pad_right": tail_pad_right,
        "left_samples_to_cut": total_left_samples,
        "right_samples_to_cut": total_right_samples,
        "left_cut": left_cut,
        "right_cut": right_cut,
        "time_slice": slice(t_start, t_stop),
        "freq_slice": freq_slice,
        "trim_m0": bool(drop_m0),
    }


def apply_wdm_cut(coeffs, cut_info):
    """Apply cut slices returned by ``compute_wdm_cut_info`` to ``(Nt, Nf)`` arrays."""
    arr = np.asarray(coeffs)
    if arr.ndim != 2:
        raise ValueError(f"coeffs must be a 2D array, got shape {arr.shape}.")
    return arr[cut_info["time_slice"], cut_info["freq_slice"]]

# Add your utility functions here

def compare_response_signals(h_ref, h_test):
    """Compute overlap, mismatch, and residual diagnostics between responses."""
    h_ref  = np.asarray(h_ref).reshape(-1)
    h_test = np.asarray(h_test).reshape(-1)

    # L2 norms
    res = h_ref - h_test
    res_L2 = np.linalg.norm(res)
    ref_L2 = np.linalg.norm(h_ref)
    rel_L2 = res_L2 / ref_L2 if ref_L2 != 0.0 else np.nan

    # Overlap / mismatch (unweighted)
    num = np.vdot(h_ref, h_test)
    den = np.linalg.norm(h_ref) * np.linalg.norm(h_test)

    if den == 0.0:
        overlap = np.nan
        mismatch = np.nan
    else:
        overlap = np.abs(num) / den
        overlap = float(np.clip(overlap, 0.0, 1.0))
        mismatch = 1.0 - overlap

    return {
        "res_L2": res_L2,
        "ref_L2": ref_L2,
        "rel_L2": rel_L2,
        "overlap": overlap,
        "mismatch": mismatch,
    }

def wdm_mismatch(w1, w2, return_raw_overlap=False):
    """
    Numerically stable phase-maximized WDM mismatch.

    This computes the standard mismatch

        mismatch = 1 - |<w1|w2>| / sqrt(<w1|w1><w2|w2>)

    but evaluates it through the mathematically equivalent normalized
    residual

        mismatch = 0.5 * ||w1_hat - exp(-i phi) w2_hat||^2,

    where phi = arg(<w1_hat|w2_hat>).

    This avoids artificial negative mismatches or NaNs when the direct
    overlap is extremely close to one.

    Parameters
    ----------
    w1, w2 : ndarray
        WDM coefficient arrays.
    return_raw_overlap : bool
        If True, also return the direct raw overlap computed from the inner
        product. This is useful for diagnostics only.

    Returns
    -------
    overlap : float
        overlap = 1 - mismatch, computed from the stable mismatch.
    mismatch : float
        Standard phase-maximized mismatch.
    raw_overlap : float, optional
        Direct inner-product overlap, returned only if return_raw_overlap=True.
    """
    w1 = np.asarray(w1, dtype=np.complex128).ravel()
    w2 = np.asarray(w2, dtype=np.complex128).ravel()

    if w1.shape != w2.shape:
        raise ValueError(f"Shape mismatch: {w1.shape} vs {w2.shape}")

    if not np.all(np.isfinite(w1)):
        if return_raw_overlap:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan

    if not np.all(np.isfinite(w2)):
        if return_raw_overlap:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan

    n1 = np.real(np.vdot(w1, w1))
    n2 = np.real(np.vdot(w2, w2))

    if not np.isfinite(n1) or not np.isfinite(n2) or n1 <= 0.0 or n2 <= 0.0:
        if return_raw_overlap:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan

    # Normalize first.
    w1_hat = w1 / np.sqrt(n1)
    w2_hat = w2 / np.sqrt(n2)

    # Direct normalized inner product, used only to determine the optimal
    # constant phase.
    c = np.vdot(w1_hat, w2_hat)

    if not np.isfinite(c):
        if return_raw_overlap:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan

    # Phase-maximize by rotating w2 so that <w1_hat|w2_hat_aligned>
    # is real and positive.
    if np.abs(c) > 0.0:
        w2_hat = w2_hat * np.exp(-1j * np.angle(c))

    # This is exactly 1 - |<w1_hat|w2_hat>| in exact arithmetic,
    # but avoids subtracting two nearly equal numbers.
    diff = w1_hat - w2_hat
    mismatch = 0.5 * np.real(np.vdot(diff, diff))

    if not np.isfinite(mismatch):
        if return_raw_overlap:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan

    # This should already be non-negative because it is a norm.
    # Only protect against impossible tiny negative roundoff.
    if mismatch < 0.0:
        if abs(mismatch) < 1e-15:
            mismatch = 0.0
        else:
            if return_raw_overlap:
                return np.nan, np.nan, np.nan
            return np.nan, np.nan

    overlap = 1.0 - mismatch

    if return_raw_overlap:
        raw_overlap = np.abs(c)
        return float(overlap), float(mismatch), float(raw_overlap)

    return float(overlap), float(mismatch)


def resolve_link_index(specifier, link_convention : tuple) -> int:
    r"""
    Resolve one link specifier to its index in a link convention.

    Parameters
    ----------
    specifier : int or tuple
        Either a position in ``link_convention``, negative counting from the
        end as in ``numpy``, or a ``(sender, receiver)`` pair naming the link
        directly.
    link_convention : tuple
        Directed links as ``(sender, receiver)`` pairs, in the order that
        indexes axis 0 of the geometry arrays.

    Returns
    -------
    index : int
        Position of the link in ``link_convention``.
    """
    num_links = len(link_convention)

    if isinstance(specifier, (int, np.integer)):
        index = int(specifier)

        assert -num_links <= index < num_links, \
                    f"Link index must lie in [-{num_links}, {num_links}), " \
                    f"got {index}."

        return index % num_links

    pair = tuple(int(entry) for entry in specifier)

    assert pair in link_convention, \
                f"Link {pair} is not in the link convention " \
                f"{link_convention}. Build a LISAresponse whose " \
                f"link_convention contains it, or pass an index instead."

    return link_convention.index(pair)


def resolve_link_selection(link, link_convention : tuple):
    r"""
    Resolve the ``link`` argument of the response methods.

    A *tuple* of two integers is read as a ``(sender, receiver)`` pair,
    following the convention used everywhere else in this package; a *list* is
    read as a sequence of specifiers, each of which may itself be an index or a
    pair. So ``(2, 1)`` is one link, while ``[2, 1]`` is two of them, namely
    ``link_convention[2]`` and ``link_convention[1]``.

    Selecting a single link with a scalar specifier drops the link axis from
    the result, the same way ``array[0]`` drops an axis where ``array[[0]]``
    keeps it.

    This lives here rather than on ``LISAresponse`` because it is pure tuple
    lookup that happens to need one attribute. As a method it could only be
    exercised by building a whole instance, which means an orbit provider and a
    WDM transform; as a function it is testable on a bare tuple.

    Parameters
    ----------
    link : None or int or tuple or list
        ``None`` selects every link in ``link_convention``. Otherwise an index,
        a ``(sender, receiver)`` pair, or a list of either.
    link_convention : tuple
        Directed links as ``(sender, receiver)`` pairs.

    Returns
    -------
    indices : tuple
        Positions in ``link_convention``, in the requested order.
    drop_axis : bool
        Whether the caller should drop the link axis from its output.
    """
    if link is None:
        return tuple(range(len(link_convention))), False

    if isinstance(link, (int, np.integer)):
        return (resolve_link_index(link, link_convention),), True

    if (isinstance(link, tuple) and len(link) == 2
            and all(isinstance(entry, (int, np.integer)) for entry in link)):
        return (resolve_link_index(link, link_convention),), True

    indices = tuple(
        resolve_link_index(entry, link_convention) for entry in link
    )

    assert len(indices) > 0, \
                "At least one link must be selected, got an empty link."

    return indices, False


def interpolate_uniform_grid(samples,
                             grid_start : float,
                             grid_step : float,
                             query_times,
                             num_taps : int = 12):
    r"""
    Interpolate uniformly sampled series at arbitrary times.

    The samples sit at :math:`t_k = t_{\mathrm{start}} + k\,\delta t`, so the
    sample coordinate of a query time is :math:`(t - t_{\mathrm{start}})/
    \delta t` and no search is needed - the stencil follows from
    :math:`\lfloor \cdot \rfloor` alone.

    The scheme is Lagrange interpolation on ``num_taps`` samples centred on the
    query, with the classical weights

    .. math::

        h(t) = \sum_j w_j(x)\, p_j , \qquad
        w_j(x) = \prod_{m \neq j} \frac{x - o_m}{o_j - o_m} ,

    where :math:`x` is the fractional part of the sample coordinate and
    :math:`o_j` runs over the integer offsets :math:`-n/2+1 \ldots n/2`. The
    denominators are products of integers, known at trace time, so each weight
    costs one running product and a single division by a constant.

    The stencil is local, gathering taps rather than solving a system over
    the whole record, which is what lets the response quadrature evaluate it at
    arbitrary retarded phases.

    Twelve taps is the smallest even count whose error stays below the
    :math:`10^{-4}` local-delay term of the wavelet path across the whole
    usable band, EMRI harmonics included: against an exact sinusoid it gives
    1.4e-5 at 6.7 samples per cycle and ~1e-14 at 33, where the four-tap
    Catmull-Rom this replaced gives 1.5e-2 and 8.0e-5. Sixteen costs 1.6 times
    as much and buys nothing real, saturating against float64 and, sooner,
    against the error the waveform samples already carry.

    Taps falling off either end are clamped to the end sample; queries outside
    the sampled span return zero. The two are deliberately separate, since
    dropping out-of-range taps would leave the weights unnormalised and ramp a
    delayed waveform to zero across a cell instead of stopping it.

    Parameters
    ----------
    samples : jnp.ndarray
        Sampled series. Array shape=(batch, num_samples).
    grid_start : float
        Time of sample zero, :math:`t_{\mathrm{start}}`, in seconds.
    grid_step : float
        Sample spacing, :math:`\delta t`, in seconds.
    query_times : jnp.ndarray
        Times at which to evaluate, in seconds. Array shape=(batch, num_query),
        sharing the leading axis with ``samples``.
    num_taps : int
        Stencil width. Must be even and at least two, and is a compile-time
        constant since it sets the loop structure. Optional.

    Returns
    -------
    values : jnp.ndarray
        Interpolated values. Array shape=(batch, num_query).
    """
    assert num_taps >= 2 and num_taps % 2 == 0, \
                f"num_taps must be even and at least two, got {num_taps=}."

    num_samples = samples.shape[-1]

    coordinate = (query_times - grid_start) / grid_step
    base = jnp.floor(coordinate)
    fraction = coordinate - base
    index = base.astype(
        jnp.int64 if jax.config.read("jax_enable_x64") else jnp.int32
    )

    half = num_taps // 2
    offsets = tuple(range(-half + 1, half + 1))

    values = jnp.zeros_like(fraction)
    for node in offsets:
        # Numerator varies with the query; the denominator is an integer
        # product fixed by the stencil, so it folds away at trace time.
        numerator = jnp.ones_like(fraction)
        denominator = 1
        for other in offsets:
            if other != node:
                numerator = numerator * (fraction - other)
                denominator *= node - other

        taps = jnp.take_along_axis(
            samples, jnp.clip(index + node, 0, num_samples - 1), axis=-1,
        )
        values = values + (numerator / denominator) * taps

    return jnp.where(
        (coordinate >= 0.0) & (coordinate <= num_samples - 1), values, 0.0,
    )


def resolve_time_chunk_size(num_samples : int,
                            batch_size : int,
                            num_links : int,
                            memory_budget_bytes : float = 2 ** 30) -> int:
    r"""
    Choose a time-block length that keeps working memory inside a budget.

    The time-domain response allocates far more than it returns, and the excess
    is invisible from the API: per block of ``chunk`` reception times it holds
    ``80 * num_links * chunk`` bytes of host geometry - a light-travel time, a
    unit vector and two positions per link and sample - and
    ``8 * batch * num_links * chunk`` per live array of quadrature working set.
    Over four years at 5 s cadence for six links that is 11 GiB of geometry and
    14 GiB of working set against a 1 GiB result, which is why the geometry is
    rebuilt per block rather than once up front.

    Only that working memory is budgeted here. The output is the caller's own
    request, sized ``8 * batch * num_links * num_samples`` and obvious from the
    arguments, so it is left to them exactly as elsewhere in the package rather
    than second-guessed. Neither ``num_integral_points`` nor ``num_taps``
    enters: both are swept one at a time rather than materialised, so they cost
    time and not memory.

    Parameters
    ----------
    num_samples : int
        Length of the reception-time grid, :math:`N`.
    batch_size : int
        Number of sources evaluated together.
    num_links : int
        Number of directed links requested.
    memory_budget_bytes : float
        Ceiling on working memory, in bytes. Optional.

    Returns
    -------
    chunk_size : int
        Block length in samples, at least one and at most ``num_samples``.
    """
    # Live (batch, links, chunk) float64 arrays at the peak of one quadrature
    # node: query times, sample index, fractional part, then per tap a running
    # numerator, a gathered tap and the accumulator, plus slack for the two
    # quadrature sums. Scales with neither the tap count nor the node count,
    # since both are accumulated one at a time. XLA fuses much of it, so this
    # is an upper bound and the sizing it drives is conservative.
    quadrature_working_arrays = 12
    bytes_per_sample = (
        80 * num_links
        + 8 * quadrature_working_arrays * batch_size * num_links
    )

    chunk_size = int(memory_budget_bytes // bytes_per_sample)

    return max(1, min(chunk_size, int(num_samples)))
