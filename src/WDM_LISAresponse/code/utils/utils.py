"""
Utility functions for WDM_LISAresponse
"""
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