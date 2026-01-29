"""
Utility functions for WDM_LISAresponse
"""
import numpy as np

# Add your utility functions here

def compare_response_signals(h_ref, h_test):
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