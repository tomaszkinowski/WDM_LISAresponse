#code to generate TDI signals in TD and WDM domains 

from WDM_GW_wavelets.src.WDM.code.discrete_wavelet_transform.WDM import WDM_transform
from WDM_LISAresponse.code.lisa_response.response import one_arm_response_wdm_dT, one_arm_response_td
from WDM_GW_wavelets.src.WDM.code.time_delay_filters.time_shift_fast import wdm_time_shift_fixed, wdm_time_shift_variable
from WDM_LISAresponse.code.lisa_response.TDI_sub import fft_time_shift, interp_time_shift

def gen1_sagnac_wdm(times, waveform_plus, waveform_cross, theta, phi, orbits, Nf, Lbar, return_zeta = False):
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
    
    
    

    Returns
    -------
    alpha : ndarray
        alpha sagnac-like wdm coefficients Nt x Nf
    beta : ndarray
        beta sagnac-like wdm coefficients Nt x Nf
    gamma : ndarray
        gamma sagnac-like wdm coefficients Nt x Nf

    """

    #generate the unshifted terms 

    w21 = one_arm_response_wdm_dT(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w32 = one_arm_response_wdm_dT(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w13 = one_arm_response_wdm_dT(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w31 = one_arm_response_wdm_dT(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w23 = one_arm_response_wdm_dT(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w12 = one_arm_response_wdm_dT(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)

    dt = times[1] - times[0]
    N_pad = len(times)
    q_val = int(0.5 * N_pad / Nf)

    wdm = WDM_transform(dt=dt, Nf=Nf, N=N_pad, q=q_val, calc_m0=True, d = 4)

    #generate shifted terms 

    #for alpha
    w32_L21 = wdm_time_shift_fixed(wdm, w32, Lbar[(2,1)], Nf)
    w13_L32_L21 = wdm_time_shift_fixed(wdm, w13, Lbar[(2,1)] + Lbar[(3,2)], Nf)
    w23_L13 = wdm_time_shift_fixed(wdm, w23, Lbar[(1,3)], Nf)
    w12_L32_L13 = wdm_time_shift_fixed(wdm, w12, Lbar[(3,2)] + Lbar[(1,3)], Nf)
    #for beta
    w13_L32 = wdm_time_shift_fixed(wdm, w13, Lbar[(3,2)], Nf)
    w21_L13_L32 = wdm_time_shift_fixed(wdm, w21, Lbar[(3,2)] + Lbar[(1,3)], Nf)
    w31_L21 = wdm_time_shift_fixed(wdm, w31, Lbar[(2,1)], Nf)
    w23_L21_L13 = wdm_time_shift_fixed(wdm, w23, Lbar[(2,1)] + Lbar[(1,3)], Nf)
    #for gamma 
    w21_L13 = wdm_time_shift_fixed(wdm, w21, Lbar[(1,3)], Nf)
    w32_L21_L13 = wdm_time_shift_fixed(wdm, w32, Lbar[(2,1)] + Lbar[(1,3)], Nf)
    w12_L32 = wdm_time_shift_fixed(wdm, w12, Lbar[(3,2)], Nf)
    w31_L32_L21 = wdm_time_shift_fixed(wdm, w31, Lbar[(3,2)] + Lbar[(2,1)], Nf)

    alpha = w21 + w32_L21 + w13_L32_L21 - w31 - w23_L13 - w12_L32_L13
    beta = w32 + w13_L32 + w21_L13_L32 - w12 - w31_L21 - w23_L21_L13
    gamma = w13 + w21_L13 + w32_L21_L13 - w23 - w12_L32 - w31_L32_L21

    if return_zeta == True: 
        #for zeta 
        w21_L32 = wdm_time_shift_fixed(wdm, w21, Lbar[(3,2)], Nf)
        w31_L32 = wdm_time_shift_fixed(wdm, w31, Lbar[(3,2)], Nf)
        w32_L13 = wdm_time_shift_fixed(wdm, w32, Lbar[(1,3)], Nf)
        w12_L13 = wdm_time_shift_fixed(wdm, w12, Lbar[(1,3)], Nf)
        w13_L21 = wdm_time_shift_fixed(wdm, w13, Lbar[(2,1)], Nf)
        w23_L21 = wdm_time_shift_fixed(wdm, w23, Lbar[(2,1)], Nf)

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
    
def gen1_michelson_wdm(times, waveform_plus, waveform_cross, theta, phi, orbits, Nf, Lbar):
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

    Returns
    -------
    X, Y, Z : ndarray
        Nt x Nf WDM coefficients
    """

    # generate the unshifted terms
    w21 = one_arm_response_wdm_dT(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w32 = one_arm_response_wdm_dT(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w13 = one_arm_response_wdm_dT(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w31 = one_arm_response_wdm_dT(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w23 = one_arm_response_wdm_dT(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w12 = one_arm_response_wdm_dT(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)

    dt = times[1] - times[0]
    N_pad = len(times)
    q_val = int(0.5 * N_pad / Nf)

    wdm = WDM_transform(dt=dt, Nf=Nf, N=N_pad, q=q_val, calc_m0=True, d=4)

    #for X
    w12_L21 = wdm_time_shift_fixed(wdm, w12, Lbar[(2, 1)], Nf)
    w31_L21_L21 = wdm_time_shift_fixed(wdm, w31, Lbar[(2, 1)] + Lbar[(2, 1)], Nf)
    w13_L13_L21_L21 = wdm_time_shift_fixed(wdm, w13, Lbar[(1, 3)] + Lbar[(2, 1)] + Lbar[(2, 1)], Nf)
    w13_L13 = wdm_time_shift_fixed(wdm, w13, Lbar[(1, 3)], Nf)
    w21_L13_L13 = wdm_time_shift_fixed(wdm, w21, Lbar[(1, 3)] + Lbar[(1, 3)], Nf)
    w12_L21_L13_L13 = wdm_time_shift_fixed(wdm, w12, Lbar[(2, 1)] + Lbar[(1, 3)] + Lbar[(1, 3)], Nf)

    X = (w21 + w12_L21 + w31_L21_L21 + w13_L13_L21_L21) - (w31 + w13_L13 + w21_L13_L13 + w12_L21_L13_L13)

    #for Y
    w23_L32 = wdm_time_shift_fixed(wdm, w23, Lbar[(3, 2)], Nf)
    w12_L32_L32 = wdm_time_shift_fixed(wdm, w12, Lbar[(3, 2)] + Lbar[(3, 2)], Nf)
    w21_L21_L32_L32 = wdm_time_shift_fixed(wdm, w21, Lbar[(2, 1)] + Lbar[(3, 2)] + Lbar[(3, 2)], Nf)
    w21_L21 = wdm_time_shift_fixed(wdm, w21, Lbar[(2, 1)], Nf)
    w32_L21_L21 = wdm_time_shift_fixed(wdm, w32, Lbar[(2, 1)] + Lbar[(2, 1)], Nf)
    w23_L32_L21_L21 = wdm_time_shift_fixed(wdm, w23, Lbar[(3, 2)] + Lbar[(2, 1)] + Lbar[(2, 1)], Nf)

    Y = (w32 + w23_L32 + w12_L32_L32 + w21_L21_L32_L32) - (w12 + w21_L21 + w32_L21_L21 + w23_L32_L21_L21)

    #for Z 
    w31_L13 = wdm_time_shift_fixed(wdm, w31, Lbar[(1, 3)], Nf)
    w23_L13_L13 = wdm_time_shift_fixed(wdm, w23, Lbar[(1, 3)] + Lbar[(1, 3)], Nf)
    w32_L32_L13_L13 = wdm_time_shift_fixed(wdm, w32, Lbar[(3, 2)] + Lbar[(1, 3)] + Lbar[(1, 3)], Nf)
    w32_L32 = wdm_time_shift_fixed(wdm, w32, Lbar[(3, 2)], Nf)
    w13_L32_L32 = wdm_time_shift_fixed(wdm, w13, Lbar[(3, 2)] + Lbar[(3, 2)], Nf)
    w31_L13_L32_L32 = wdm_time_shift_fixed(wdm, w31, Lbar[(1, 3)] + Lbar[(3, 2)] + Lbar[(3, 2)], Nf)

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

def gen2_sagnac_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    shifts_alpha,
    shifts_beta,
    shifts_gamma,
):
    """
    Gen-2 Sagnac-type variables (alpha1, beta1, gamma1) in the WDM domain,
    using PRECOMPUTED composed delay arrays.

    shifts_alpha / shifts_beta / shifts_gamma are dicts mapping:
        key = ((s1,r1),(s2,r2),...)
        val = Δ(t) array, length len(times)

    IMPORTANT:
      - shifts_beta must be computed from the alpha patterns rotated by +1 (1->2->3->1)
      - shifts_gamma must be computed from the alpha patterns rotated by +2
      (i.e. from your precompute_gen2_sagnac_shifts)
    """

    # -----------------------
    # unshifted one-arm responses (directed links)
    # -----------------------
    w21 = one_arm_response_wdm_dT(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w32 = one_arm_response_wdm_dT(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w13 = one_arm_response_wdm_dT(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w31 = one_arm_response_wdm_dT(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w23 = one_arm_response_wdm_dT(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w12 = one_arm_response_wdm_dT(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)

    dt = times[1] - times[0]
    N_pad = len(times)
    q_val = int(0.5 * N_pad / Nf)
    wdm = WDM_transform(dt=dt, Nf=Nf, N=N_pad, q=q_val, calc_m0=True, d=4)

    def TS(w, shift_dict, chain):
        return wdm_time_shift_variable(wdm, w, shift_dict[chain], Nf=Nf)

    # -----------------------
    # rotate a chain key by k steps (1->2->3->1)
    # -----------------------
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    # =====================================================================
    # Base chains for ALPHA (must match your alpha_patterns in precompute)
    # (renamed as requested: c_21, c_21_32, ...)
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

    # Corresponding chains for BETA (rotate alpha chains by +1)
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

    # Corresponding chains for GAMMA (rotate alpha chains by +2)
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
    # ALPHA1 (use shifts_alpha with ALPHA chains)
    # =====================================================================
    w32_c21                  = TS(w32, shifts_alpha, cA_21)
    w13_c21_32               = TS(w13, shifts_alpha, cA_21_32)
    w31_c21_32_13            = TS(w31, shifts_alpha, cA_21_32_13)
    w23_c21_32_13_31         = TS(w23, shifts_alpha, cA_21_32_13_31)
    w12_c21_32_13_31_23      = TS(w12, shifts_alpha, cA_21_32_13_31_23)

    w23_c31                  = TS(w23, shifts_alpha, cA_31)
    w12_c31_23               = TS(w12, shifts_alpha, cA_31_23)
    w21_c31_23_12            = TS(w21, shifts_alpha, cA_31_23_12)
    w32_c31_23_12_21         = TS(w32, shifts_alpha, cA_31_23_12_21)
    w13_c31_23_12_21_32      = TS(w13, shifts_alpha, cA_31_23_12_21_32)

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

    # =====================================================================
    # BETA1 (cyclic permute links AND use shifts_beta with BETA chains)
    #
    # Link permutation (k=+1):
    #   w21->w32, w32->w13, w13->w21, w31->w12, w23->w31, w12->w23
    # =====================================================================
    w13b_c21                 = TS(w13, shifts_beta, cB_21)
    w21b_c21_32              = TS(w21, shifts_beta, cB_21_32)
    w12b_c21_32_13           = TS(w12, shifts_beta, cB_21_32_13)
    w31b_c21_32_13_31        = TS(w31, shifts_beta, cB_21_32_13_31)
    w23b_c21_32_13_31_23     = TS(w23, shifts_beta, cB_21_32_13_31_23)

    w31b_c31                 = TS(w31, shifts_beta, cB_31)
    w23b_c31_23              = TS(w23, shifts_beta, cB_31_23)
    w32b_c31_23_12           = TS(w32, shifts_beta, cB_31_23_12)
    w13b_c31_23_12_21        = TS(w13, shifts_beta, cB_31_23_12_21)
    w21b_c31_23_12_21_32     = TS(w21, shifts_beta, cB_31_23_12_21_32)

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

    # =====================================================================
    # GAMMA1 (cyclic permute twice AND use shifts_gamma with GAMMA chains)
    #
    # Link permutation (k=+2):
    #   w21->w13, w32->w21, w13->w32, w31->w23, w23->w12, w12->w31
    # =====================================================================
    w21g_c21                 = TS(w21, shifts_gamma, cG_21)
    w32g_c21_32              = TS(w32, shifts_gamma, cG_21_32)
    w23g_c21_32_13           = TS(w23, shifts_gamma, cG_21_32_13)
    w12g_c21_32_13_31        = TS(w12, shifts_gamma, cG_21_32_13_31)
    w31g_c21_32_13_31_23     = TS(w31, shifts_gamma, cG_21_32_13_31_23)

    w12g_c31                 = TS(w12, shifts_gamma, cG_31)
    w31g_c31_23              = TS(w31, shifts_gamma, cG_31_23)
    w13g_c31_23_12           = TS(w13, shifts_gamma, cG_31_23_12)
    w21g_c31_23_12_21        = TS(w21, shifts_gamma, cG_31_23_12_21)
    w32g_c31_23_12_21_32     = TS(w32, shifts_gamma, cG_31_23_12_21_32)

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

def gen2_sagnac_td(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    shifts_alpha,
    shifts_beta,
    shifts_gamma,
    order=7,
    mode="zero",
):
    """
    Gen-2 Sagnac-type variables (alpha1, beta1, gamma1) in the TIME DOMAIN,
    in the same "style" as gen2_sagnac_wdm: use PRECOMPUTED composed delay arrays.

    shifts_alpha / shifts_beta / shifts_gamma are dicts mapping:
        key = ((s1,r1),(s2,r2),...)
        val = Δ(t) array, length len(times)
    where Δ(t) is the PHYSICAL delay so we apply x(t-Δ(t)).

    Notes:
      - shifts_beta should be alpha chains rotated by +1 (1->2->3->1)
      - shifts_gamma should be alpha chains rotated by +2
    """

    # -----------------------
    # unshifted one-arm responses (directed links)
    # -----------------------
    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, 1000)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, 1000)

    dt = times[1] - times[0]

    # -----------------------
    # generic time-shift helper: x(t) -> x(t - Δ(t)) via interpolation
    # -----------------------
    def TS(x, shift_dict, chain):
        Delta_t = shift_dict[chain]  # length len(times)
        return interp_time_shift(x, dt, Delta_t, order=order, mode=mode)

    # -----------------------
    # rotate a chain key by k steps (1->2->3->1)
    # -----------------------
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

    # Corresponding chains for BETA (rotate alpha chains by +1)
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

    # Corresponding chains for GAMMA (rotate alpha chains by +2)
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

def gen2_michelson_wdm(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    Nf,
    shifts_X,
    shifts_Y,
    shifts_Z,
):
    """
    Gen-2 Michelson-type variables (X1, Y1, Z1) in the WDM domain,
    in the same explicit "named-chain" style as your gen2_sagnac_wdm.

    Uses PRECOMPUTED composed delay arrays:
      shifts_X / shifts_Y / shifts_Z : dict
        key = ((s1,r1),(s2,r2),...)
        val = Δ(t) array, length len(times)
    """

    # -----------------------
    # unshifted one-arm responses (directed links)
    # -----------------------
    w21 = one_arm_response_wdm_dT(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w32 = one_arm_response_wdm_dT(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w13 = one_arm_response_wdm_dT(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w31 = one_arm_response_wdm_dT(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w23 = one_arm_response_wdm_dT(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, Nf)
    w12 = one_arm_response_wdm_dT(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, Nf)

    dt = times[1] - times[0]
    N_pad = len(times)
    q_val = int(0.5 * N_pad / Nf)
    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N_pad, q=q_val, calc_m0=True, d=4)

    def TS(w, shift_dict, chain):
        return wdm_time_shift_variable(wdm, w, shift_dict[chain], Nf=Nf)

    # -----------------------
    # rotate a chain key by k steps (1->2->3->1)
    # -----------------------
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    # =====================================================================
    # Base chains for X (MUST match your X_patterns in precompute_gen2_michelson_shifts)
    # =====================================================================
    cX_21                              = ((2, 1),)
    cX_21_12                           = ((2, 1), (1, 2))
    cX_21_12_31                        = ((2, 1), (1, 2), (3, 1))
    cX_21_12_31_13                     = ((2, 1), (1, 2), (3, 1), (1, 3))
    cX_21_12_31_13_31                  = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1))
    cX_21_12_31_13_31_13               = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3))
    cX_21_12_31_13_31_13_21            = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3), (2, 1))

    cX_31                              = ((3, 1),)
    cX_31_13                           = ((3, 1), (1, 3))
    cX_31_13_21                        = ((3, 1), (1, 3), (2, 1))
    cX_31_13_21_12                     = ((3, 1), (1, 3), (2, 1), (1, 2))
    cX_31_13_21_12_21                  = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1))
    cX_31_13_21_12_21_12               = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2))
    cX_31_13_21_12_21_12_31            = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2), (3, 1))

    # Y chains
    cY_21                              = rot_chain(cX_21, 1)
    cY_21_12                           = rot_chain(cX_21_12, 1)
    cY_21_12_31                        = rot_chain(cX_21_12_31, 1)
    cY_21_12_31_13                     = rot_chain(cX_21_12_31_13, 1)
    cY_21_12_31_13_31                  = rot_chain(cX_21_12_31_13_31, 1)
    cY_21_12_31_13_31_13               = rot_chain(cX_21_12_31_13_31_13, 1)
    cY_21_12_31_13_31_13_21            = rot_chain(cX_21_12_31_13_31_13_21, 1)

    cY_31                              = rot_chain(cX_31, 1)
    cY_31_13                           = rot_chain(cX_31_13, 1)
    cY_31_13_21                        = rot_chain(cX_31_13_21, 1)
    cY_31_13_21_12                     = rot_chain(cX_31_13_21_12, 1)
    cY_31_13_21_12_21                  = rot_chain(cX_31_13_21_12_21, 1)
    cY_31_13_21_12_21_12               = rot_chain(cX_31_13_21_12_21_12, 1)
    cY_31_13_21_12_21_12_31            = rot_chain(cX_31_13_21_12_21_12_31, 1)

    # Z chains
    cZ_21                              = rot_chain(cX_21, 2)
    cZ_21_12                           = rot_chain(cX_21_12, 2)
    cZ_21_12_31                        = rot_chain(cX_21_12_31, 2)
    cZ_21_12_31_13                     = rot_chain(cX_21_12_31_13, 2)
    cZ_21_12_31_13_31                  = rot_chain(cX_21_12_31_13_31, 2)
    cZ_21_12_31_13_31_13               = rot_chain(cX_21_12_31_13_31_13, 2)
    cZ_21_12_31_13_31_13_21            = rot_chain(cX_21_12_31_13_31_13_21, 2)

    cZ_31                              = rot_chain(cX_31, 2)
    cZ_31_13                           = rot_chain(cX_31_13, 2)
    cZ_31_13_21                        = rot_chain(cX_31_13_21, 2)
    cZ_31_13_21_12                     = rot_chain(cX_31_13_21_12, 2)
    cZ_31_13_21_12_21                  = rot_chain(cX_31_13_21_12_21, 2)
    cZ_31_13_21_12_21_12               = rot_chain(cX_31_13_21_12_21_12, 2)
    cZ_31_13_21_12_21_12_31            = rot_chain(cX_31_13_21_12_21_12_31, 2)

    # X1 shifted pieces
    w12_cX_21                           = TS(w12, shifts_X, cX_21)
    w31_cX_21_12                        = TS(w31, shifts_X, cX_21_12)
    w13_cX_21_12_31                     = TS(w13, shifts_X, cX_21_12_31)
    w31_cX_21_12_31_13                  = TS(w31, shifts_X, cX_21_12_31_13)
    w13_cX_21_12_31_13_31               = TS(w13, shifts_X, cX_21_12_31_13_31)
    w21_cX_21_12_31_13_31_13            = TS(w21, shifts_X, cX_21_12_31_13_31_13)
    w12_cX_21_12_31_13_31_13_21         = TS(w12, shifts_X, cX_21_12_31_13_31_13_21)

    w13_cX_31                           = TS(w13, shifts_X, cX_31)
    w21_cX_31_13                        = TS(w21, shifts_X, cX_31_13)
    w12_cX_31_13_21                     = TS(w12, shifts_X, cX_31_13_21)
    w21_cX_31_13_21_12                  = TS(w21, shifts_X, cX_31_13_21_12)
    w12_cX_31_13_21_12_21               = TS(w12, shifts_X, cX_31_13_21_12_21)
    w31_cX_31_13_21_12_21_12            = TS(w31, shifts_X, cX_31_13_21_12_21_12)
    w13_cX_31_13_21_12_21_12_31         = TS(w13, shifts_X, cX_31_13_21_12_21_12_31)

    X1 = (
        w21
        + w12_cX_21
        + w31_cX_21_12
        + w13_cX_21_12_31
        + w31_cX_21_12_31_13
        + w13_cX_21_12_31_13_31
        + w21_cX_21_12_31_13_31_13
        + w12_cX_21_12_31_13_31_13_21
        - w31
        - w13_cX_31
        - w21_cX_31_13
        - w12_cX_31_13_21
        - w21_cX_31_13_21_12
        - w12_cX_31_13_21_12_21
        - w31_cX_31_13_21_12_21_12
        - w13_cX_31_13_21_12_21_12_31
    )

    # Rotate which raw link each measurement uses
    W = {(2,1): w21, (3,2): w32, (1,3): w13, (3,1): w31, (2,3): w23, (1,2): w12}

    def eval_michelson_from_template_wdm(shift_dict,
                                         c21, c21_12, c21_12_31, c21_12_31_13, c21_12_31_13_31, c21_12_31_13_31_13, c21_12_31_13_31_13_21,
                                         c31, c31_13, c31_13_21, c31_13_21_12, c31_13_21_12_21, c31_13_21_12_21_12, c31_13_21_12_21_12_31,
                                         k):
        def R(link): return rot_link(link, k)

        y231 = W[R((2,1))]
        y13p2 = W[R((1,2))]
        y32p1 = W[R((3,1))]
        y123 = W[R((1,3))]

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

        return (
            y231
            + t1 + t2 + t3 + t4 + t5 + t6 + t7
            - y32p1
            - n1 - n2 - n3 - n4 - n5 - n6 - n7
        )

    Y1 = eval_michelson_from_template_wdm(
        shifts_Y,
        cY_21, cY_21_12, cY_21_12_31, cY_21_12_31_13, cY_21_12_31_13_31, cY_21_12_31_13_31_13, cY_21_12_31_13_31_13_21,
        cY_31, cY_31_13, cY_31_13_21, cY_31_13_21_12, cY_31_13_21_12_21, cY_31_13_21_12_21_12, cY_31_13_21_12_21_12_31,
        k=1
    )

    Z1 = eval_michelson_from_template_wdm(
        shifts_Z,
        cZ_21, cZ_21_12, cZ_21_12_31, cZ_21_12_31_13, cZ_21_12_31_13_31, cZ_21_12_31_13_31_13, cZ_21_12_31_13_31_13_21,
        cZ_31, cZ_31_13, cZ_31_13_21, cZ_31_13_21_12, cZ_31_13_21_12_21, cZ_31_13_21_12_21_12, cZ_31_13_21_12_21_12_31,
        k=2
    )

    return X1, Y1, Z1

def gen2_michelson_td(
    times,
    waveform_plus,
    waveform_cross,
    theta,
    phi,
    orbits,
    shifts_X,
    shifts_Y,
    shifts_Z,
    order=7,
    mode="zero",
    num_int_points=1000,
):
    """
    Gen-2 Michelson-type variables (X1, Y1, Z1) in the TIME DOMAIN,
    in the same explicit "named-chain" style as your gen2_sagnac_td.

    Uses PRECOMPUTED composed delay arrays:
      shifts_X / shifts_Y / shifts_Z : dict
        key = ((s1,r1),(s2,r2),...)
        val = Δ(t) array, length len(times)

    Notes
    -----
    - Y and Z are cyclic permutations (1->2->3->1) of X.
    - No "empty / no-shift" pattern is included.
    """

    # -----------------------
    # unshifted one-arm responses (directed links)
    # -----------------------
    dl_21 = one_arm_response_td(times, 2, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_32 = one_arm_response_td(times, 3, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_13 = one_arm_response_td(times, 1, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_31 = one_arm_response_td(times, 3, 1, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_23 = one_arm_response_td(times, 2, 3, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)
    dl_12 = one_arm_response_td(times, 1, 2, orbits, theta, phi, waveform_plus, waveform_cross, num_int_points)

    dt = times[1] - times[0]

    # -----------------------
    # generic time-shift helper: x(t) -> x(t - Δ(t)) via interpolation
    # -----------------------
    def TS(x, shift_dict, chain):
        Delta_t = shift_dict[chain]  # length len(times)
        return interp_time_shift(x, dt, Delta_t, order=order, mode=mode)

    # -----------------------
    # rotate a chain key by k steps (1->2->3->1)
    # -----------------------
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    # =====================================================================
    # Base chains for X (MUST match your X_patterns in precompute_gen2_michelson_shifts)
    # =====================================================================
    cX_21                              = ((2, 1),)
    cX_21_12                           = ((2, 1), (1, 2))
    cX_21_12_31                        = ((2, 1), (1, 2), (3, 1))
    cX_21_12_31_13                     = ((2, 1), (1, 2), (3, 1), (1, 3))
    cX_21_12_31_13_31                  = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1))
    cX_21_12_31_13_31_13               = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3))
    cX_21_12_31_13_31_13_21            = ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3), (2, 1))

    cX_31                              = ((3, 1),)
    cX_31_13                           = ((3, 1), (1, 3))
    cX_31_13_21                        = ((3, 1), (1, 3), (2, 1))
    cX_31_13_21_12                     = ((3, 1), (1, 3), (2, 1), (1, 2))
    cX_31_13_21_12_21                  = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1))
    cX_31_13_21_12_21_12               = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2))
    cX_31_13_21_12_21_12_31            = ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2), (3, 1))

    # Corresponding chains for Y (rotate X chains by +1)
    cY_21                              = rot_chain(cX_21, 1)
    cY_21_12                           = rot_chain(cX_21_12, 1)
    cY_21_12_31                        = rot_chain(cX_21_12_31, 1)
    cY_21_12_31_13                     = rot_chain(cX_21_12_31_13, 1)
    cY_21_12_31_13_31                  = rot_chain(cX_21_12_31_13_31, 1)
    cY_21_12_31_13_31_13               = rot_chain(cX_21_12_31_13_31_13, 1)
    cY_21_12_31_13_31_13_21            = rot_chain(cX_21_12_31_13_31_13_21, 1)

    cY_31                              = rot_chain(cX_31, 1)
    cY_31_13                           = rot_chain(cX_31_13, 1)
    cY_31_13_21                        = rot_chain(cX_31_13_21, 1)
    cY_31_13_21_12                     = rot_chain(cX_31_13_21_12, 1)
    cY_31_13_21_12_21                  = rot_chain(cX_31_13_21_12_21, 1)
    cY_31_13_21_12_21_12               = rot_chain(cX_31_13_21_12_21_12, 1)
    cY_31_13_21_12_21_12_31            = rot_chain(cX_31_13_21_12_21_12_31, 1)

    # Corresponding chains for Z (rotate X chains by +2)
    cZ_21                              = rot_chain(cX_21, 2)
    cZ_21_12                           = rot_chain(cX_21_12, 2)
    cZ_21_12_31                        = rot_chain(cX_21_12_31, 2)
    cZ_21_12_31_13                     = rot_chain(cX_21_12_31_13, 2)
    cZ_21_12_31_13_31                  = rot_chain(cX_21_12_31_13_31, 2)
    cZ_21_12_31_13_31_13               = rot_chain(cX_21_12_31_13_31_13, 2)
    cZ_21_12_31_13_31_13_21            = rot_chain(cX_21_12_31_13_31_13_21, 2)

    cZ_31                              = rot_chain(cX_31, 2)
    cZ_31_13                           = rot_chain(cX_31_13, 2)
    cZ_31_13_21                        = rot_chain(cX_31_13_21, 2)
    cZ_31_13_21_12                     = rot_chain(cX_31_13_21_12, 2)
    cZ_31_13_21_12_21                  = rot_chain(cX_31_13_21_12_21, 2)
    cZ_31_13_21_12_21_12               = rot_chain(cX_31_13_21_12_21_12, 2)
    cZ_31_13_21_12_21_12_31            = rot_chain(cX_31_13_21_12_21_12_31, 2)

    # =====================================================================
    # X1 (use shifts_X with X chains)
    # Expression as you gave it:
    # X1 = y231 + y13'2;3 + y32'1;3'3 + y123;2'3'3 + y32'1;22'3'3 + y123;2'22'3'3 + y231;22'22'3'3 + y13'2;322'22'3'3
    #    - y32'1 - y123;2' - y231;22' - y13'2;322' - y231;3'322' - y13'2;33'322' - y32'1;3'33'322' - y123;2'3'33'322'
    #
    # In your link notation mapping:
    #   y231   -> dl_21
    #   y13'2  -> dl_12
    #   y32'1  -> dl_31
    #   y123   -> dl_13
    # =====================================================================

    # + terms
    dl_12_cX_21                              = TS(dl_12, shifts_X, cX_21)                 # ;3
    dl_31_cX_21_12                           = TS(dl_31, shifts_X, cX_21_12)              # ;3'3
    dl_13_cX_21_12_31                        = TS(dl_13, shifts_X, cX_21_12_31)           # ;2'3'3
    dl_31_cX_21_12_31_13                     = TS(dl_31, shifts_X, cX_21_12_31_13)        # ;22'3'3
    dl_13_cX_21_12_31_13_31                  = TS(dl_13, shifts_X, cX_21_12_31_13_31)     # ;2'22'3'3
    dl_21_cX_21_12_31_13_31_13               = TS(dl_21, shifts_X, cX_21_12_31_13_31_13)  # ;22'22'3'3
    dl_12_cX_21_12_31_13_31_13_21            = TS(dl_12, shifts_X, cX_21_12_31_13_31_13_21)  # ;322'22'3'3

    # - terms
    dl_13_cX_31                              = TS(dl_13, shifts_X, cX_31)                 # ;2'
    dl_21_cX_31_13                           = TS(dl_21, shifts_X, cX_31_13)              # ;22'
    dl_12_cX_31_13_21                        = TS(dl_12, shifts_X, cX_31_13_21)           # ;322'
    dl_21_cX_31_13_21_12                     = TS(dl_21, shifts_X, cX_31_13_21_12)        # ;3'322'
    dl_12_cX_31_13_21_12_21                  = TS(dl_12, shifts_X, cX_31_13_21_12_21)     # ;33'322'
    dl_31_cX_31_13_21_12_21_12               = TS(dl_31, shifts_X, cX_31_13_21_12_21_12)  # ;3'33'322'
    dl_13_cX_31_13_21_12_21_12_31            = TS(dl_13, shifts_X, cX_31_13_21_12_21_12_31) # ;2'3'33'322'

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

    # =====================================================================
    # Y1 and Z1: exact cyclic permutations of X1.
    #
    # We implement them by permuting the base links and using shifts_Y/shifts_Z
    # with the corresponding rotated chain keys.
    # =====================================================================

    # Convenience dict for rotating the *data links* (directed one-arm measurements)
    DL = {(2,1): dl_21, (3,2): dl_32, (1,3): dl_13, (3,1): dl_31, (2,3): dl_23, (1,2): dl_12}

    def eval_michelson_from_template(shift_dict, c21, c21_12, c21_12_31, c21_12_31_13, c21_12_31_13_31, c21_12_31_13_31_13, c21_12_31_13_31_13_21,
                                     c31, c31_13, c31_13_21, c31_13_21_12, c31_13_21_12_21, c31_13_21_12_21_12, c31_13_21_12_21_12_31,
                                     k):
        # rotate which raw one-arm series each y(...) uses:
        # link (s,r) -> rot_link((s,r), k)
        def R(link): return rot_link(link, k)

        y231 = DL[R((2,1))]
        y13p2 = DL[R((1,2))]
        y32p1 = DL[R((3,1))]
        y123 = DL[R((1,3))]

        # shifted pieces (use rotated chains already passed in)
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

        return (
            y231
            + t1 + t2 + t3 + t4 + t5 + t6 + t7
            - y32p1
            - n1 - n2 - n3 - n4 - n5 - n6 - n7
        )

    # Y1: k=+1, use shifts_Y and Y-chains
    Y1 = eval_michelson_from_template(
        shifts_Y,
        cY_21, cY_21_12, cY_21_12_31, cY_21_12_31_13, cY_21_12_31_13_31, cY_21_12_31_13_31_13, cY_21_12_31_13_31_13_21,
        cY_31, cY_31_13, cY_31_13_21, cY_31_13_21_12, cY_31_13_21_12_21, cY_31_13_21_12_21_12, cY_31_13_21_12_21_12_31,
        k=1
    )

    # Z1: k=+2, use shifts_Z and Z-chains
    Z1 = eval_michelson_from_template(
        shifts_Z,
        cZ_21, cZ_21_12, cZ_21_12_31, cZ_21_12_31_13, cZ_21_12_31_13_31, cZ_21_12_31_13_31_13, cZ_21_12_31_13_31_13_21,
        cZ_31, cZ_31_13, cZ_31_13_21, cZ_31_13_21_12, cZ_31_13_21_12_21, cZ_31_13_21_12_21_12, cZ_31_13_21_12_21_12_31,
        k=2
    )

    return X1, Y1, Z1