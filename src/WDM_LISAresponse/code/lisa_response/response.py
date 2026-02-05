"""
LISA response calculations for WDM_LISAresponse
"""

import jax
import jax.numpy as jnp
import numpy as np 
from lisaconstants import c
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import WDM
from numpy.fft import fft, fftfreq
from .response_sub import (
    basis_vectors,
    basis_tensors,
    _compute_ltt_iterative,
    rhat, 
    gnm_at_times_grid_np,
    gnm_at_times_grid_np_interp,
)
# Add your LISA response functions here

def one_arm_response_td(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, num_int_points=10):
    """
    Eq. 12

    Parameters 
    ----------
    times : ndarray
        the sample times of the LISA response
        this is the time at which the signal is sent by SC 
    s : int
        sending SC
    r : int
        sending SC
    orbits : LISA orbits class objects
        computes the positions of all the SC
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]
    td_waveform_plus : ndarray
        time domain waveform of the plus polarisation of len = times  
    td_waveform_cross :
        time domain waveform of the cross polarisation of len = times 
    num_int_points : int
        number of points used for trapezium rule integration 

    Returns
    -------
    """
    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr = _compute_ltt_iterative(times, s, r, orbits)
    rhat_sr = rhat(times, s, r, orbits)

    t_rec = times + l_sr
    x_r_rec = orbits.compute_position(t_rec, sc=[r])[:,0,:]

    D = np.einsum('tx,tX->txX', rhat_sr, rhat_sr)

    _, _, n = basis_vectors(theta, phi)

    Omega = -n 

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)

    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)
    xi_r = t_rec - np.einsum('x,tx->t', Omega, x_r_rec/c)

    xi = np.linspace(xi_s, xi_r, num_int_points) # (num_int_points, num_times)

    h_plus_interp = interp1d(times, td_waveform_plus, kind="cubic", bounds_error=False, fill_value="extrapolate") #change back to extrapolate
    h_cross_interp = interp1d(times, td_waveform_cross, kind="cubic", bounds_error=False, fill_value="extrapolate")
    h_plus = h_plus_interp(xi)
    h_cross = h_cross_interp(xi)

    H_plus = np.trapezoid(h_plus, x=xi, axis=0)
    H_cross = np.trapezoid(h_cross, x=xi, axis=0)

    e_plus, e_cross = basis_tensors(theta, phi)

    H = H_plus[:, np.newaxis, np.newaxis] * e_plus[np.newaxis, :, :] + \
        H_cross[:, np.newaxis, np.newaxis] * e_cross[np.newaxis, :, :]

    dl_sr = 0.5 * (1-Om_dot_rhat)**(-1) * np.einsum('txX,txX->t', D, H)

    return c* dl_sr

def one_arm_response_freq_domain(
    times, s, r, orbits, theta, phi,
    td_waveform_plus, td_waveform_cross,
    chunk_size=512
):
    """
    Faster FD one-arm response:
    - avoids building 4D tensors (Nt x Nw x 3 x 3)
    - avoids large Nt x Nw arrays via frequency chunking
    - uses scalar antenna factors F_plus(t), F_cross(t)
    """
    times = np.asarray(times)
    Nt = times.size

    # Assume uniform sampling
    dt = times[1] - times[0]

    # Geometry (still computed here; biggest speedup across many calls is caching these externally)
    x_s = orbits.compute_position(times, sc=[s])[:, 0, :]          # (Nt,3)
    l_sr = _compute_ltt_iterative(times, s, r, orbits)              # (Nt,)
    rhat_sr = rhat(times, s, r, orbits)                             # (Nt,3)

    _, _, n = basis_vectors(theta, phi)
    Omega = -n

    Om_dot_rhat = np.einsum('i,ti->t', Omega, rhat_sr)              # (Nt,)
    xi_s = times - np.einsum('i,ti->t', Omega, x_s / c)             # (Nt,)

    # FFT of waveform polarisations
    H_plus = fft(td_waveform_plus)
    H_cross = fft(td_waveform_cross)

    h_freq = fftfreq(Nt, dt)
    omega = 2.0 * np.pi * h_freq                                    # (Nt,)

    # Antenna scalars: F_+(t) = rhat^T e_+ rhat, same for cross
    e_plus, e_cross = basis_tensors(theta, phi)
    F_plus  = np.einsum('ti,ij,tj->t', rhat_sr, e_plus,  rhat_sr)   # (Nt,)
    F_cross = np.einsum('ti,ij,tj->t', rhat_sr, e_cross, rhat_sr)   # (Nt,)

    # Combine constant phase shift into one term
    phase_t = (xi_s - times[0])                                     # (Nt,)

    # Accumulate inverse-sum in chunks to reduce memory and often improve speed
    acc = np.zeros(Nt, dtype=complex)
    one_minus = (1.0 - Om_dot_rhat)                                 # (Nt,)

    for j0 in range(0, Nt, chunk_size):
        j1 = min(j0 + chunk_size, Nt)

        w = omega[j0:j1]                                            # (M,)
        Hp = H_plus[j0:j1]
        Hc = H_cross[j0:j1]

        # transfer_func_arg = 0.5 * omega * l_sr * (1 - Om·rhat)
        arg = 0.5 * (l_sr[:, None] * one_minus[:, None]) * w[None, :]  # (Nt,M)

        # transfer function: sinc(arg/pi) * exp(i*arg)
        tf = np.sinc(arg / np.pi) * np.exp(1j * arg)                # (Nt,M)

        # scalar contraction: 0.5 * tf * (Hp*F_plus + Hc*F_cross)
        resp = 0.5 * tf * (F_plus[:, None] * Hp[None, :] + F_cross[:, None] * Hc[None, :])  # (Nt,M)

        # exp(i w * (xi_s - t0))
        ph = np.exp(1j * phase_t[:, None] * w[None, :])             # (Nt,M)

        acc += np.sum(ph * resp, axis=1)                            # (Nt,)

    dl_sr = c * l_sr * (acc / Nt)
    return dl_sr



def one_arm_response_wdm(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, gnm_method="cubic", gnm_wrap=True, d = 4):
    """
    Implementation of the wdm one arm response as coded in the Overleaf, equation 20.  

    Parameters 
    ----------
    times : ndarray
        the sample times of the LISA response
        this is the time at which the signal is sent by SC 
    s : int
        sending SC
    r : int
        sending SC
    orbits : LISA orbits class objects
        computes the positions of all the SC
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]
    td_waveform_plus : ndarray
        time domain waveform of the plus polarisation of len = times  
    td_waveform_cross : ndarray
        time domain waveform of the cross polarisation of len = times 
    Nf : int
        number of basis element in the frequency dimension 

    Returns
    -------
    """
    current_length = len(times)

    dt = np.mean(np.diff(times))

    N = WDM.utils.next_multiple(current_length, 2*Nf)

    if int(N/Nf) == 2:
        q_val = 1
    else:
        q_val = int(0.5 * N/Nf)

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=q_val, calc_m0=True, d = d)
    print(wdm)

    h_plus_padded, mask = WDM.utils.pad_signal(td_waveform_plus, N)

    print(f"How much padding was added: {(N-current_length)*wdm.dt}")

    w_t = wdm.dwt(h_plus_padded)

    #basic implementation of wdm transform 

    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr = _compute_ltt_iterative(times, s, r, orbits) 
        
    rhat_sr = rhat(times, s, r, orbits)

    rhat_cross = np.einsum('tx,tX->txX', rhat_sr, rhat_sr)

    _, _, n = basis_vectors(theta, phi)

    Omega = -n 

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)
        
    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)

    freq_vals = np.arange(0, wdm.Nf) * wdm.dF

    tau_s = l_sr * (1 - Om_dot_rhat) * 0.5
    sinc_arg = 2.0 * np.pi * freq_vals[np.newaxis, :] * tau_s[:, np.newaxis]

    e_plus, _ = basis_tensors(theta, phi)

    n_grid, m_grid = np.meshgrid(np.arange(wdm.Nt), np.arange(wdm.Nf), indexing="ij")

    targ = xi_s + tau_s - times[0]

    vals = gnm_at_times_grid_np_interp(wdm, n_grid, m_grid, xi_s + tau_s - times[0], method=gnm_method, wrap=True)

    sum_nm = np.sinc(sinc_arg / np.pi)[:, np.newaxis, :] * w_t[np.newaxis, :, :] * vals

    dl_sr_wdm = 0.5 * c * l_sr * np.einsum('tXx,Xx->t', rhat_cross, e_plus) * np.einsum("tnm->t", sum_nm)

    return dl_sr_wdm

def one_arm_response_wdm_dT(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, return_td = False, gnm_method="cubic", gnm_wrap=True, d = 4):
    """
    WDM response function where we also approximate the times t_i as t_n, see equation 21 for details

    Parameters 
    ----------
    times : ndarray
        the sample times of the LISA response
        this is the time at which the signal is sent by SC 
    s : int
        sending SC
    r : int
        sending SC
    orbits : LISA orbits class objects
        computes the positions of all the SC
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]
    td_waveform_plus : ndarray
        time domain waveform of the plus polarisation of len = times  
    td_waveform_cross : ndarray
        time domain waveform of the cross polarisation of len = times 
    Nf : int
        number of basis element in the frequency dimension 
    return_td : bool
        indicate whether we also want the time domain lisa response

    Returns
    -------
    """
    current_length = len(times)
    dt = np.mean(np.diff(times))

    N = WDM.utils.next_multiple(current_length, 2*Nf)
    q_val = 1 if int(N/Nf) == 2 else int(0.5 * N/Nf)

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=q_val, calc_m0=True, d=d, A_frac=0.2)
    #print(wdm)
    h_plus_padded, mask = WDM.utils.pad_signal(td_waveform_plus, N)
    w_t = wdm.dwt(h_plus_padded)  # shape (Nt, Nf) presumably

    x_s = orbits.compute_position(times, sc=[s])[:, 0, :]

    _, _, n = basis_vectors(theta, phi)
    Omega = -n

    # faster than einsum in many cases
    xi_s = times - (x_s @ (Omega / c))

    freq_vals = np.arange(wdm.Nf) * wdm.dF
    times_vals = np.arange(wdm.Nt) * wdm.dT
    int_times_vals = times[0] + times_vals

    l_sr = _compute_ltt_iterative(times, s, r, orbits)
    l_sr_dT = _compute_ltt_iterative(int_times_vals, s, r, orbits)

    rhat_sr = rhat(times, s, r, orbits)
    rhat_sr_dT = rhat(int_times_vals, s, r, orbits)

    Om_dot_rhat = rhat_sr @ Omega
    Om_dot_rhat_dT = rhat_sr_dT @ Omega

    tau_s = 0.5 * l_sr * (1 - Om_dot_rhat)
    tau_n = 0.5 * l_sr_dT * (1 - Om_dot_rhat_dT)

    e_plus, _ = basis_tensors(theta, phi)

    # rhat_cross_dT is (Nt,3,3); proj_dT -> (Nt,)
    # compute outer products + projection
    rhat_cross_dT = rhat_sr_dT[:, :, None] * rhat_sr_dT[:, None, :]
    proj_dT = np.einsum("tij,ij->t", rhat_cross_dT, e_plus)

    # sinc simplification: np.sinc(2 f tau)
    sinc_mat = np.sinc(2.0 * freq_vals[None, :] * tau_n[:, None])

    A_nm = 0.5 * c * l_sr_dT[:, None] * proj_dT[:, None] * sinc_mat

    if not return_td:
        return A_nm * w_t

    # For return_td: strongly consider chunking if gnm_at_times_grid_np_interp is heavy
    n_grid, m_grid = np.meshgrid(np.arange(wdm.Nt), np.arange(Nf), indexing="ij")
    tq = xi_s + tau_s - times[0]
    vals = gnm_at_times_grid_np_interp(
        wdm, n_grid, m_grid, tq, method=gnm_method, wrap=True
    )
    dl_sr = np.einsum("tnm,nm->t", vals, (A_nm * w_t))
    return dl_sr, (A_nm * w_t)


