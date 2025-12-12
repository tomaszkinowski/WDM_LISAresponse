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
from scipy.fft import fft, fftfreq
from .time_domain_response import (
    basis_vectors,
    basis_tensors,
    _compute_ltt_iterative,
    rhat, gnm_at_times_grid_np
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
    x_r = orbits.compute_position(times, sc=[r])[:,0,:]

    l_sr = _compute_ltt_iterative(times, s, r, orbits)
    rhat_sr = rhat(times, s, r, orbits)

    D = np.einsum('tx,tX->txX', rhat_sr, rhat_sr)

    _, _, n = basis_vectors(theta, phi)

    Omega = -n 

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)

    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)
    xi_r = times + l_sr - np.einsum('x,tx->t', Omega, x_r/c)

    xi = np.linspace(xi_s, xi_r, num_int_points) # (num_int_points, num_times)

    h_plus_interp = interp1d(times, td_waveform_plus, kind="linear", bounds_error=False, fill_value="extrapolate")
    h_cross_interp = interp1d(times, td_waveform_cross, kind="linear", bounds_error=False, fill_value="extrapolate")

    h_plus = h_plus_interp(xi)
    h_cross = h_cross_interp(xi)

    H_plus = np.trapezoid(h_plus, x=xi, axis=0)
    H_cross = np.trapezoid(h_cross, x=xi, axis=0)

    e_plus, e_cross = basis_tensors(theta, phi)

    H = H_plus[:, np.newaxis, np.newaxis] * e_plus[np.newaxis, :, :] + \
        H_cross[:, np.newaxis, np.newaxis] * e_cross[np.newaxis, :, :]

    dl_sr = 0.5 * (1-Om_dot_rhat)**(-1) * np.einsum('txX,txX->t', D, H)

    return c* dl_sr

def one_arm_response_freq_domain(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross):
    """
    Eq. 19

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

    Returns
    -------
    """
    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr = _compute_ltt_iterative(times, s, r, orbits)
    
    omega_sr = (l_sr)**(-1)  


    rhat_sr = rhat(times, s, r, orbits)

    rhat_cross = np.einsum('tx,tX->txX', rhat_sr, rhat_sr)

    _, _, n = basis_vectors(theta, phi)

    Omega = -n 

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)
    

    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)

    #times_f = np.linspace(xi_s.min(), xi_s.max(), 10000)

    dt = np.diff(times).mean()
    len_times = len(times)
    H_plus = fft(td_waveform_plus)
    H_cross = fft(td_waveform_cross)
    h_freq = fftfreq(len_times, dt)
    omega = 2.0 * np.pi * h_freq
    
    transfer_func_arg = 0.5 * omega[np.newaxis, :] * (omega_sr**(-1))[:, np.newaxis] * (1.0 - Om_dot_rhat)[:, np.newaxis]

    transfer_func = np.sinc(transfer_func_arg / np.pi) * np.exp(transfer_func_arg*1j)

    one_arm_detect_tensor = 0.5 * rhat_cross[:, np.newaxis, :, :] * transfer_func[:, :, np.newaxis, np.newaxis]
    
    e_plus, e_cross = basis_tensors(theta, phi)

    H_freq = H_plus[np.newaxis, :, np.newaxis, np.newaxis] * e_plus[np.newaxis, np.newaxis, :, :] + \
        H_cross[np.newaxis, :, np.newaxis, np.newaxis] * e_cross[np.newaxis, np.newaxis, :, :] 
    
    integrand = np.exp(1j*omega[np.newaxis, :]*xi_s[:, np.newaxis]) * np.einsum('twxX,twxX->tw',one_arm_detect_tensor, H_freq)  * np.exp(-1j * omega[np.newaxis, :] * times[0])
    
    dl_sr = c*l_sr* (np.sum(integrand, axis=1)/(len_times)) 
    return dl_sr


def one_arm_response_wdm(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500):
    """
    WDM LISA Response 

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
    plt.plot(times, td_waveform_plus)
    plt.show()


    current_length = len(times)
    print(f"Current length of time series: {current_length}")

    dt = np.mean(np.diff(times))

    N = WDM.utils.next_multiple(current_length, 2*Nf)

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=1, calc_m0=True)
    print(wdm)

    h_plus_padded, mask = WDM.utils.pad_signal(td_waveform_plus, N)

    print(f"Padded length of time series: {N}")

    print((N-current_length)*wdm.dt)

    w_t = wdm.dwt(h_plus_padded)

    fig, ax = WDM.plotting.time_frequency_plot(wdm, w_t, part='real', scale='linear')
    plt.show()

    #basic implementation of wdm transform 

    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr = _compute_ltt_iterative(times, s, r, orbits)
        
    omega_sr = (l_sr)**(-1)  
        
    rhat_sr = rhat(times, s, r, orbits)

    rhat_cross = np.einsum('tx,tX->txX', rhat_sr, rhat_sr)

    _, _, n = basis_vectors(theta, phi)

    Omega = -n 

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)
        

    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)

    freq_vals = np.arange(0, wdm.Nf) * wdm.dF
        
    transfer_func_arg = 0.5 * 2.0 * np.pi * freq_vals[np.newaxis, :] * (omega_sr**(-1))[:, np.newaxis] * (1.0 - Om_dot_rhat)[:, np.newaxis]

    transfer_func = np.sinc(transfer_func_arg / np.pi) * np.exp(transfer_func_arg*1j) #shape time x N_f
        
    e_plus, _ = basis_tensors(theta, phi)

    Nt = wdm.Nt   # number of time bins
    Nf = wdm.Nf   # number of frequency bins

    # Create a grid of (n, m)
    n_grid, m_grid = np.meshgrid(np.arange(Nt), np.arange(Nf), indexing="ij")

    vals = gnm_at_times_grid_np(wdm, n_grid, m_grid, xi_s-times[0])

    sum_nm = (transfer_func[:,np.newaxis, :]+np.conj(transfer_func[:,np.newaxis, :])) * w_t[np.newaxis, :, :] * vals

    dl_sr_wdm = np.pi *c * l_sr *np.einsum('tXx,Xx-> t', rhat_cross, e_plus) * np.einsum("tnm->t", sum_nm)

    return dl_sr_wdm