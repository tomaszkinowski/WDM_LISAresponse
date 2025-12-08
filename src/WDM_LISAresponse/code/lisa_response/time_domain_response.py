#load the appropriate libraries 
import numpy as np
import matplotlib.pyplot as plt

from lisaorbits.file import read_orbit_file
from lisaorbits import InterpolatedOrbits
from lisaconstants import SUN_SCHWARZSCHILD_RADIUS, c, ASTRONOMICAL_YEAR
from numpy.linalg import norm
from scipy.fft import fft, fftfreq

year_SI = 3.154e7

def basis_vectors(theta, phi):
    """
    Parameters
    ----------
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]

    Returns
    -------
    u : ndarray
        North/South basis vector, shape=(3,)
    v : ndarray
        East/West basis vector, shape=(3,)
    n : ndarray
        Radial basis vector, shape=(3,)
    """
    u = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), - np.sin(theta)])
    v = np.array([np.sin(phi), -np.cos(phi), 0])
    n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return u, v, n

def basis_tensors(theta, phi):
    """
    Parameters
    ----------
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]

    Returns
    -------
    e_plus : ndarray
        plus polarised tensor, shape=(3,3)
    e_cross : ndarray
        cross polarised tensor, shape=(3,3)
    """
    u, v, n = basis_vectors(theta, phi)
    e_plus = np.tensordot(u, u, axes=0) - np.tensordot(v, v, axes=0)
    e_cross = np.tensordot(u, v, axes=0) + np.tensordot(v, u, axes=0)
    return e_plus, e_cross

def waveform_polarisations(time, A_plus, A_cross, f, phi_0):
    """
    This is for a sine wave signal.

    Parameters
    ----------
    time : ndarray
        time in the solar system barycenter [s]
    A_plus : float
        plus polarisation amplitude [strain]
    A_cross : float
        cross polarisation amplitude [strain]
    f : float
        GW frequency [Hz]
    phi_0 : float
        phase angle [rad]

    Returns
    -------
    h_plus : ndarray
        plus polarised signal, shaped like time.
    h_cross : ndarray
        cross polarised signal, shaped like time.
    """
    h_plus = A_plus * np.cos(2*np.pi*f*time+phi_0)
    h_cross = A_cross * np.sin(2*np.pi*f*time+phi_0)
    return h_plus, h_cross
