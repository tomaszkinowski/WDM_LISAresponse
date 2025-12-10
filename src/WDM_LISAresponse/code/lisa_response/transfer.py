import jax
import jax.numpy as jnp

from jaxtyping import Array
from jax import debug as jax_debug


def _check_n_unit_vec(n):
    """
    For debugging: check that n, with shape=(3,...), is a unit vector in 3D.
    """
    assert n.shape[0] == 3, \
        f"n must be a vector in 3D: {n.shape=}"
    assert jnp.allclose(jnp.linalg.norm(n, axis=0), 1.0), \
        f"n must be a unit vector: {jnp.linalg.norm(n, axis=0)=}"

def _check_something():
    """
    For debugging:
    """
    assert True, "This is a test assertion."


@jax.jit(static_argnames=("debug",))
def transfer_functions(omega: Array,
                       time: Array,
                       n: Array, 
                       *,
                       debug: bool=True) -> Array:
    r"""
    The transfer functions for the one-arm LISA response for a laser travelling
    from spacecraft :math:`i` to spacecraft :math:`j` is

    .. math::
        T_{ij}(\omega, t, \hat{n}) = 
                    \sinc\left(\frac{1}{2}\omega\ell_{ij}(t)(1+\hat{n}\cdot
                            \hat{r}_{ij}(t))\right) 
                    \exp\left(\frac{i}{2}\omega\ell_{ij}(t)(1+\hat{n}\cdot
                            \hat{r}_{ij}(t))\right) .

    Here :math:`\ell_{ij}(t)` is the light-travel time between the spacecraft
    and :math:`\hat{r}_{ij}(t)` is the unit vector pointing from spacecraft
    :math:`i` (at the time :math:`t` the signal is sent) to spacecraft :math:`j` 
    (at the time :math:`t+\ell_{ij}(t)` the signal is received). The GW comes 
    from a source in direction :math:`\hat{n}`.

    Parameters
    ----------
    omega : Array
        The angular frequency :math:`\omega=2\pi f` of the GW [rad/s]. At least 
        a 1D array.
    time : Array
        The time [s]. This is the time the signal leaves spacecraft :math:`i`. 
        At least a 1D array.
    n: Array
        The unit vector pointing to the GW source, Cartesian coordinates,
        shape=(3,) or (3, ...). This must be a unit vector, i.e. normalised.
    debug: bool, keyword-only
        If True, run assertion checks on function arguments. Default is True.

    Returns
    -------
    transfer_funcs: Array
        The one-arm transfer functions, shape=(6,*omega.shape,*time.shape,...).
        The first axis labels the 6 one-are links :math:`i \to j` in the order 
        :math:`\{1\to 2, 2\to 3, 3\to 1, 1\to 3, 3\to 2, 2\to 1\}`. 
        The second and third axes correspond to the frequency and time 
        respectively.
        The remaining axes correspond to the shape of :math:`\hat{n}` allowing
        the transfer function to be vectorised over source sky locations.

    Notes
    -----
    - The sinc function above is defined as :math:`\sinc(x) = \frac{\sin(x)}{x}` 
    (for :math:`x\neq 0`) which differs the `numpy` and `jax.numpy` conventions.
    - See Eq.21 of Cornish & Rubbo (2003) https://arxiv.org/pdf/gr-qc/0209011
    """
    omega = jnp.atleast_1d(omega)
    time = jnp.atleast_1d(time)
    n = jnp.atleast_1d(n)

    if debug: 
        jax_debug.callback(_check_n_unit_vec, n)
        jax_debug.callback(_check_something)

    # light_travel_time.shape = (*time.shape, 6) for the 6 laser links
    light_travel_time = jnp.ones((*time.shape, 6)) 

    # arm_vectors.shape = (*time.shape, 6, 3) for 6 links and 3 spatial dims
    arm_vectors = jnp.ones((*time.shape, 6, 3))

    # n_dot_r.shape = (*n.shape[1:], *time.shape, 6) for the remaining 
    # dimensions of n, time and 6 laser links
    n_dot_r = jnp.tensordot(n, arm_vectors, axes=(0, -1))

    # transfer_funcs.shape = (6, *omega.shape, *time.shape, *n.shape[1:])
    arg = 0.5 * omega * light_travel_time #* ( 1.0 + n_dot_r )
    transfer_funcs = jnp.sinc(arg/jnp.pi) * jnp.exp(1j*arg)

    return transfer_funcs


@jax.jit(static_argnames=("debug",))
def detector_tensors(omega: Array,
                     time: Array,
                     n: Array, 
                     orbits: callable,
                     *,
                     debug: bool=True) -> Array:
    r"""
    The detector tensor for the one-arm LISA response for a laser travelling
    from spacecraft :math:`i` to spacecraft :math:`j` is

    .. math::
        \mathbf{D}_{ij}(\omega, t, \hat{n}) = \frac{1}{2} \hat{r}_{ij}(t) 
                        \otimes \hat{r}_{ij}(t) T_{ij}(\omega, t, \hat{n}).

    Here :math:`T_{ij}(t)` is the transfer function and :math:`\hat{r}_{ij}(t)`
    is the unit vector pointing from spacecraft :math:`i` to spacecraft 
    :math:`j`. The GW comes from a source in the direction :math:`\hat{n}`.
    
    Parameters
    ----------
    omega : Array
        The angular frequency :math:`\omega=2\pi f` of the GW [rad/s]. At least
        a 1D array.
    time : Array
        The time [s]. This is the time the signal leaves spacecraft :math:`i`. 
        At least a 1D array.
    n: Array
        The unit vector pointing to the GW source, Cartesian coordinates,
        shape=(3,) or (3, ...). This must be a unit vector, i.e. normalised.
    orbits: callable
        The LISA orbit model. This is used to compute the arm unit vectors
        :math:`\hat{r}_{ij}(t)` and light-travel times :math:`\ell_{ij}(t)` 
        between spacecraft.
    debug: bool, keyword-only
        If True, run assertion checks on function arguments. Default is True.

    Returns
    -------
    detector_tensors: Array
        The one-arm detector tensors, 
        shape=(6, 3, 3, *omega.shape, *time.shape, ...).
        The first axis labels the 6 one-are links :math:`i \to j` in the order 
        :math:`\{1\to 2, 2\to 3, 3\to 1, 1\to 3, 3\to 2, 2\to 1\}`.
        The second and third axes label the Cartesian components of the
        detector tensor; i.e. the outer product :math:`\hat{r}\otimes\hat{r}`. 
        The fourth and fifth axes correspond to the frequency and time 
        respectively.
        The remaining axes correspond to the shape of :math:`\hat{n}` allowing
        the transfer function to be vectorised over source sky locations.

    Notes
    -----
    - See Eq.20 of Cornish & Rubbo (2003) https://arxiv.org/pdf/gr-qc/0209011
    """
    omega = jnp.atleast_1d(omega)
    time = jnp.atleast_1d(time)
    n = jnp.atleast_1d(n)

    if debug: 
        jax_debug.callback(_check_n_unit_vec, n)

    # should compute arm_vectors here and pass into transfer_functions
    # ...maybe light_travel_time too?
    T = transfer_functions(omega, time, n, debug=debug)

    # arm_vectors.shape = (*time.shape, 6, 3) for 6 links and 3 spatial dims
    arm_vectors = jnp.ones((*time.shape, 6, 3))

    # detector_tensors.shape = (*time.shape, 6, 3, 3)
    detector_tensors = 0.5 * jnp.einsum('...i,...j->...ij',
                                        arm_vectors, arm_vectors)
    
    return detector_tensors

