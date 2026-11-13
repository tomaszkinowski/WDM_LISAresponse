import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
import WDM
DT=4.0
_cache={}
def wdm(Nf,N,L=25,P=128):
    k=(Nf,N,L,P)
    if k not in _cache:
        w=WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=DT,Nf=Nf,N=N,q=16)
        tab=w.build_time_delay_filter_interpolants(L,P)
        _cache[k]=(w,tab)
    return _cache[k]
def tone(w,f,delta_t=None,phi=0.3):
    t=np.asarray(w.times)
    d=0 if delta_t is None else delta_t(t)
    return np.cos(2*np.pi*f*(t+d)+phi)
def shift_err(Nf,N,m,r,delta_fn,L=25,P=128,phi=0.3):
    w,tab=wdm(Nf,N,L,P)
    f=(m+r)*float(w.dF)
    x=tone(w,f,phi=phi); y=tone(w,f,delta_fn,phi=phi)
    W=w.dwt(jnp.asarray(x)); Wex=np.asarray(w.dwt(jnp.asarray(y)))
    tn=np.arange(w.Nt)*float(w.dT)
    Wop=np.asarray(w.apply_variable_time_shift(tab,W,jnp.asarray(delta_fn(tn))))
    num=np.linalg.norm(Wop-Wex); return num/np.linalg.norm(Wex)
