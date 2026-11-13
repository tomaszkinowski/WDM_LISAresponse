"""Which table makes the shift exact? Constant delay on a table node, so linear
interpolation adds nothing. (a) periodic tone, spectral shift == exact for the
periodic record. (b) compact Gaussian-windowed tone in the middle of the record,
shifted analytically, zero beyond: the non-periodic (padded) situation."""
import numpy as np
from common import *
from continuous_tables import build_tables_continuous
P=129
for dt,Nf,N in ((0.5,8,512),(4.0,50,50*124),(4.0,50,50*126),(4.0,64,2**14)):
    w=WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=dt,Nf=Nf,N=N,q=4 if Nf==8 else 16)
    Nt,dT,dF=w.Nt,float(w.dT),float(w.dF); t=np.asarray(w.times); T=N*dt
    for L in sorted({min(25,Nt//2-1), Nt//2-1}):
        D=w.build_time_delay_filter_interpolants(L,P); Cc=jnp.asarray(build_tables_continuous(L,P))
        delta=dT*(-0.5+40/(P-1))                  # a table node
        m=Nf//3; f=(m+0.40625)*dF; f=round(f*T)/T  # on the df grid
        cases={'periodic tone':(np.cos(2*np.pi*f*t+0.3), np.cos(2*np.pi*f*(t+delta)+0.3))}
        sig=T/12; g=lambda s: np.exp(-0.5*((s-T/2)/sig)**2)*np.cos(2*np.pi*f*s+0.3)
        cases['compact burst']=(g(t), g(t+delta))
        for name,(x,y) in cases.items():
            W=w.dwt(jnp.asarray(x)); Wex=np.asarray(w.dwt(jnp.asarray(y))); dn=jnp.full(Nt,delta)
            errs=[np.linalg.norm(np.asarray(w.apply_variable_time_shift(tb,W,dn))-Wex)/np.linalg.norm(Wex) for tb in (D,Cc)]
            print(f"Nt={Nt:<5d} L={L:<3d} {name:<14s} L2 err: discrete {errs[0]:.2e}  continuous {errs[1]:.2e}")
