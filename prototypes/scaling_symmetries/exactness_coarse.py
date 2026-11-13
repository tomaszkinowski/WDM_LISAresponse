"""Few rows (Nt <= 2L), as at the coarsest paper tilings. Constant delay on a
table node; periodic tone and compact burst; L = 25 vs L = Nt/2 - 1."""
import numpy as np
from common import *
from continuous_tables import build_tables_continuous
P=129
for Nf,Nt in ((2048,32),(2048,40),(2048,50),(2048,64)):
    N=Nf*Nt; w=WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=4.0,Nf=Nf,N=N,q=16)
    dT,dF=float(w.dT),float(w.dF); t=np.asarray(w.times); T=N*4.0
    delta=dT*(-0.5+40/(P-1)); f=(500+0.40625)*dF; f=round(f*T)/T
    sig=T/12; g=lambda s: np.exp(-0.5*((s-T/2)/sig)**2)*np.cos(2*np.pi*f*s+0.3)
    for L in (25, Nt//2-1):
        D=w.build_time_delay_filter_interpolants(L,P); Cc=jnp.asarray(build_tables_continuous(L,P))
        for name,(x,y) in {'periodic tone':(np.cos(2*np.pi*f*t+0.3),np.cos(2*np.pi*f*(t+delta)+0.3)),'compact burst':(g(t),g(t+delta))}.items():
            W=w.dwt(jnp.asarray(x)); Wex=np.asarray(w.dwt(jnp.asarray(y))); dn=jnp.full(Nt,delta)
            e=[np.linalg.norm(np.asarray(w.apply_variable_time_shift(tb,W,dn))-Wex)/np.linalg.norm(Wex) for tb in (D,Cc)]
            print(f"Nt={Nt:<3d} L={L:<3d} {name:<14s} discrete {e[0]:.2e}  continuous {e[1]:.2e}")
