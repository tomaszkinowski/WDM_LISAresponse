from common import *
from WDM.code.time_delay_filters.filters import build_filter_tables
x=np.linspace(-6,6,241)          # x = l - delta/dT
out={}
for Nf,N in ((32,2**14),(64,2**16),(250,250*2124),(250,250*2126),(1000,1000*512)):
    w=WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=DT,Nf=Nf,N=N,q=16)
    dT=float(w.dT); dF=float(w.dF)
    # represent each x as lag l=round(x), delta=(l-x) dT
    l=np.round(x).astype(int); d=(l-x)*dT
    tab=np.asarray(build_filter_tables(jnp.arange(-7,8), jnp.asarray(np.unique(d)/dT), w.d, w.A_frac))
    ud=np.unique(d); T=np.array([tab[:, li+7, np.searchsorted(ud,di)] for li,di in zip(l,d)])
    out[(Nf,N)]=T
    print(f"Nf={Nf:5d} Nt={N//Nf:5d} dT={dT:6.0f}: tau(0)={T[120,1]:.15f} tau'(0)={T[120,0]:.3e}")
ref=out[(1000,1000*512)]
for k,T in out.items():
    print(k, "max|T/dF - ref| (T, T'):", np.abs(T[:,1]-ref[:,1]).max(), np.abs(T[:,0]-ref[:,0]).max())
