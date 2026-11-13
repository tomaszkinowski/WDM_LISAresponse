import time, numpy as np
from common import *
from continuous_tables import build_tables_continuous
L,P=25,128
t0=time.perf_counter(); C64=build_tables_continuous(L,P,q=64); tc=time.perf_counter()-t0
C128=build_tables_continuous(L,P,q=128)
print(f"continuous build L={L} P={P}: {tc*1e3:.1f} ms; quadrature q=64 vs 128: {np.abs(C64-C128).max():.1e}")
print(f"{'grid':<34}{'Nt':>7}  {'max|T_disc-T_cont|':>19} {'max|Tp_disc-Tp_cont|':>21}  {'disc build':>10}")
for dt,Nf,N in ((0.5,8,256),(0.5,8,512),(4.0,50,50*124),(4.0,50,50*126),(4.0,64,2**16),(4.0,64,2**18),(4.0,250,250*33582),(4.0,2500,2500*3358)):
    w=WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=dt,Nf=Nf,N=N,q=4 if Nf==8 else 16)
    LL=min(L, w.Nt-1)
    t0=time.perf_counter(); D=np.asarray(w.build_time_delay_filter_interpolants(LL,P)); D.sum(); td=time.perf_counter()-t0
    Cc=build_tables_continuous(LL,P)
    e=np.abs(D-Cc).max(axis=(1,2))
    print(f"dt={dt} Nf={Nf:<5d} N={N:<11d}{w.Nt:>7d}  {e[1]:>19.2e} {e[0]:>21.2e}  {td:>9.2f}s")
