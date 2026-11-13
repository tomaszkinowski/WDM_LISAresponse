from common import *
Nf=24; N=24*2730; dT=96.0
for L in (25,15):
  for m,r in ((10,0.4),(10,0.0),(18,0.4)):
    a=shift_err(Nf,N,m,r,lambda t:0*t+20.0,L=L,P=2049); b=shift_err(Nf,N,m,r,lambda t:0*t+500.0,L=L,P=2049)
    print(f"L={L} dT=96 m={m} r={r}: delta=20 s M={a*a/2:.2e}   delta=500 s (=20+5 dT) M={b*b/2:.2e}")
