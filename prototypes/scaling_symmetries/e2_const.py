from common import *
Nf,N=64,2**16   # dT=256 s, Nt=1024 -> r multiple of 1/512 on the df grid
for P in (128,2049):
  print(f"--- constant delay, Nf={Nf}, dT=256 s, L=25, P={P}; rows: m, cols: (r, delta/dT)")
  rs=(0.0,0.25,0.40625); ds=(0.0273,0.3,0.4609)
  print("m   "+"  ".join(f"r={r:.3f},d={d:.3f}" for r in rs for d in ds))
  for m in (1,2,3,4,5,10,21,32,50,61,62):
    print(f"{m:<3d} "+"  ".join(f"{shift_err(Nf,N,m,r,lambda t,d=d:0*t+d*256.0,P=P):15.3e}" for r in rs for d in ds))
