from common import *
N=2**16; T=N*DT
res=[]
print("sinusoidal delay A sin(2 pi t/T + 0.4); M=L2^2/2; L=40,P=2049. eps = m*rate_max")
print("Nf   dT    m   r        rate_max  eps=m*rate   f[mHz]   M          M/eps^2")
for r in (0.40625, 0.25, 0.0):
  for Nf in (32,64,128):
    dT=DT*Nf
    for rate in (1e-4,4e-4,1.6e-3):
        A=rate*T/(2*np.pi)
        fn=lambda t,A=A: A*np.sin(2*np.pi*t/T+0.4)
        for m in (4,8,16,24):
            if m>=Nf-2: continue
            e=shift_err(Nf,N,m,r,fn,L=40,P=2049); Mm=e*e/2
            eps=m*rate; f=(m+r)/(2*dT)
            res.append((r,Nf,m,rate,eps,Mm))
            print(f"{Nf:<4d} {dT:<5.0f} {m:<3d} {r:<8.4f} {rate:<9.1e} {eps:<11.2e} {1e3*f:<8.3f} {Mm:<10.3e} {Mm/eps**2:.3e}")
np.save('e3.npy',np.array(res))
