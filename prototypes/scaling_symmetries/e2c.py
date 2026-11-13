from common import *
Nf,N=64,2**16; dT=256.0
M=lambda e: e*e/2
def E(m,r,d,L=25,P=2049): return M(shift_err(Nf,N,m,r,lambda t:0*t+d*dT,L=L,P=P))
print("(i) integer part of delta/dT (fold): m=10, r=0.40625; psi = m*d mod 1 kept fixed by m*k even")
for d in (0.3,1.3,2.3,5.3,10.3,20.3):
    print(f"  d={d:5.1f}  M={E(10,0.40625,d):.3e}   (L=25)")
print("(ii) lags needed vs delta/dT, r=0.40625, m=10 (psi varies; take max over m in 10..13)")
print("  d       "+"  ".join(f"L={L:<6d}" for L in (2,4,6,8,12,16,25,40)))
for d in (0.001,0.004,0.016,0.0625,0.25,0.5):
    vals=[max(E(m,0.40625,d,L=L) for m in (10,11,12,13)) for L in (2,4,6,8,12,16,25,40)]
    print(f"  {d:<7.3f} "+"  ".join(f"{v:.1e}" for v in vals))
print("(iii) endpoints, d=0.3, M for each (m,r)")
for m,r in ((1,-0.40625),(1,-0.25),(1,0.0),(2,-0.40625),(62,0.40625),(63,-0.25),(63,0.0),(63,0.25)):
    print(f"  m={m:2d} r={r:+.3f} M={E(m,r,0.3):.3e}")
