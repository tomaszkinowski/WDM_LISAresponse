import numpy as np
from scipy.special import betainc
def Phi(r, d=4, A=0.25):           # Meyer window in units of dF (flat |r|<A, zero beyond 1-A)
    x=np.clip((np.abs(r)-A)/(1-2*A),0,1)
    return np.cos(0.5*np.pi*betainc(d,d,x))
def c(r,h=1e-6):
    rs=np.array([r, r-1, r+1])
    P=Phi(rs); dP=(Phi(rs+h)-Phi(rs-h))/(2*h)
    return 0.25*np.sum(dP**2)/np.sum(P**2)
if __name__=='__main__':
    for r in (0.0,0.083333,0.125,0.2,0.25,0.3,0.35,0.40625,0.416667,0.45,0.5):
        print(f"r={r:.4f}  c(r)={c(r):.4f}")
    print("check sum Phi^2 (partition of unity):", [round(float(np.sum(Phi(np.array([r,r-1,r+1]))**2)),12) for r in (0.1,0.4)])
