import sys; sys.path.insert(0,'.'); sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
import numpy as np
import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import GC_THETA, GC_PHI, DT, N_SIGNAL
from cr import Phi
orbits,start,_,link=fig14.setup()
t=start+np.linspace(0,DT*N_SIGNAL,40000); c=299792458.0
th,ph=GC_THETA,GC_PHI
u=np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),-np.sin(th)]); v=np.array([np.sin(ph),-np.cos(ph),0]); n=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
x1=orbits.compute_position(t,[1])[:,0,:]; x2=orbits.compute_position(t,[2])[:,0,:]
d=x2-x1; ell=np.linalg.norm(d,axis=1)/c; l=d/np.linalg.norm(d,axis=1)[:,None]
Fp=(l@u)**2-(l@v)**2; Fx=2*(l@u)*(l@v); tau=ell*(1+l@n)
rate=orbits.compute_velocity(t,[2])[:,0,:]@n/c
def window_factors(r,h=1e-6):
    ks=np.array([0,1,-1]); rr=r-ks
    P=Phi(rr); dP=(Phi(rr+h)-Phi(rr-h))/(2*h)
    return np.sum(dP**2), np.sum(P**2*rr**2)      # S(r), R(r)
def predict(f,dT,r,sgn):
    dF=1/(2*dT)
    A=ell*(Fp+sgn*1j*Fx)*np.sinc(f*tau)
    w=np.abs(A)**2
    dlogA=np.gradient(A,t)/A                       # (Adot/A) complex
    nu=f*rate + dlogA/(2j*np.pi)                   # local complex frequency offset [Hz]
    x=f*tau; dsinc=np.gradient(np.sinc(np.linspace(0,1,1)),1) if False else None
    g=tau*np.where(np.abs(x)>1e-12,(np.pi*x*np.cos(np.pi*x)-np.sin(np.pi*x))/(np.pi*x*np.sin(np.pi*x)+1e-300)*np.pi* (x!=0),0)  # tau * dln sinc/dx
    S,R=window_factors(abs(r))
    ks=np.array([0,1,-1])[:,None]; eps=np.real(nu)[None,:]/dF
    MD=0.5*np.sum(w*np.sum((Phi(r-ks+eps)-Phi(r-ks))**2,0))/np.sum(w)
    Mrate=MD+0.5*S*np.sum(w*np.imag(nu)**2)/np.sum(w)/dF**2
    Msinc=0.5*R*dF**2*np.sum(w*g**2)/np.sum(w)
    return Mrate,Msinc
C='cache/fig10_mismatch_map_production.npz'
D=np.load(C); Nf=D['Nf']; F=D['f']; M=D['mismatch']; rs=D['r']; dTs=DT*Nf
for sgn in (-1,):
  print(f"=== sign {sgn:+d}: log10(M_meas/M_pred), floor 2e-10 added to prediction; rows dT, cols f-cells; '.'=blank")
  for ir in (5,4,3,2,1,0):
    print(f" r={rs[ir]:+.3f}")
    for k in range(len(Nf)):
        if dTs[k]>4e5: continue
        out=[]
        for j in range(F.shape[0]):
            f=F[j,ir,k]
            if not np.isfinite(f): out.append('   . '); continue
            a,b=predict(f,dTs[k],rs[ir],sgn); p=a+b+2e-10
            out.append(f"{np.log10(M[j,ir,k]/p):+5.1f}")
        print(f"  dT={dTs[k]:8.0f} "+" ".join(out))
  if sgn==1:
    # show which term dominates the prediction at r=5/12
    pass
