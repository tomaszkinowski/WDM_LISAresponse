import sys; sys.path.insert(0,'.'); sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from scipy.integrate import quad
import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import VIRGO_THETA as th, VIRGO_PHI as ph, DT, N_SIGNAL
from cr import Phi
h=1e-6; dPhi=lambda x:(Phi(x+h)-Phi(x-h))/(2*h)
Sbar=quad(lambda x:dPhi(x)**2,-1,1,points=[-0.75,-0.25,0.25,0.75],limit=200)[0]
print(f"<S>_r = int Phi'^2 dx = {Sbar:.4f}  (bin units)")
e=np.load('cache/emri_waveform.npz'); hp=e['h_plus']; hc=e['h_cross']
P=np.abs(np.fft.rfft(hp))**2+np.abs(np.fft.rfft(hc))**2; fr=np.fft.rfftfreq(hp.size,DT)
f2=np.sum(P*fr**2)/np.sum(P); print(f"EMRI power-weighted rms f = {np.sqrt(f2)*1e3:.3f} mHz")
orbits,start,_,link=fig14.setup(); c=299792458.0
t=start+np.linspace(0,DT*N_SIGNAL,40000)
u=np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),-np.sin(th)]); v=np.array([np.sin(ph),-np.cos(ph),0]); n=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
x1=orbits.compute_position(t,[1])[:,0,:]; x2=orbits.compute_position(t,[2])[:,0,:]
l=(x2-x1)/np.linalg.norm(x2-x1,axis=1)[:,None]
A=(l@u)**2-(l@v)**2-1j*2*(l@u)*(l@v); w=np.abs(A)**2
rate=orbits.compute_velocity(t,[2])[:,0,:]@n/c
r2=np.sum(w*rate**2)/np.sum(w); mod2=np.sum(w*np.abs(np.gradient(A,t)/A)**2)/np.sum(w)
d=np.load('cache/investigation_fig8_production.npz'); dT=DT*d['Nf']; M=d['EMRI']
print(" dT[s]     M_meas     M_dopp     M_mod      M_pred    meas/pred")
for k in range(len(dT)):
    Md=0.5*Sbar*(2*dT[k])**2*f2*r2
    Mm=0.5*Sbar*(2*dT[k])**2*mod2/(2*np.pi)**2
    print(f"{dT[k]:8.0f}  {M[k]:.2e}  {Md:.2e}  {Mm:.2e}  {Md+Mm:.2e}  {M[k]/(Md+Mm):.2f}")
