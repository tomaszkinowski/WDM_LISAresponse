import sys; sys.path.insert(0,'.')
import numpy as np
import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import GC_THETA, GC_PHI, DT, N_SIGNAL
orbits,start,_,link=fig14.setup()
t=start+np.linspace(0,DT*N_SIGNAL,20000)
n=np.array([np.sin(GC_THETA)*np.cos(GC_PHI),np.sin(GC_THETA)*np.sin(GC_PHI),np.cos(GC_THETA)])
c=299792458.0
for sc in (1,2,3):
    v=orbits.compute_velocity(t,[sc])[:,0,:]
    rate=v@n/c
    x=orbits.compute_position(t,[sc])[:,0,:]; xc=np.mean(orbits.compute_position(t,[1,2,3]),axis=1)
    rest=(x-xc)@n/c
    print(f"SC{sc}: rate rms {np.sqrt(np.mean(rate**2)):.3e}  max {np.abs(rate).max():.3e} | rest (x-x_c).n/c: range {rest.min():.2f}..{rest.max():.2f} s, rate rms {np.sqrt(np.mean(np.gradient(rest,t)**2)):.2e}")
x1=orbits.compute_position(t,[1])[:,0,:]; x2=orbits.compute_position(t,[2])[:,0,:]
l=(x2-x1); l/=np.linalg.norm(l,axis=1)[:,None]
w=(1-(l@n)**2)**2
for sc in (1,2):
    rate=orbits.compute_velocity(t,[sc])[:,0,:]@n/c
    print(f"receiver SC{sc}: power-weighted rms rate {np.sqrt(np.sum(w*rate**2)/np.sum(w)):.3e}  (weight min/max {w.min():.3f}/{w.max():.3f})")
