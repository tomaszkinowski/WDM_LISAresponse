from common import *
Nf,N=64,2**16; dT=256.0
def err2(m,r,d,phi,L=25,P=2049):
    return shift_err(Nf,N,m,r,lambda t:0*t+d*dT,L=L,P=P,phi=phi)
rs=(0.0,0.25,0.40625); ds=(0.0273,0.3,0.4609)
print("M = L2^2/2, phase-averaged (phi, phi+pi/2); P=2049, L=25")
print("m   "+"  ".join(f"r={r:.3f},d={d:.3f}" for r in rs for d in ds))
for m in (1,2,3,4,5,10,21,32,50,61,62):
    row=[]
    for r in rs:
        for d in ds:
            e=[err2(m,r,d,p) for p in (0.3,0.3+np.pi/2)]
            row.append(0.5*(e[0]**2+e[1]**2)/2)
    print(f"{m:<3d} "+"  ".join(f"{v:15.3e}" for v in row))
