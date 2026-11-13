import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
C=str(ROOT/'plotting/cache')+'/'
for kind in ['_production','']:
    d=np.load(C+f'fig10_mismatch_map{kind}.npz')
    Nf=d['Nf']; f=d['f']; M=d['mismatch']; r=d['r']; dT=4.0*Nf
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for j,ir in enumerate([3,4,5]):
        for row,xname in enumerate(['f','2 f dT  (= m + r)']):
            a=ax[row,j]
            for k in range(len(Nf)):
                x = f[:,ir,k] if row==0 else 2*f[:,ir,k]*dT[k]
                a.loglog(x, M[:,ir,k], '.-', color=plt.cm.viridis(k/len(Nf)), label=f'{dT[k]:.0e}' if j==0 and row==0 else None)
            a.set_title(f'{kind or "split_amp_sinc_first"} r={r[ir]:.3f}'); a.set_xlabel(xname); a.grid(alpha=.3)
    ax[0,0].legend(fontsize=6,ncol=2)
    fig.tight_layout(); fig.savefig(str(ROOT)+f'/plotting/figures/investigations/scaling_collapse{kind or "_split"}.png',dpi=80)
