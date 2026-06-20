import numpy as np, os
import analytical_solutions as A
from analytical_solutions import normal_force_linear as nfl

R=1.0; density=2500; mass=4/3*np.pi*R**3*density
kn=1.0e8; etan=1.0e6; g=-9.81

def load(p):
    r=[]
    for l in open(p):
        s=l.strip()
        if not s or s.startswith('#'): continue
        c=s.split()
        if len(c)<13: continue
        try: r.append([float(x) for x in c[:13]])
        except ValueError: continue
    return np.array(r)

def apex(z, st):
    return [(st[i]*1e-4, z[i]) for i in range(1,len(z)-1)
            if z[i]>z[i-1] and z[i]>=z[i+1] and z[i]>1.0]

def bounce(nf, n=50000, dt=1e-4):
    z=1.5; v=0.0; Z=np.zeros(n); Z[0]=z
    for i in range(1,n):
        vn=-v
        fz = mass*g if z>R else nf(R-z,vn,R,kn,etan)+mass*g
        a=fz/mass; v+=a*dt; z+=v*dt+0.5*a*dt*dt; Z[i]=z
    return Z

def e_from_apex(pk):
    if len(pk)<2: return float('nan')
    h1=pk[0][1]-R; h2=pk[1][1]-R
    return (h2/h1)**0.5 if h1>0 else float('nan')

ana = bounce(A.normal_force_linear)
stf = np.arange(50000)
ap_ana = apex(ana, stf)
print("analytical (force-clip h_eff) apexes:", [f"{z:.4f}@{t:.3f}" for t,z in ap_ana][:6])
print(f"analytical restitution: {e_from_apex(ap_ana):.3f}\n")

for fn in ['dump_grain_bounce_100.txt','dump_grain_bounce_400.txt']:
    if not os.path.exists(fn):
        print(f"[missing] {fn}"); continue
    s=load(fn); z=s[:,3]; vz=s[:,6]; fz=s[:,9]; st=s[:,0]
    ap=apex(z,st)
    N=fn.split('_')[-1].split('.')[0]
    print(f"--- {fn} (N={N}) ---")
    print("  sim apexes:", [f"{zz:.4f}@{tt:.3f}" for tt,zz in ap][:6])
    print(f"  sim restitution: {e_from_apex(ap):.3f}")
    # per-overlap force factor at max compression of first contact
    fc=fz-mass*g
    incon=(np.abs(fc)>100)&(z<1.05); incon[0]=False
    idx=np.where(incon)[0]
    if len(idx):
        seg=idx[idx<idx[0]+400]
        j=seg[np.argmin(np.abs(vz[seg]))]   # closest to vz=0 = max compression
        h=R-z[j]; law=nfl(h,-vz[j],R,kn,etan)
        print(f"  max-compression: h={h:.4f} vz={vz[j]:+.3f} fz_sim={fc[j]:.0f} "
              f"analytical={law:.0f} ratio={fc[j]/law:.3f}")
    print()
