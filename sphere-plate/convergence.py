import numpy as np, glob, re
from analytical_solutions import analytical_solution, normal_force_linear

DT=1e-4; R=1.0; dens=2500; MASS=4/3*np.pi*R**3*dens; KN=1e8; ETAN=1e6; G=-9.81
CASES={1:'bounce',2:'tangent',3:'shear',4:'twist',5:'roll'}

def load(p):
    rows=[]
    for l in open(p):
        s=l.strip()
        if not s or s.startswith('#'): continue
        c=s.split()
        if len(c)<13: continue
        try: rows.append([float(x) for x in c[:13]])
        except ValueError: continue
    return np.array(rows)

def dumps(base):
    d={}
    for p in glob.glob(f'dump_grain_{base}_*.txt'):
        m=re.search(r'_0*(\d+)\.txt$',p)
        if m: d[int(m.group(1))]=p
    return d

def fit(Ns, errs):
    Ns=np.array(Ns,float); errs=np.array(errs,float)
    ok=errs>0
    if ok.sum()<2: return float('nan')
    return np.polyfit(np.log(Ns[ok]), np.log(errs[ok]), 1)[0]

# case 1: relative error of the FIRST-bounce apex height vs analytical
def bounce(n):
    z=1.5;v=0.;Z=np.zeros(n);Z[0]=z
    for i in range(1,n):
        vn=-v; fz=MASS*G if z>R else normal_force_linear(R-z,vn,R,KN,ETAN)+MASS*G
        a=fz/MASS;v+=a*DT;z+=v*DT+.5*a*DT*DT;Z[i]=z
    return Z
def apex1(z,st):
    for i in range(1,len(z)-1):
        if z[i]>z[i-1] and z[i]>=z[i+1] and z[i]>R: return z[i]
    return np.nan

print("Convergence of steady-state relative error  err = |sim-ana|/|ana|,  slope p in err ~ N^p")
for cf,base in CASES.items():
    d=dumps(base); Ns=sorted(d)
    if not Ns:
        print(f"case {cf} ({base}): no dumps"); continue
    if cf==1:
        za=apex1(bounce(31000),None)-R
        Es=[]
        for N in Ns:
            s=load(d[N]); zp=apex1(s[:,3],s[:,0])-R
            Es.append(abs(zp-za)/abs(za))
        print(f"case {cf} ({base:8s}) apex-height err vs N {Ns}: "
              + " ".join(f"{e:.3f}" for e in Es) + f"   slope={fit(Ns,Es):+.2f}")
    else:
        # dominant components: Fz (all), in-plane Ft (2,3,5), Tz(4) / in-plane T(2,3,5)
        comps = {'Fz':9}
        if cf in (2,3,5): comps['Ft']=None; comps['Tip']=None
        if cf==4: comps['Tz']=12
        out={}
        for N in Ns:
            s=load(d[N]); st=s[:,0].astype(int); tmax=st.max()
            ana=analytical_solution(np.arange(tmax+1)*DT,cf)
            m=(st>0)&(st>=0.7*tmax); idx=np.where(m)[0]; si=st[idx]
            def rel(simcol,anacol):
                sm=s[idx,simcol].mean(); am=ana[si,anacol].mean()
                return abs(sm-am)/abs(am) if abs(am)>1e-9 else np.nan
            out.setdefault('Fz',[]).append(rel(9,8))
            if cf in (2,3,5):
                # in-plane force magnitude
                sfx,sfy=s[idx,7].mean(),s[idx,8].mean(); afx,afy=ana[si,6].mean(),ana[si,7].mean()
                out.setdefault('Ft',[]).append(abs(np.hypot(sfx,sfy)-np.hypot(afx,afy))/np.hypot(afx,afy))
                stx,sty=s[idx,10].mean(),s[idx,11].mean(); atx,aty=ana[si,9].mean(),ana[si,10].mean()
                out.setdefault('Tip',[]).append(abs(np.hypot(stx,sty)-np.hypot(atx,aty))/np.hypot(atx,aty))
            if cf==4:
                out.setdefault('Tz',[]).append(rel(12,11))
        for k,v in out.items():
            print(f"case {cf} ({base:8s}) {k:4s} err vs N {Ns}: "
                  + " ".join(f"{e:.3f}" for e in v) + f"   slope={fit(Ns,v):+.2f}")
