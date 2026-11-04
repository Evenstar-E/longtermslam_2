"""Quantitative disjoint-inlier mass witnesses, without fitted poses or GT IDs."""
import numpy as np
from scipy.spatial import cKDTree

def mass_conflicts(H,C,radius,tau,mass):
    diameter=[min(np.linalg.norm(np.ptp(h['xyz'],axis=0)),2*radius[i]) for i,h in enumerate(H)]
    result=[];quantiles=[0.,.1,.25,.5,.75,.9,1.]
    for j in range(len(C)):
        x=C[j]['xyz']
        for k in range(j+1,len(C)):
            y=C[k]['xyz'];separation=float(cKDTree(y).query(x,workers=1)[0].min())
            directions=list(np.eye(3));delta=y.mean(0)-x.mean(0)
            if np.linalg.norm(delta)>1e-12:directions.append(delta/np.linalg.norm(delta))
            projected=[(x@u,y@u) for u in directions]
            for i,D in enumerate(diameter):
                W=D+2*tau;best=(1.,1.) if separation>W+1e-9 else (0.,0.)
                best_score=min(mass[j]*best[0],mass[k]*best[1])
                if best_score==0:
                    for xx,yy in projected:
                        thresholds=np.unique(np.r_[np.quantile(xx,quantiles),np.quantile(yy,quantiles)-W-1e-8])
                        for t in thresholds:
                            pairs=[(np.mean(xx<=t),np.mean(yy>t+W+1e-8)),(np.mean(xx>t+W+1e-8),np.mean(yy<=t))]
                            for fj,fk in pairs:
                                score=min(mass[j]*fj,mass[k]*fk)
                                if score>best_score+1e-15:best=(float(fj),float(fk));best_score=score
                if best_score>0:result.append((i,j,k,separation,float(D),best[0],best[1]))
    return result
