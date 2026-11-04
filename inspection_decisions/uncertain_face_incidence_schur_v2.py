"""Standard Schur elimination + bound-active depth columns; same v1 energy.

Not new mathematical inference: each residual touches <=1 latent depth, hence
the depth normal block is diagonal. Pose/face Schur system stays small.
"""
import numpy as np


def schur_step(J,r,theta,damping,active=None):
    A=J[:,:theta];g=np.asarray(A.T@r).ravel();H=(A.T@A).toarray();H.flat[::len(H)+1]+=damping*damping
    if J.shape[1]==theta:return np.linalg.lstsq(H,-g,rcond=1e-9)[0],{'diagonal_depths':0}
    D=J[:,theta:];Q=(D.T@D).tocoo();assert np.all(Q.row==Q.col),'a residual may not couple two independent pixel depths'
    den=np.asarray(D.power(2).sum(0)).ravel()+damping*damping;gd=np.asarray(D.T@r).ravel();B=(A.T@D).toarray();free=np.ones(D.shape[1],bool) if active is None else ~active;C=B[:,free]/den[free][None];S=H-C@B[:,free].T;rhs=-g+C@gd[free];dx=np.linalg.lstsq(S,rhs,rcond=1e-9)[0];dz=np.zeros(D.shape[1]);dz[free]=-(gd[free]+B[:,free].T@dx)/den[free];return np.r_[dx,dz],{'diagonal_depths':D.shape[1]}


def fit(problem,initial,iterations=8):
    p=problem.project(np.array(initial,float));r,J,diag=problem.evaluate(p);cost=float(r@r);start=cost;trace=[];active_total=0
    for iteration in range(iterations):
        active=np.zeros(problem.N,bool) if problem.free_depth else None
        for _ in range(6):
            step,_=schur_step(J,r,problem.depth_start,problem.cfg['lsmr_damping'],active)
            if not problem.free_depth:break
            z=p[problem.depth_start:];dz=step[problem.depth_start:];eps=problem.cfg['depth_noise_bound_m'];out=((z<=problem.z-eps+1e-10)&(dz<0))|((z>=problem.z+eps-1e-10)&(dz>0));out&=~active
            if not out.any():break
            active|=out
        active_total+=int(active.sum()) if active is not None else 0;accepted=False
        for back in range(problem.cfg['backtracking_steps']):
            trial=problem.project(p+step*.5**back);rr,_,_=problem.evaluate(trial,False);value=float(rr@rr)
            if value<cost-problem.cfg['descent_tolerance']:
                trace.append({'iteration':iteration,'before':cost,'after':value,'backtracks':back});p=trial;r,J,diag=problem.evaluate(p);cost=value;accepted=True;break
        if not accepted:break
    return p,{'initial_cost':start,'final_cost':cost,'accepted_steps':trace,'active_depth_columns_total':active_total,'surface_saturated':diag['surface_saturated'],'incidence_saturated':diag['incidence_saturated'],'pixels':problem.N,'boundary_pairs':problem.M,'faces':problem.F,'edges':len(problem.graph['edges']),'depths_at_bounds':int(np.sum(abs(p[problem.depth_start:]-problem.z)>=problem.cfg['depth_noise_bound_m']-1e-9)) if problem.free_depth else 0}


def map_parameters(source,target,parameters):
    p=target.start();p[:12]=parameters[:12]
    if target.free_faces:p[12:12+target.F]=parameters[12:12+source.F] if source.free_faces else source.face0
    if target.free_depth:p[target.depth_start:]=parameters[source.depth_start:] if source.free_depth else source.z
    return target.project(p)
