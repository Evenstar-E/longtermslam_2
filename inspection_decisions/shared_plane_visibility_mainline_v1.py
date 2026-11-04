"""Finite-catalog transport of uncertain HISTORY planes to anonymous epochs.

All controls use identical support/query halfspaces. No generated geometry or
current physical identity is used. Standard polytope/LP machinery, not novelty.
"""
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from inspection_decisions.first_return_material_surface_v1 import labels,project
from inspection_decisions.ray_conditioned_plane_polytope_v1 import fit_patch
from inspection_decisions.shared_plane_visibility_v1 import visibility_halfspaces
from scripts.completion_catalog_scene_v1 import transform


def prepare_patch(patch):
    v=patch['coefficients'];center=v.mean(0);scale=np.maximum(np.ptp(v,axis=0)/2,1e-12)
    patch['center']=center;patch['scale']=scale;patch['scaled_vertices']=(v-center)/scale
    A,b=normalize_rows(patch,patch['A'],patch['b'])
    patch['base_A']=A;patch['base_b']=b
    # Source patch corners bound the projection of every possible planar patch.
    u0,v0=patch['center_uv'];pr=patch['principal'];f=patch['focal']
    uv=np.array([[u0+dx,v0+dy] for dx in [-1,1] for dy in [-1,1]])
    rays=np.c_[(uv[:,0]-pr[0])/f,(uv[:,1]-pr[1])/f,np.ones(4)]
    z=1/(patch['normals']@rays.T)
    assert (z>0).all()
    patch['corner_points']=(z[:,:,None]*rays[None,:,:]).reshape(-1,3)
    return patch


def normalize_rows(patch,A,b):
    A=np.asarray(A,float).reshape(-1,3);b=np.asarray(b,float).reshape(-1)
    aa=A*patch['scale'];bb=b-A@patch['center'];norm=np.linalg.norm(aa,axis=1)
    # Preserve constant rows too; tiny normals are normalized conservatively.
    norm=np.maximum(norm,1e-30)
    return aa/norm[:,None],bb/norm


def history_patches(frames,slots,groups,cfg):
    out=[[] for _ in slots];eligible=0;rejected=0
    principal=[(cfg['width']-1)/2,(cfg['height']-1)/2]
    for view,frame in enumerate(frames):
        lab,_=labels(frame,slots,groups,cfg)
        for y in range(1,cfg['height']-1,cfg['patch_stride_px']):
            for x in range(1,cfg['width']-1,cfg['patch_stride_px']):
                slot=int(lab[y,x])
                if slot<0 or not np.all(lab[y-1:y+2,x-1:x+2]==slot):continue
                eligible+=1
                p=fit_patch(frame['depth'][y-1:y+2,x-1:x+2],[x,y],cfg['focal_px'],principal,cfg['depth_noise_bound_m'])
                if p is None:rejected+=1;continue
                prepare_patch(p);p.update(view=view,pixel=[x,y],source_camera=frame['C'])
                out[slot].append(p)
    return out,{'eligible':eligible,'nonplanar_or_infeasible':rejected,'retained':sum(map(len,out))}


def compile_factor(patch,T0,T,frames,cfg):
    # Coordinates: source camera -> historical world -> asset -> target world.
    M=T@np.linalg.inv(T0)@patch['source_camera'];world=transform(patch['corner_points'],M)
    aa=[];bb=[];used=queried=0
    for frame in frames:
        cam,uv=project(world,frame['C'],cfg)
        if (cam[:,2]<=0).any():raise ValueError('Source patch crosses target camera plane')
        low=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0])
        high=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[cfg['width']-1,cfg['height']-1])
        if (low>high).any():continue
        yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        rc=np.c_[(xx.ravel()-(cfg['width']-1)/2)/cfg['focal_px'],(yy.ravel()-(cfg['height']-1)/2)/cfg['focal_px'],np.ones(xx.size)]
        relative=np.linalg.inv(M)@frame['C'];origin=relative[:3,3];r=rc@relative[:3,:3].T
        factor=visibility_halfspaces(patch,origin,r,frame['depth'][yy.ravel(),xx.ravel()],cfg['depth_tolerance_m'])
        queried+=len(r);used+=factor['used_rays'];aa.append(factor['A']);bb.append(factor['b'])
    if aa:A=np.concatenate(aa);b=np.concatenate(bb)
    else:A=np.empty((0,3));b=np.empty(0)
    A,b=normalize_rows(patch,A,b)
    all_rows=len(b)
    if len(b):
        # A row true throughout the original polytope remains true throughout
        # every later intersection: eliminating it preserves the feasible set.
        worst=np.max(patch['scaled_vertices']@A.T-b[None,:],axis=0)
        keep=worst>cfg['lp_tolerance'];A=A[keep];b=b[keep]
    return {'A':A,'b':b,'queried_rays':queried,'supported_rays':used,'rows_before_redundancy':all_rows,'effective_rows':len(b)}


class PlaneFeasibility:
    def __init__(self,tolerance=1e-9):self.stats=Counter();self.tol=tolerance

    def solve(self,patch,factors,force_lp=False):
        self.stats['calls']+=1
        factors=[v for v in factors if len(v['b'])]
        if factors:
            A=np.concatenate([v['A'] for v in factors]);b=np.concatenate([v['b'] for v in factors])
        else:A=np.empty((0,3));b=np.empty(0)
        residual=patch['scaled_vertices']@A.T-b[None,:]
        if not force_lp:
            if not len(b):self.stats['no_effective_rows']+=1;return True,'unrestricted'
            if np.any(np.all(residual<=self.tol,axis=1)):
                self.stats['original_vertex_witness']+=1;return True,'vertex_witness'
            if np.any(np.all(residual>self.tol,axis=0)):
                self.stats['single_row_separation']+=1;return False,'row_separation'
        ans=linprog(np.zeros(3),A_ub=np.r_[patch['base_A'],A],b_ub=np.r_[patch['base_b'],b],
                    bounds=[(None,None)]*3,method='highs',options={'primal_feasibility_tolerance':self.tol,'dual_feasibility_tolerance':self.tol})
        self.stats['lp_calls']+=1;self.stats[f'lp_status_{ans.status}']+=1
        return (True if ans.status==0 else False if ans.status==2 else None),f'lp_{ans.status}'


def masks_for_asset(patches,T0,branches,frames,cfg,solver):
    factors=[[[compile_factor(p,T0,b['T'],frames[t],cfg) for p in patches] for b in branches[t]] for t in [0,1]]
    nominal=[];independent=[];unary_trace=[]
    for t in [0,1]:
        nom=np.ones(len(branches[t]),bool);ind=nom.copy();trace=[]
        for k,ff in enumerate(factors[t]):
            outcomes=[]
            for p,factor in zip(patches,ff):
                if len(factor['b']):nom[k]&=bool(np.all(-factor['b']<=solver.tol)) # scaled centroid = 0
                ok,how=solver.solve(p,[factor]);outcomes.append({'feasible':ok,'how':how})
                if ok is False:ind[k]=False
            trace.append(outcomes)
        nominal.append(nom);independent.append(ind);unary_trace.append(trace)
    nom=nominal[0][:,None]&nominal[1][None,:]
    indep=independent[0][:,None]&independent[1][None,:];shared=indep.copy();witnesses=[];active_joint=0
    for k,l in np.argwhere(indep):
        for j,p in enumerate(patches):
            f1,f2=factors[0][k][j],factors[1][l][j]
            if not len(f1['b']) or not len(f2['b']):continue
            active_joint+=1;ok,how=solver.solve(p,[f1,f2])
            if ok is False:
                shared[k,l]=False
                if len(witnesses)<5:witnesses.append({'k':int(k),'l':int(l),'patch':j,'how':how,'rows':[len(f1['b']),len(f2['b'])]})
                break
    assert not np.any(nom&~shared),'Nominal subset of shared violated'
    assert not np.any(shared&~indep),'Shared subset of independent violated'
    counts={'patches':len(patches),'unary_nominal':[int(x.sum()) for x in nominal],
        'unary_independent':[int(x.sum()) for x in independent], 'nominal_pairs':int(nom.sum()),'independent_pairs':int(indep.sum()),
        'shared_pairs':int(shared.sum()),'joint_only_rejections':int((indep&~shared).sum()),'active_joint_patch_checks':active_joint,
        'queried_rays':sum(v['queried_rays'] for epoch in factors for branch in epoch for v in branch),
        'supported_rays':sum(v['supported_rays'] for epoch in factors for branch in epoch for v in branch),
        'rows_before_redundancy':sum(v['rows_before_redundancy'] for epoch in factors for branch in epoch for v in branch),
        'effective_rows':sum(v['effective_rows'] for epoch in factors for branch in epoch for v in branch)}
    return {'nominal_plane':nom,'independent_period_planes':indep,'shared_period_plane':shared},counts,factors,unary_trace,witnesses
