"""Full map-frame SE(3) registration, competing correspondences and known-patch readout.

Ordinary multi-start rigid registration + heuristic late visibility re-ranking.
No ground truth access; no novelty or calibrated posterior claim.
"""
import copy,itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from comparative_evidence.estimation_v1 import reference_map,world_components
from inspection_decisions.projective_visibility_v1 import decode,component_mask,triangles_from_mask,make_view,optimize
from inspection_decisions.correspondence_joint_v1 import assign,prepare as pair_prepare
from inspection_decisions.correspondence_motion_objective_v1 import solve as motion_solve
from scripts.joint_estimation import fit_rigid,transform


def valid_pose(G,cfg):
    return np.max(abs(G[:3,3]))<=cfg['se3_translation_bound_m'] and np.rad2deg(Rotation.from_matrix(G[:3,:3]).magnitude())<=cfg['se3_rotation_bound_deg']


def centres(refs,frames):
    return np.array([r['centre'] for r in refs]),[np.array([c['centre'] for c in cs]) for cs in frames]


def centre_matches(G,refs,frames,cfg):
    X,Ys=centres(refs,frames);P=transform(X,G);matches=[];score=0.
    for Y,cs in zip(Ys,frames):
        V=np.array([[c['group']==r['group'] for r in refs] for c in cs],bool)
        C=np.log1p(np.sum((Y[:,None]-P[None])**2,axis=2)/cfg['centre_sigma_m']**2)
        pairs=assign(C,V);matches.append(pairs);score+=12*(len(Y)-len(pairs))+sum(C[i,j] for i,j in pairs)
    return matches,float(score)


def proposals(refs,frames,cfg):
    X,Ys=centres(refs,frames);n=len(refs);permutations=np.array(list(itertools.permutations(range(n),3)),int)
    pool=[np.eye(4)];signatures=set();rng=np.random.default_rng(cfg['proposal_seed'])
    if len(permutations):
        edges=np.array([np.linalg.norm(X[permutations[:,a]]-X[permutations[:,b]],axis=1) for a,b in [(0,1),(0,2),(1,2)]]).T
        for Y,cs in zip(Ys,frames):
            triples=list(itertools.combinations(range(len(Y)),3));rng.shuffle(triples)
            for ids in triples:
                ids=np.array(ids);lengths=np.array([np.linalg.norm(Y[ids[a]]-Y[ids[b]]) for a,b in [(0,1),(0,2),(1,2)]])
                keep=np.max(abs(edges-lengths),axis=1)<=cfg['triangle_tolerance_m']
                for a,i in enumerate(ids):keep&=np.array([refs[j]['group']==cs[i]['group'] for j in permutations[:,a]])
                candidates=np.flatnonzero(keep);rng.shuffle(candidates)
                for k in candidates:
                    P=X[permutations[k]];Q=Y[ids]
                    if np.linalg.norm(np.cross(P[1]-P[0],P[2]-P[0]))<.02:continue
                    G=fit_rigid(P,Q)
                    if not valid_pose(G,cfg):continue
                    signature=tuple(np.r_[np.round(Rotation.from_matrix(G[:3,:3]).as_rotvec()/.02),np.round(G[:3,3]/.1)].astype(int))
                    if signature in signatures:continue
                    signatures.add(signature);pool.append(G)
                    if len(pool)>=cfg['proposal_pool']:break
                if len(pool)>=cfg['proposal_pool']:break
            if len(pool)>=cfg['proposal_pool']:break
    ranked=sorted(pool,key=lambda G:centre_matches(G,refs,frames,cfg)[1])
    return ranked[:cfg['pose_starts']],len(pool)


def refine_pose(G,refs,frames,cfg):
    G=G.copy();X,Ys=centres(refs,frames)
    for _ in range(cfg['centre_iterations']):
        matches,_=centre_matches(G,refs,frames,cfg);source=[];target=[];weights=[]
        for Y,pairs in zip(Ys,matches):
            for i,j in pairs:
                error=np.linalg.norm(transform(X[j:j+1],G)[0]-Y[i]);source.append(X[j]);target.append(Y[i]);weights.append(1/(1+(error/.2)**4))
        if len(source)>=3:
            candidate=fit_rigid(np.array(source),np.array(target),weights)
            if valid_pose(candidate,cfg):G=candidate
    for _ in range(cfg['cloud_iterations']):
        matches,_=centre_matches(G,refs,frames,cfg);source=[];target=[];weights=[]
        for cs,pairs in zip(frames,matches):
            for i,j in pairs:
                p=refs[j]['points'];q=cs[i]['points'];idx=np.linspace(0,len(q)-1,min(72,len(q)),dtype=int);q=q[idx]
                distance,nearest=cKDTree(transform(p,G)).query(q);w=1/(1+(distance/cfg['cloud_sigma_m'])**2)/max(len(q),1)
                source.extend(p[nearest]);target.extend(q);weights.extend(w)
        if len(source)>=3:
            candidate=fit_rigid(np.array(source),np.array(target),weights)
            if valid_pose(candidate,cfg):G=candidate
    matches,_=centre_matches(G,refs,frames,cfg);score=0.
    for cs,pairs in zip(frames,matches):
        score+=12*(len(cs)-len(pairs))
        for i,j in pairs:
            distance=cKDTree(transform(refs[j]['points'],G)).query(cs[i]['points'])[0]
            score+=float(np.mean(np.log1p((distance/cfg['cloud_sigma_m'])**2)))
    return dict(G=G.tolist(),matches=matches,score=float(score))


def register(raw,cfg):
    hs=[decode(o) for o in raw['history']];qs=[decode(o) for o in raw['current']];refs=reference_map(hs);frames=[world_components(o) for o in qs]
    starts,pool=proposals(refs,frames,cfg);solutions=[refine_pose(G,refs,frames,cfg) for G in starts];ranked=sorted(solutions,key=lambda r:r['score']);retained=[]
    for r in ranked:
        G=np.array(r['G']);same=False
        for s in retained:
            H=np.array(s['G']);angle=np.rad2deg(Rotation.from_matrix(G[:3,:3]@H[:3,:3].T).magnitude())
            if r['matches']==s['matches'] and np.linalg.norm(G[:3,3]-H[:3,3])<.03 and angle<1.:same=True;break
        if not same:retained.append(r)
        if len(retained)>=cfg['retained_poses']:break
    return dict(solutions=ranked,retained=retained,pool_size=pool,
        distinct_initial_assignments=len({str(centre_matches(G,refs,frames,cfg)[0]) for G in starts}),
        distinct_refined_assignments=len({str(s['matches']) for s in ranked}))


def meshes_from_history(hs,sensor,cfg):
    refs=reference_map(hs);parts=[[] for r in refs]
    for f,o in enumerate(hs):
        for i,c in enumerate(world_components(o)):
            if f==0:j=i
            else:
                choices=[(np.linalg.norm(c['centre']-r['centre']),j) for j,r in enumerate(refs) if c['group']==r['group']]
                if not choices or min(choices)[0]>1.25:continue
                j=min(choices)[1]
            mesh=triangles_from_mask(o['depth'],o['nominal_T'],component_mask(o,c),sensor,cfg)
            if len(mesh):parts[j].append(mesh)
    return [np.concatenate(v) if v else np.empty((0,3,3)) for v in parts]


def readout(raw,G,matches,cfg,projective=True):
    hs=[decode(o) for o in raw['history']];qs=[decode(o) for o in raw['current']];inverse=np.linalg.inv(G)
    for o in qs:o['nominal_T']=inverse@o['nominal_T']
    data=pair_prepare(hs,qs);observations=[[] for r in data['references']]
    for f,pairs in enumerate(matches):
        for i,j in pairs:observations[j].append((f,i))
    for f,frame in enumerate(data['frames']):
        V=np.zeros_like(np.array(frame['valid'],bool))
        for i,j in matches[f]:V[i,j]=bool(frame['valid'][i][j])
        frame['valid']=V.tolist()
    fits=[]
    if projective:
        meshes=meshes_from_history(hs,raw['sensor'],cfg);views=[[make_view(o,c) for c in world_components(o)] for o in qs]
        for j,obs in enumerate(observations):
            if not obs or not len(meshes[j]):continue
            start=np.median([data['frames'][f]['D'][i][j] for f,i in obs],axis=0)
            for f,i in obs:
                fit=optimize(meshes[j],[views[f][i]],start,raw['sensor'],cfg)
                data['frames'][f]['D'][i][j]=fit['d'];data['frames'][f]['valid'][i][j]=sum(s['comparable_pixels'] for s in fit['stats'])>=cfg['minimum_comparable_pixels']
                fits.append(dict(reference=j,frame=f,component=i,fit=fit))
    result=motion_solve(data,'shared_motion',cfg);residual=np.eye(4);residual[:3,3]=result['best']['translation']
    result['map_transform_current_from_archive']=(G@residual).tolist();result['initial_map_transform']=G.tolist()
    result['prediction']['delta_frame']='archive';result['projective_fits']=fits
    result['late_score']=float(result['best']['score']+sum(r['fit']['loss'] for r in fits))
    return result


def run_methods(raw,cfg):
    registration=register(raw,cfg);best=registration['retained'][0];G=np.array(best['G'])
    hs=[decode(o) for o in raw['history']];qs=[decode(o) for o in raw['current']]
    nominal=motion_solve(pair_prepare(hs,qs),'shared_motion',cfg);frozen=nominal['best']['matches']
    baseline=readout(raw,G,best['matches'],cfg);alternatives=[baseline]
    for branch in registration['retained'][1:]:alternatives.append(readout(raw,np.array(branch['G']),branch['matches'],cfg))
    chosen=min(range(len(alternatives)),key=lambda i:alternatives[i]['late_score'])
    outputs=dict(se3_rescore=baseline,se3_late_visibility=copy.deepcopy(alternatives[chosen]),
        se3_frozen_identity=readout(raw,G,frozen,cfg),se3_cloud_only=readout(raw,G,best['matches'],cfg,False),
        translation_projective=readout(raw,np.eye(4),frozen,cfg))
    return dict(registration=registration,late_selected=chosen,late_scores=[r['late_score'] for r in alternatives],outputs=outputs)
