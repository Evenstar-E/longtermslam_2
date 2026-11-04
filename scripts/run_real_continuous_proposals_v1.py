"""Standard continuous estimators; freeze ALL outputs before GT evaluation."""
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair


def cloud(frame,voxel,cfg):
    p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(frame.all_xyz);p.colors=o3d.utility.Vector3dVector(frame.rgb)
    p=p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*cfg['normal_radius_voxel_multiplier'],max_nn=cfg['normal_max_neighbors']))
    return p


def main():
    cp=Path('configs/real_continuous_proposals_v1.json');cfg=read(cp);root=Path(cfg['public_root']);base=Path(cfg['pool_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for name,h in read(base/'prediction_seal.json')['sha256'].items():assert sha(base/name)==h
    sources=[cp,Path(__file__),Path('inspection_decisions/ray_material_evidence_v1.py'),Path('research/real_continuous_proposals_contract_v1_2026-10-24.md')]
    inputs=[root/'public.json',base/'effective_configuration.json']+[p for p in (root/'public').rglob('*') if p.is_file()]+list(base.glob('pair_*/predictions.json'))
    save(out/'manifest.json',dict(config=cfg,open3d_version=o3d.__version__,sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},GT_read=False))
    fc=read(base/'effective_configuration.json');meta=read(root/'public.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in meta['clouds']}
    pyramid={(stem,v):cloud(frame,v,cfg) for stem,frame in frames.items() for v in cfg['voxel_scales_m']};runs=[];start=time.perf_counter()
    for path in sorted(base.glob('pair_*/predictions.json')):
        p=read(path);pair=p['pair'];model=RayPair(frames[pair['history_stem']],frames[pair['current_stem']],p['anchor'],fc);extra=[];traces=[];endpoints={}
        for method in cfg['methods']:
            for seed in cfg['seeds']:
                initial=p['choices'][seed];T=np.array(p['pool'][initial]['T_current_from_history']);last=None
                for scale,(voxel,iterations) in enumerate(zip(cfg['voxel_scales_m'],cfg['max_iterations'])):
                    h=pyramid[(pair['history_stem'],voxel)];c=pyramid[(pair['current_stem'],voxel)];clock=time.perf_counter();error=None
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,relative_rmse=1e-6,max_iteration=iterations)
                    try:
                        if method=='point_to_plane':
                            ans=o3d.pipelines.registration.registration_icp(h,c,voxel*cfg['correspondence_distance_voxel_multiplier'],T,o3d.pipelines.registration.TransformationEstimationPointToPlane(),criteria)
                        else:
                            ans=o3d.pipelines.registration.registration_colored_icp(h,c,voxel*cfg['correspondence_distance_voxel_multiplier'],T,o3d.pipelines.registration.TransformationEstimationForColoredICP(cfg['colored_lambda_geometric']),criteria)
                        proposal=np.asarray(ans.transformation)
                        if not np.isfinite(proposal).all() or len(ans.correspondence_set)==0:raise ValueError('nonfinite pose or no correspondences')
                        T=proposal.copy();scores,diagnosis=model.score(T);last=len(p['pool'])+len(extra)
                        extra.append(dict(T_current_from_history=T.tolist(),source=method,seed=seed,scale=scale,voxel_m=voxel,scores=scores,diagnosis=diagnosis))
                        stat=dict(fitness=ans.fitness,inlier_rmse=ans.inlier_rmse,correspondences=len(ans.correspondence_set))
                    except (RuntimeError,ValueError) as exc:
                        error=str(exc);stat={}
                    traces.append(dict(method=method,seed=seed,initial_index=initial,scale=scale,iterations_cap=iterations,elapsed_s=time.perf_counter()-clock,error=error,output_index=last if error is None else None,**stat))
                    if error is not None:break
                endpoints[method+'__'+seed]=last
        pool=p['pool']+extra;choices={}
        for group in ['old','point_to_plane','colored_icp','all']:
            eligible=[i for i,e in enumerate(pool) if i<len(p['pool']) or group=='all' or e['source']==group]
            for score in ['old_nn','ordinary_overlap_rgb']:
                choices[group+'__'+score]=min(eligible,key=lambda i:pool[i]['scores'][score])
        dest=out/path.parent.name;dest.mkdir();save(dest/'predictions.json',dict(pair=pair,anchor=p['anchor'],old_pool_size=len(p['pool']),pool=pool,choices=choices,endpoints=endpoints,traces=traces,old_pool_sha256=sha(path),GT_read=False))
        runs.append(dict(pair=path.parent.name,new_poses=len(extra),failures=sum(t['error'] is not None for t in traces),elapsed_s=time.perf_counter()-start));print('CONTINUOUS',runs[-1],flush=True)
    save(out/'runs.json',runs);save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))


if __name__=='__main__':main()
