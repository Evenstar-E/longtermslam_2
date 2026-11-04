"""Post-seal only: descriptor pruning, physical-node coverage, family oracle."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_material_surface_correspondence_v13b import configuration,load,setup
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from scripts.cache_material_surface_geometry_v9 import unpack
from inspection_decisions.material_surface_correspondence_v13 import phase_angle,rotation_z
from inspection_decisions.material_surface_correspondence_v13b import ChartPair
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from scripts.score_material_native_modes_v6 import physical_error
from scripts.score_material_surface_patch_registration_v9 import pool_error
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);rows=[];node_rows=[];coverage=[];phase_rows=[]
    oldoracle=list(csv.DictReader((Path(cfg['previous_root'])/'oracle_by_stage.csv').open(encoding='utf-8-sig')));oldlookup={(r['variant'],int(r['asset'])):r for r in oldoracle}
    currentoracle=list(csv.DictReader((root/'oracle_by_stage.csv').open(encoding='utf-8-sig')));currentlookup={(r['variant'],int(r['asset'])):r for r in currentoracle}
    for v in cfg['variants']:
        _,frames,ids,_,_,_,views,_,_=setup(cfg,v);charts=unpack(Path(cfg['chart_cache'])/f'{v}_charts.npz');private_chart_material(scene,charts,ids);pairposes=load(root/f'{v}_pair_proposals.npz');cache=load(root/f'{v}_cache.npz');coupled=load(root/f'{v}_proposals.npz');lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);oracle_poses=[[],[]];oracle_cut=[[],[]];fitted=[]
            for pe,target in [(0,j),(1,k)]:
                a,b=charts[0][i],charts[pe+1][target];p=ChartPair(a,b,views[0][i],views[pe+1][target],i,target,cfg);dist,index=cKDTree(b['q']).query(a['q']);rev=cKDTree(a['q']).query(b['q'])[1];source=np.flatnonzero((dist<=.015)&(rev[index]==np.arange(len(a['q']))));T=rigid(a['xyz'][source],b['xyz'][index[source]]);fitted.append(T)
                allsame=np.linalg.norm(a['q'][:,None]-b['q'][None],axis=-1)<=.015;accepted=np.zeros_like(allsame)
                for u in range(len(a['xyz'])):accepted[u,np.argsort(p.descriptor[u])[:cfg['proposal_edges_per_source']]]=True
                GT=poses[pe+1,i]@inverse(poses[0,i])
                for u,w in enumerate(index):
                    if dist[u]>.03:continue
                    frame=b['basis'][w].T@GT[:3,:3]@a['basis'][u];theta=np.arctan2(frame[1,0],frame[0,0]);hs=np.argsort(-p.corr[u,w])[:2];angles=np.array([phase_angle(p.corr[u,w],int(h)) for h in hs]);error=np.degrees(np.min(abs(np.angle(np.exp(1j*(angles-theta))))))
                    rank=int(np.flatnonzero(np.argsort(p.descriptor[u])==w)[0])+1
                    phase_rows.append(dict(variant=v,asset=i,pair=pe,rear=int(perm[2,i]%3==0),source_node=u,target_node=int(w),material_center_offset_m=float(dist[u]),normal_cosine=float(frame[2,2]),descriptor_rank=rank,reliability=float(p.reliability[u,w]),closest_top2_phase_error_deg=float(error)))
                for u,w in np.argwhere(allsame):
                    for h in np.argsort(-p.corr[u,w])[:2]:
                        R=b['basis'][w]@rotation_z(phase_angle(p.corr[u,w],int(h)))@a['basis'][u].T;U=np.eye(4);U[:3,:3]=R;U[:3,3]=b['xyz'][w]-R@a['xyz'][u];oracle_poses[pe].append(U)
                        if accepted[u,w]:oracle_cut[pe].append(U)
                node_rows.append(dict(variant=v,asset=i,pair=pe,rear=int(perm[2,i]%3==0),source_nodes=len(a['xyz']),target_nodes=len(b['xyz']),near_material_edges=int(allsame.sum()),near_edges_in_descriptor_top2=int((allsame&accepted).sum()),mutual_material_pairs=len(source),q_nearest_median_m=float(np.median(dist)),association_rigid_fit_exists=int(T is not None)))
            families={};h=lookup.get((i,j,k));retained=None
            if h is not None:
                U=coupled[f'{h}_transforms'];stage=coupled[f'{h}_stage'];retained=U[stage==0]
            for family in ['ordinary','frame']:
                A=pairposes.get(f'0_{i}_{j}_{family}',np.empty((0,4,4)));B=pairposes.get(f'1_{i}_{k}_{family}',np.empty((0,4,4)))
                value=pool_error(A,B,x,y1,y2) if len(A) and len(B) else float('inf')
                if retained is not None:value=min(value,min(float(physical_error(*T,x,y1,y2)) for T in retained))
                families[family]=None if not np.isfinite(value) else value
            for family,AB in [('q_assisted_phase_all_edges',oracle_poses),('q_assisted_phase_top2_edges',oracle_cut)]:
                # Cap diagnostic Cartesian by best per-edge anchor errors before
                # exact three-segment comparison. This is a bounded oracle probe.
                reduced=[]
                for pe,TT in enumerate(AB):
                    yy=[y1,y2][pe];err=[np.quantile(np.linalg.norm(apply(T,x)-yy,axis=1),.95) for T in TT];idx=np.argsort(err)[:32];reduced.append(np.array(TT)[idx] if len(TT) else np.empty((0,4,4)))
                value=pool_error(*reduced,x,y1,y2) if all(len(T) for T in reduced) else None;families[family]=value
            fiterr=float(physical_error(*fitted,x,y1,y2)) if all(T is not None for T in fitted) else None
            rr=currentlookup[v,i];old=oldlookup[v,i];coverage.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),correct_identity_searched=int(h is not None),v12_full_error_m=float(old['continuous_error_m']),v13_retained_error_m=float(rr['retained_error_m']) if rr['retained_error_m'] else None,old_qualified_lost_by_finite_retention=int(h is not None and float(old['continuous_error_m'])<=.05 and (not rr['retained_error_m'] or float(rr['retained_error_m'])>.05)),old_qualified_outside_shared_identity_graph=int(h is None and float(old['continuous_error_m'])<=.05)))
            rows.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),q_assisted_rigid_error_m=fiterr,**{name+'_error_m':value for name,value in families.items()}))
    write_csv(root/'node_material_coverage.csv',node_rows);write_csv(root/'candidate_family_oracle.csv',rows);write_csv(root/'retained_pool_coverage.csv',coverage);write_csv(root/'phase_vs_center_offset.csv',phase_rows);summary=[]
    for v in cfg['variants']:
        rr=[r for r in rows if r['variant']==v];nn=[r for r in node_rows if r['variant']==v and r['rear'] and r['pair']==1];cc=[r for r in coverage if r['variant']==v]
        summary.append(dict(variant=v,old_qualified_lost_by_finite_retention=[r['asset'] for r in cc if r['old_qualified_lost_by_finite_retention']],old_qualified_outside_shared_identity_graph=[r['asset'] for r in cc if r['old_qualified_outside_shared_identity_graph']],rear_no_15mm_material_pair=[r['asset'] for r in nn if not r['near_material_edges']],rear_near_edges=sum(r['near_material_edges'] for r in nn),rear_near_edges_in_descriptor_top2=sum(r['near_edges_in_descriptor_top2'] for r in nn),rear_oracle_qualified={name:sum(r['rear'] and r[name+'_error_m'] is not None and r[name+'_error_m']<=.05 for r in rr) for name in ['ordinary','frame','q_assisted_phase_all_edges','q_assisted_phase_top2_edges','q_assisted_rigid']}))
    phase_bins=[]
    for lo,hi in [(0,.003),(.003,.01),(.01,.02),(.02,.03)]:
        pp=[r for r in phase_rows if lo<=r['material_center_offset_m']<hi and r['normal_cosine']>.8]
        phase_bins.append(dict(offset_min_m=lo,offset_max_m=hi,rows=len(pp),median_top2_phase_error_deg=float(np.median([r['closest_top2_phase_error_deg'] for r in pp])) if pp else None,median_descriptor_rank=float(np.median([r['descriptor_rank'] for r in pp])) if pp else None,scope='Correlated observation diagnostic, not causal intervention or independent scenes.'))
    save(root/'chart_coverage_summary.json',dict(summary=summary,phase_offset_bins=phase_bins,scope='Private GT association/oracle diagnostics AFTER seal, not predictions. q-assistedphase uses measured correlations but GT selects near-material links and best poses. 32 endpoints per side bounds the oracle probe. Retained-pool audit explicitly checks changed search scope. No raw-data impossibility claim.'));print(json.dumps(dict(summary=summary,phase_offset_bins=phase_bins),indent=2))


if __name__=='__main__':main()
