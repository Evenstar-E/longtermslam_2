"""POST-SEAL diagnostic of observed-source identity; never an inference input."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.completion_catalog_scene_v1 import ray_box
from scripts.first_return_surface_scene_v1 import geometry_for
from inspection_decisions.typed_material_incidence_graph_v1 import JointProblem


def source_truth(data,T,center,boxes):
    """True FIRST union-of-boxes hit, then all coincident true source planes."""
    O=np.array([p['origin'] for p in data['pixels']]);r=np.array([p['ray'] for p in data['pixels']]);o=(O+center-T[:3,3])@T[:3,:3];r=r@T[:3,:3];labels=[]
    for origin,ray in zip(o,r):
        depths=np.array([float(ray_box(origin,ray,lo,hi)) for lo,hi in boxes]);z=depths.min();q=origin+ray*z;planes=[]
        if np.isfinite(z):
            for (lo,hi),depth in zip(boxes,depths):
                if abs(depth-z)<1e-7:
                    for a in range(3):
                        for v in [lo[a],hi[a]]:
                            if abs(q[a]-v)<1e-7:planes.append((a,float(v)))
        labels.append(planes)
    return labels


def main():
    root=Path('results/typed_material_incidence_graph_v2');assert read(root/'prediction_seal.json')['scenes']==8 and read(root/'audit.json')['passed'];cfg=read(root/'config.json');limits=read(root/'manifest.json')['motion_limits'];scores={(r['scene'],r['asset'],r['family'],r['method'],r['stage']):r for r in read(root/'scores.json')};rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag;geometry=geometry_for(run['seed'],24,run['geometry'])
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl')
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        centers=inp['T0'][:,:3,3];labels={}
        for i in range(24):
            slots=[i,int(truth['slot1'][i]),int(truth['slot2'][i])];poses=[inp['T0'][i],truth['poses'][0,i],truth['poses'][1,i]];labels[i]=[source_truth(data[t][s],T,centers[s],geometry[i]) for t,(s,T) in enumerate(zip(slots,poses))]
        for case in read(root/tag/'predictions.json'):
            i=case['asset']
            for m,v in case['methods'].items():
                s=scores[(tag,i,case['family'],m,'final')]
                if not s['both_ids']:continue
                G=case['graphs']['clipped' if m=='clipped_ray' else 'typed'];slots=v['stages']['final']['slots'];track=[data[0][i],data[1][slots[0]],data[2][slots[1]]];p=JointProblem(track,G,v['initial_parameters'],limits,cfg,m);q=np.array(v['final_parameters']);_,_,diag=p.evaluate(q,False);f=diag['pixel_face'];F0=len(case['graphs']['clipped']['faces']);offset=0;epoch=[]
                for t,d in enumerate(track):
                    ff=f[offset:offset+len(d['pixels'])];distance=[];assigned_termination=0;unanchored=0;truth_available=0;axis_correct=0;correct=0;total=0
                    for j,face in enumerate(ff):
                        total+=1
                        if face<0:continue
                        a=G['faces'][face]['axis'];v0=q[12+face];true=labels[i][t][j];truth_available+=bool(true);candidates=[abs(v0-d0) for a0,d0 in true if a0==a];axis_correct+=bool(candidates);dist=min(candidates,default=np.inf);correct+=dist<=.006;distance.append(float(dist));assigned_termination+=face>=F0;unanchored+=not bool(G['faces'][face]['historical_intervals'])
                    epoch.append({'epoch':t,'pixels':total,'truth_available':truth_available,'source_axis_correct':axis_correct,'assigned_source_plane_within6mm':correct,'assigned_termination_nodes':assigned_termination,'assigned_unanchored_nodes':unanchored,'median_same_axis_plane_error_m':float(np.median([x for x in distance if np.isfinite(x)])) if any(np.isfinite(distance)) else None});offset+=len(d['pixels'])
                rows.append({'scene':tag,'asset':i,'family':case['family'],'method':m,'success':s['success'],'error_m':s['error_m'],'initial_both_ids':scores[(tag,i,case['family'],'original','original')]['both_ids'],'epochs':epoch})
        print('DIAGNOSED',tag,flush=True)
    summary={}
    for family in ['catalogue_plane','material_shared']:
        summary[family]={}
        for m in cfg['methods']:
            rr=[r for r in rows if r['family']==family and r['method']==m];groups={}
            for group,sub in [('all_correct_ID',rr),('success',[r for r in rr if r['success']]),('quantification_failure',[r for r in rr if not r['success']])]:
                groups[group]={'tasks':len(sub),'epoch_totals':[{k:sum(r['epochs'][t][k] for r in sub) for k in ['pixels','truth_available','source_axis_correct','assigned_source_plane_within6mm','assigned_termination_nodes','assigned_unanchored_nodes']} for t in range(3)]}
            summary[family][m]=groups
    lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows};paired=[]
    for family in ['catalogue_plane','material_shared']:
        for control in ['clipped_ray','line_joint','ray_no_history']:
            aa=[r for r in rows if r['family']==family and r['method']=='typed_ray' and (r['scene'],r['asset'],family,control) in lookup];bb=[lookup[(r['scene'],r['asset'],family,control)] for r in aa];item={'family':family,'control':control,'common_correct_ID_tasks':len(aa)}
            for name,rr in [('typed_ray',aa),(control,bb)]:item[name]=[{k:sum(r['epochs'][t][k] for r in rr) for k in ['pixels','source_axis_correct','assigned_source_plane_within6mm','assigned_termination_nodes','assigned_unanchored_nodes']} for t in range(3)]
            paired.append(item)
    save(root/'source_association_postseal.json',rows);save(root/'source_association_summary.json',summary);save(root/'source_association_paired.json',paired);save(root/'source_association_sources.json',{'script_sha256':sha(Path(__file__)),'truth_sha256':truth_hash,'postseal_only':True,'no_GT_into_inference':True,'correlation_not_per_failure_causal_intervention':True,'threshold_m':.006});print('DIAGNOSIS_SAVED',len(rows),'correct-ID method/tracks',flush=True)


if __name__=='__main__':main()
