"""Provenance, constraints, actual score reconstruction; no truth input."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.uncertain_face_incidence_graph_v1 import JointProblem,induce_graph
from inspection_decisions.material_supported_continuous_alignment_v1 import world_pose
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path(read('configs/uncertain_face_incidence_graph_v1.json')['output_root']);cfg=read(root/'config.json');manifest=read(root/'manifest.json');assert read(root/'prediction_seal.json')['complete'];checks=0;maximum=0.;counts={m:[] for m in cfg['methods']};rotations={m:0 for m in cfg['methods']};depthbounds={m:0 for m in cfg['methods']};pixels=0;graph_stats=[]
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;checks+=1
        for name,h in read(Path(cfg['input_root'])/'manifest.json')[field].items():assert sha(name)==h;checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];old=read(Path(cfg['input_root'])/tag/'predictions.json');results=read(root/tag/'predictions.json');assert len(old)==len(results)==48
        with (root/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3]
        for k,(row,prior) in enumerate(zip(results,old)):
            assert (row['asset'],row['family'],row['selected_slots'])==(prior['asset'],prior['family'],prior['selected_slots']);assert row['predictions']['original']=={'T1':prior['initial_T1'],'T2':prior['initial_T2']};checks+=2;i=row['asset'];s1,s2=row['selected_slots'];track=[data[0][i],data[1][s1],data[2][s2]];graph=row['graph'];graph_stats.append((len(graph['faces']),len(graph['edges'])))
            if k%6==0:
                regenerated=induce_graph([(np.zeros(6),track[0]),(np.array(row['initial_parameters'])[:6],track[1])],cfg);assert regenerated==graph;checks+=1
            for method in cfg['methods']:
                problem=JointProblem(track,graph,row['initial_parameters'],manifest['motion_limits'],cfg,method);p=np.array(row['parameters'][method]);assert len(p)==problem.size;assert np.allclose(problem.project(p),p,rtol=0,atol=1e-12);checks+=2
                r,_,diag=problem.evaluate(p,False);err=abs(float(r@r)-row['logs'][method]['final_cost']);maximum=max(maximum,err);assert err<1e-11;checks+=1
                for t,slot in enumerate([s1,s2]):
                    T=world_pose(p[6*t:6*t+6],centers[slot]);assert np.allclose(T,row['predictions'][method]['T'+str(t+1)],atol=1e-13,rtol=0);checks+=1
                log=row['logs'][method];cost=log['initial_cost']
                for step in log['accepted_steps']:assert abs(step['before']-cost)<1e-11 and step['after']<step['before'];cost=step['after'];checks+=1
                assert abs(cost-log['final_cost'])<1e-11;checks+=1;counts[method].append(len(log['accepted_steps']));rotations[method]+=int(np.linalg.norm(p[:3]-np.array(row['initial_parameters'])[:3])>1e-7 or np.linalg.norm(p[6:9]-np.array(row['initial_parameters'])[6:9])>1e-7);depthbounds[method]+=log['depths_at_bounds'];pixels+=problem.N if method==cfg['methods'][0] else 0
                assert all(problem.pi[j]==sum(len(d['pixels']) for d in track[:b['epoch']])+b['pixel_index'] for j,b in enumerate([b for d in track for b in d['boundary']]));checks+=problem.M
    report={'checks':checks,'fits_reconstructed':1536,'max_recomputed_energy_error':maximum,'unique_pixels_across_conditional_tracks':pixels,'median_faces':float(np.median([x[0] for x in graph_stats])),'median_edges':float(np.median([x[1] for x in graph_stats])),'accepted_steps':{m:sum(v) for m,v in counts.items()},'paths_no_accepted_steps':{m:sum(s==0 for s in v) for m,v in counts.items()},'paths_rotation_actually_updated':rotations,'depth_variables_at_bounds':depthbounds,'GT_read':False,'identities_not_updated':True};save(root/'audit.json',report);print(report)


if __name__=='__main__':main()
