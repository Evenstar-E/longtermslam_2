"""Posthoc GT-guided feasible ID-swap diagnostic; never modifies predictions."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from scripts.run_textured_material_comparison_v1 import save,sha
from scripts.score_textured_material_comparison_v1 import material_anchors


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));cfg=json.loads(Path(v4['base_config']).read_text(encoding='utf-8'));root=Path(v4['comparison_root']);assert json.loads((root/'scoring_audit.json').read_text(encoding='utf-8'))['prescore_control_completed_before_truth']
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    cases=[('test_000','rich',7),('test_001','rich',10),('test_000','changed',11),('test_001','flat',23)];results=[]
    for name,variant,i in cases:
        truth=json.loads((Path(cfg['dataset'])/name/'supervision/scene.json').read_text(encoding='utf-8'))
        anchors=material_anchors(Path(cfg['dataset'])/name)
        frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        lf=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)];ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,cfg) for f in frames]
        with np.load(root/name/f'{variant}_surface.npz') as f:pred=dict(f)
        with np.load(root/name/f'{variant}_surface_family_pools.npz') as f:pool=dict(f)
        scorers={}
        def score(e,f,a,b):
            key=e,f,a,b
            if key not in scorers:
                aa,bb=ep[e][a],ep[f][b];scorers[key]=PairScore(aa,bb,old.matches(aa,bb,bcfg),groups[e][a],groups[f][b],{},frames[f],b,cfg)
            return scorers[key]
        def energy(asset,j,k,G1,G2):
            return float(score(0,1,asset,j).score(G1[None],['ray'])['ray'][0]+score(0,2,asset,k).score(G2[None],['ray'])['ray'][0]+.5*score(1,2,j,k).score((G2@inverse(G1))[None],['ray'])['ray'][0])
        def optimum(asset,j,k,details=False):
            A=pool['transforms'][0,asset,j,:pool['counts'][0,asset,j]];B=pool['transforms'][1,asset,k,:pool['counts'][1,asset,k]]
            T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1));v=score(0,1,asset,j).score(T1,['ray'])['ray']+score(0,2,asset,k).score(T2,['ray'])['ray']+.5*score(1,2,j,k).score(T2@inverse(T1),['ray'])['ray']
            if not details:return float(v.min())
            q=anchors[asset];poses=np.array(truth['poses']);x=old.apply(poses[0,asset],q);real=[old.apply(poses[e,asset],q) for e in range(3)]
            z1=np.einsum('bij,nj->bni',T1[:,:3,:3],x)+T1[:,None,:3,3];z2=np.einsum('bij,nj->bni',T2[:,:3,:3],x)+T2[:,None,:3,3]
            errors=np.maximum.reduce([np.quantile(np.linalg.norm(z1-real[1],axis=2),.95,axis=1),np.quantile(np.linalg.norm(z2-real[2],axis=2),.95,axis=1),
                np.quantile(np.linalg.norm((z2-z1)-(real[2]-real[1]),axis=2),.95,axis=1)])
            good=errors<=.05
            return dict(best_pool_energy=float(v.min()),best_energy_pose_worst_error_m=float(errors[v.argmin()]),
                best_qualified_pose_energy=float(v[good].min()) if good.any() else None)
        current=pred['target_slots'];proposed=current.copy();desired=np.array(truth['permutations'])[1:,i]
        for e in [0,1]:
            other=int(np.flatnonzero(proposed[:,e]==desired[e])[0]);proposed[i,e],proposed[other,e]=proposed[other,e],proposed[i,e]
        affected=np.flatnonzero((proposed!=current).any(1));before=sum(energy(a,*current[a],*pred['transforms'][a]) for a in affected)
        after=sum(optimum(a,*proposed[a]) for a in affected)
        record=dict(scene=name,variant=variant,asset=i,current_slots=current[i].tolist(),GT_slots=desired.tolist(),affected=affected.tolist(),
            selected_asset_energy=energy(i,*current[i],*pred['transforms'][i]),GT_identity_best_pool_energy=optimum(i,*desired),
            feasible_swap_energy_before=before,feasible_swap_energy_after=after,delta=after-before,feasible_swap_improves=after<before-1e-9,
            GT_identity_pose_diagnostic=optimum(i,*desired,details=True))
        results.append(record);print('V4 SELECTION DIAGNOSTIC',record,flush=True)
    save(root/'selection_diagnostic.json',dict(source_sha=sha(__file__),posthoc_GT_guided=True,not_algorithm_scores=True,predictions_unchanged=True,
        scope='four surface failures with oracle or regression, feasible full bijection swaps and exhaustive pose-pair evaluation; not exhaustive joint ID optimization',results=results))


if __name__=='__main__':main()
