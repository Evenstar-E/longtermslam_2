"""Ordinary metadata-derived feasible-domain control; no pose optimization."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import catalog
from inspection_decisions.continuous_material_pose_mainline_v1 import select


def main():
    base=Path('results/continuous_material_pose_mainline_v1');cfg=read(base/'config.json');out=Path('results/continuous_material_motion_envelope_v1');out.mkdir(exist_ok=False)
    for name,h in read(base/'prediction_seal.json')['sha256'].items():assert sha(base/name)==h
    cat=catalog();rotation_limit=float(np.linalg.norm(Rotation.from_matrix(cat[:,:3,:3]).as_rotvec(),axis=1).max());translation_limit=float(np.linalg.norm(cat[:,:3,3],axis=1).max())
    sources=[Path(__file__),Path('scripts/completion_catalog_scene_v1.py'),Path('inspection_decisions/continuous_material_pose_mainline_v1.py'),Path('research/continuous_material_motion_envelope_contract_v1_2026-10-24.md')]
    inputs=list(base.glob('*/candidates.pkl'))+list(base.glob('*/optimization_logs.json'))+list(base.glob('*/predictions.json'))+list(Path(cfg['base_root']).glob('*/inference.pkl'))
    save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'base_seal_sha256':sha(base/'prediction_seal.json'),'GT_read':False,'rotation_limit_rad':rotation_limit,'translation_limit_m':translation_limit,'numerical_tolerance':1e-8,'motion_prior_is_catalogue_outer_relaxation':True})
    runs=[]
    for run in read(base/'runs.json'):
        tag=run['scene'];dest=out/tag;dest.mkdir()
        with (base/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        slots=inp['T0'][:,:3,3];allowed=[];maps=[];filtered=[];records=[]
        for i,bb in enumerate(candidates):
            aa=[];mm=[];ff=[]
            for t,branches in enumerate(bb):
                rotations=np.array([np.linalg.norm(Rotation.from_matrix(b['T'][:3,:3]).as_rotvec()) for b in branches]);translations=np.array([np.linalg.norm(b['T'][:3,3]-slots[b['slot']]) for b in branches])
                ok=(rotations<=rotation_limit+1e-8)&(translations<=translation_limit+1e-8)
                assert ok[:54].all();index=np.flatnonzero(ok).tolist();aa.append(ok);mm.append(index);ff.append([branches[j] for j in index])
                records.append({'asset':i,'epoch':t,'total':len(branches),'allowed':int(ok.sum()),'continuous_allowed':int(ok[54:].sum())})
            allowed.append(aa);maps.append(mm);filtered.append(ff)
        predictions={}
        for score in cfg['scores']:
            result=select(filtered,score,True);result['selection']=[[i,maps[i][0][k],maps[i][1][l]] for i,k,l in result['selection']]
            predictions['envelope_'+score]=result
        old=read(base/tag/'predictions.json');logs=read(base/tag/'optimization_logs.json');endpoints={(l['asset'],l['epoch'],l['parent']):l['endpoint_index'] for l in logs};direct=[];fallback=0
        for i,k,l in old['catalogue_nn']['selection']:
            picked=[]
            for t,j in enumerate([k,l]):
                endpoint=endpoints[(i,t,j)];valid=endpoint is not None and allowed[i][t][endpoint]
                picked.append(endpoint if valid else j);fallback+=not valid
            direct.append([i,*picked])
        predictions['envelope_fixed_nn_ids_refined']={'status':0,'selection':direct,'fallback_epochs':int(fallback),'objective':None}
        np.savez_compressed(dest/'allowed.npz',**{f'{i}_{t}':v for i,aa in enumerate(allowed) for t,v in enumerate(aa)})
        save(dest/'predictions.json',predictions);save(dest/'records.json',records)
        rr={k:run[k] for k in ['scene','geometry','seed','history_views']};rr.update(total_candidates=sum(v['total'] for v in records),allowed_candidates=sum(v['allowed'] for v in records),allowed_continuous=sum(v['continuous_allowed'] for v in records),fallback_epochs=int(fallback));runs.append(rr)
    save(out/'runs.json',runs);save(out/'prediction_seal.json',{'GT_read':False,'complete':True,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print('SEALED',{'rotation_limit_deg':np.degrees(rotation_limit),'translation_limit_m':translation_limit,'allowed_continuous':sum(r['allowed_continuous'] for r in runs),'all_continuous':sum(r['total_candidates']-2592 for r in runs)},flush=True)


if __name__=='__main__':main()
