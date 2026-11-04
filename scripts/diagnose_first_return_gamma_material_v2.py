"""After seal/score: decompose evidence and intervene with true IDs/poses.

No model parameters or predictions are changed. True pose statistics are
diagnostic interventions, never counted as an algorithm's task successes.
"""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v2 import load_frames
from inspection_decisions.first_return_gamma_material_v1 import stats,log_gamma,posterior


def decompose(H,A,B,cfg):
    a,b,_,_=posterior(H,cfg);n,L=A[:2];p,M=B[:2]
    cross=log_gamma(a,b,n+p,L+M)-log_gamma(a,b,n,L)-log_gamma(a,b,p,M)
    empty=(n+p)==0;unknown=(H[0]+H[1])==0
    return {'cross_log_evidence':float(cross.sum()),'no_current_hit_cell_cross':float(cross[empty].sum()),'current_hit_cell_cross':float(cross[~empty].sum()),'unvisited_H_cross':float(cross[unknown].sum()),'C2_hits':int(p.sum()),'C2_hits_H_unvisited':int(p[unknown].sum()),'C2_hits_H_no_positive':int(p[H[0]==0].sum()),'C2_hits_H_positive':int(p[H[0]>0].sum()),'C2_hits_shared_C1':int(p[n>0].sum())}


def main():
    root=Path('results/first_return_gamma_material_v2');assert read(root/'audit.json')['passed'] and (root/'evaluation_sources.json').exists();rows=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;cfg=read(dest/'scene_config.json');base=Path(cfg['base_root'])/tag
        with (base/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (base/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        frames,_,_=load_frames(Path(cfg['base_root']),tag,inp['T0'][:,:3,3],cfg);H=np.load(dest/'history_stats.npz')['H'];cur=[np.load(dest/f'current_stats_{t}.npz') for t in range(2)];Qcache=[v['Q'] for v in cur];mapping=[v['mapping'] for v in cur];pred=read(dest/'predictions.json')
        for i in range(24):
            for m in cfg['methods']:
                aa,bb=pred[i]['methods'][m]['candidate_indices'];A=Qcache[0][mapping[0][i,aa]];B=Qcache[1][mapping[1][i,bb]];rows.append({'scene':tag,'history_views':run['history_views'],'asset':i,'method':m,'stage':'selected_candidate',**decompose(H[i],A,B,cfg)})
            slots=[truth['slot1'][i],truth['slot2'][i]];Q=[stats(frames[t+1],truth['poses'][t,i],slots[t],cfg)[0] for t in range(2)];rows.append({'scene':tag,'history_views':run['history_views'],'asset':i,'method':'true_ID_pose_intervention','stage':'diagnostic_only',**decompose(H[i],*Q,cfg)})
        print('DIAGNOSED',tag,flush=True)
    summary=[]
    for h in [1,3]:
        for m in cfg['methods']+['true_ID_pose_intervention']:
            rr=[r for r in rows if r['history_views']==h and r['method']==m];keys=['cross_log_evidence','no_current_hit_cell_cross','current_hit_cell_cross','unvisited_H_cross','C2_hits','C2_hits_H_unvisited','C2_hits_H_no_positive','C2_hits_H_positive','C2_hits_shared_C1'];summary.append({'history_views':h,'method':m,'n':len(rr),**{k:sum(r[k] for r in rr) for k in keys}})
    save(root/'mechanism_rows.json',rows);save(root/'mechanism_summary.json',{'rows':summary,'script_sha256':sha(Path(__file__)),'true_ID_pose_read_only_after_prediction_seal_audit_and_score':True,'true_pose_not_algorithm_prediction':True})


if __name__=='__main__':main()
