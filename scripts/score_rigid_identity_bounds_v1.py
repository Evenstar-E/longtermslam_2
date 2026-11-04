"""Pointwise persistent-ID evaluation AFTER each prediction chain has been sealed."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.prepare_real_temporal_v1 import read_ply

METHODS=['pair_one_to_one','pair_many_independent_pose','ordinary_common_rigid','ordinary_extra_restarts','bounded_candidate','bounded_singleton','cut_singleton','mass_cut_singleton','ordinary_budget_filled']

def main():
    cfg=read('configs/rigid_identity_bounds_v1.json');root=Path(cfg['public_root']);public=read(root/'public.json')
    truthids=read('data/real_cross_session_v1/prepared_v2/evaluation/identity.json');all_rows=[];set_rows=[]
    for scene in cfg['spaces']:
        out=Path('results/rigid_identity_cached_v1')/scene;cutroot=Path('results/rigid_identity_cuts_v1_completed')/scene
        massroot=Path('results/rigid_identity_mass_cuts_v1')/scene
        for folder in [out,cutroot,massroot]:
            for name,h in read(folder/'prediction_seal.json')['sha256'].items():assert sha(folder/name)==h
        gt=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply');ids=gt['instance_idx']
        histids={truthids[scene]['0'][h['token']] for h in public[scene]['0'] if h['eligible']};masks={}
        for c in public[scene]['1']:
            if c['eligible']:
                with np.load(root/c['path']) as z:masks[c['token']]=z['point_indices']
        labels={m:np.full(len(gt),-1,int) for m in METHODS};sets_by_method={'bounded_singleton':{},'cut_singleton':{},'mass_cut_singleton':{}}
        for folder in sorted(out.glob('class_*')):
            result=read(folder/'results.json');bounded=read(folder/'bounded.json');cuts=read(cutroot/(folder.name+'.json'));H=result['history'];C=result['current']
            for method in METHODS[:5]:
                a=result['candidate']['assignment'] if method=='bounded_candidate' else result['controls'][method]['assignment']
                for j,i in enumerate(a):
                    if i>=0:labels[method][masks[C[j]]]=truthids[scene]['0'][H[i]]
            masscuts=read(massroot/(folder.name+'.json'))
            filled=read(Path('results/rigid_identity_budget_fill_v1')/scene/(folder.name+'.json'))
            for j,i in enumerate(filled['best']['assignment']):
                if i>=0:labels['ordinary_budget_filled'][masks[C[j]]]=truthids[scene]['0'][H[i]]
            for method,rows in [('bounded_singleton',bounded['identity_sets']),('cut_singleton',cuts['identity_sets']),('mass_cut_singleton',masscuts['identity_sets'])]:
                for j,row in enumerate(rows):
                    possible=row['possible_identities'];possible_ids=[-1 if i<0 else truthids[scene]['0'][H[i]] for i in possible];sets_by_method[method][C[j]]=possible_ids
                    if len(possible_ids)==1:labels[method][masks[C[j]]]=possible_ids[0]
        ambiguity={};p=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|');ambiguity[int(left)]=[int(v) for v in right.split()]
        for identity in np.unique(ids):
            index=ids==identity;cls,count=np.unique(gt['class_idx'][index],return_counts=True)
            if int(cls[np.argmax(count)]) in [0,1,2,22] or index.sum()<128:continue
            identity=int(identity);allowed=ambiguity.get(identity,[identity]);persistent=bool(set(allowed)&histids)
            for method in METHODS:
                u,n=np.unique(labels[method][index],return_counts=True);k=np.argmax(n);dominant=int(u[k]);purity=float(n[k]/index.sum());fraction=float(np.isin(labels[method][index],allowed).mean());null=float((labels[method][index]<0).mean())
                status='correct' if persistent and fraction>=.8 else 'wrong_archive_identity' if persistent and dominant in histids and purity>=.8 else 'mixed_or_unresolved' if persistent else 'correct_new_rejection' if null>=.8 else 'new_assigned_to_archive'
                all_rows.append({'scene':scene,'method':method,'identity':identity,'persistent':persistent,'correct_fraction':fraction,'null_fraction':null,'dominant_prediction':dominant,'purity':purity,'status':status})
        for method,sets in sets_by_method.items():
            for token,possible in sets.items():
                u,n=np.unique(ids[masks[token]],return_counts=True);identity=int(u[np.argmax(n)]);allowed=ambiguity.get(identity,[identity]);persistent=bool(set(allowed)&histids)
                set_rows.append({'scene':scene,'method':method,'token':token,'dominant_GT_id':identity,'dominant_GT_fraction':float(n.max()/n.sum()),'persistent':persistent,'possible_ids':possible,'size':len(possible),'dominant_true_id_retained':bool(set(possible)&set(allowed)) if persistent else -1 in possible,'singleton_wrong':len(possible)==1 and (not bool(set(possible)&set(allowed)) if persistent else possible[0]!=-1)})
    summary=[]
    for method in METHODS:
        for scene in cfg['spaces']+['all']:
            rr=[r for r in all_rows if r['method']==method and (scene=='all' or r['scene']==scene)];persistent=[r for r in rr if r['persistent']]
            summary.append({'method':method,'scene':scene,'persistent':len(persistent),'correct':sum(r['status']=='correct' for r in persistent),'wrong':sum(r['status']=='wrong_archive_identity' for r in persistent),'mixed_or_unresolved':sum(r['status']=='mixed_or_unresolved' for r in persistent),'new_or_missing':len(rr)-len(persistent),'new_correct_rejections':sum(r['status']=='correct_new_rejection' for r in rr),'new_assigned_to_archive':sum(r['status']=='new_assigned_to_archive' for r in rr)})
    save('results/rigid_identity_cached_v1/scores.json',{'summary':summary,'rows':all_rows,'identity_set_diagnostics':set_rows,'motion_accuracy':None,'scope':'Already observed real spaces, native predicted masks/classes, no current GT inference, no independent motion truth. Cut bound ablation after seeing gap, not independent confirmation.'})
    print([r for r in summary if r['scene']=='all'])

if __name__=='__main__':main()
