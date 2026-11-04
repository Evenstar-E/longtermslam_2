"""Post-seal original-point identity and source-provenance diagnosis."""
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha,xyz
from scripts.prepare_real_temporal_v1 import read_ply


def main():
    import argparse
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group();g.add_argument('--nn-readout',action='store_true');g.add_argument('--supported-readout',action='store_true');g.add_argument('--history-support',action='store_true');args=ap.parse_args()
    prefix='history_' if args.history_support else ('nn_' if args.nn_readout else ('supported_' if args.supported_readout else ''))
    prediction_key='readout_identity' if args.nn_readout else 'identity'
    out=Path('results/shared_surface_identity_likelihood_v3');seal=read(out/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(out/p)==h,p
    cfg=read(out/'config.json');pub=read('data/rescan_three_period_residual_v1/public.json')
    rows=[];summaries=[];sources=[];changes=[];truth_inputs={};source_corrections=[]
    for scene in cfg['scenes']:
        assets=read(out/scene/'assets.json');eligible={h['uid'] for h in assets if h['raw_points']>=128 and h['class'] not in [0,1,2,22]}
        zz=np.load(out/scene/'observations.npz'); truths=[];equivs=[]
        for period in [1,2]:
            root=Path('data/real_cross_session_v1/rescan/raw') if period==1 else Path('data/rescan_third_capture_v1/evaluation/raw')
            p=root/scene/'gt_segmentation'/f'{scene}_{period}.ply';truth=read_ply(p);truth_inputs[str(p)]=sha(p);truths.append(truth)
            eqfile=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_{period}.txt';eq={}
            if eqfile.exists():
                truth_inputs[str(eqfile)]=sha(eqfile)
                for line in eqfile.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        left,right=line.split('|');eq[int(left)]={int(x) for x in right.split()}
            equivs.append(eq)
            record=next(r for r in pub['records'] if r['scene']==scene and r['period']==str(period))
            raw=np.load(Path('data/rescan_three_period_residual_v1')/record['path']);assert np.array_equal(xyz(truth),raw['xyz'])
            for method in cfg['methods']:
                zzpred=np.load(out/scene/method/f'prediction{period}.npz');p=zzpred[prediction_key].copy();assert len(p)==len(truth)
                if args.supported_readout:p[~zzpred['supported']]=-1
                if args.history_support:p[~zzpred['history_supported']]=-1
                for uid in np.unique(truth['instance_idx']):
                    mask=truth['instance_idx']==uid;cls,n=np.unique(truth['class_idx'][mask],return_counts=True)
                    if mask.sum()<128 or int(cls[n.argmax()]) in [0,1,2,22]:continue
                    uid=int(uid);allowed=eq.get(uid,{uid})&eligible;ids,n=np.unique(p[mask],return_counts=True)
                    strict=float(np.mean(p[mask]==uid));equivalent=float(np.mean(np.isin(p[mask],list(allowed))))
                    rows.append({'scene':scene,'period':period,'method':method,'truth_id':uid,'points':int(mask.sum()),
                         'strict_persistent':uid in eligible,'equivalent_persistent':bool(allowed),'strict_fraction':strict,
                         'equivalent_fraction':equivalent,'strict_correct':strict>=.8,'equivalent_correct':equivalent>=.8,
                         'dominant_id':int(ids[n.argmax()]),'allowed_ids':sorted(allowed),
                         'distribution':{str(int(i)):int(v) for i,v in zip(ids,n)}})
                rr=[r for r in rows if r['scene']==scene and r['period']==period and r['method']==method]
                summaries.append({'scene':scene,'period':period,'method':method,
                    'strict_correct':sum(r['strict_correct'] for r in rr if r['strict_persistent']),
                    'strict_total':sum(r['strict_persistent'] for r in rr),
                    'equivalent_correct':sum(r['equivalent_correct'] for r in rr if r['equivalent_persistent']),
                    'equivalent_total':sum(r['equivalent_persistent'] for r in rr),
                    'unknown_all_points':int(np.sum(p<0))})
        for method in cfg['methods']:
            root=out/scene/method;state=np.load(root/'state.npz')
            for target in [1,2]:
                file=root/f'source_for{target}.npz'
                if not file.exists():continue
                inc=np.load(file);source=3-target
                truth=truths[source-1]['instance_idx'][zz[f'raw_index_{source}']];eq=equivs[source-1];mass=zz[f'mass_{source}']
                for i,h in enumerate(assets):
                    idx=inc[f'source_voxel_{i}'];ids,n=np.unique(truth[idx],return_counts=True)
                    correct=np.array([h['uid'] in eq.get(int(t),{int(t)}) for t in truth[idx]],bool)
                    sources.append({'scene':scene,'method':method,'target_period':target,'source_period':source,'archive_id':h['uid'],
                        'admitted_voxels':len(idx),'represented_point_mass':float(mass[idx].sum()),
                        'allowed_mass':float(mass[idx][correct].sum()),'disallowed_mass':float(mass[idx][~correct].sum()),
                        'truth_voxel_distribution':{str(int(k)):int(v) for k,v in zip(ids,n)},
                        'scope':'Voxel representative truth with raw mass, not exact all-point contamination rate.'})
            old=state['initial_labels1'];new=state['labels1'];truth=truths[0]['instance_idx'][zz['raw_index_1']];ids=np.array([h['uid'] for h in assets])
            olduid=np.where(old>=0,ids[np.maximum(old,0)],-1);newuid=np.where(new>=0,ids[np.maximum(new,0)],-1)
            changed=old!=new;persist=np.isin(truth,list(eligible))
            old_allowed=np.array([int(p) in (equivs[0].get(int(t),{int(t)}) & eligible) for p,t in zip(olduid,truth)])
            new_allowed=np.array([int(p) in (equivs[0].get(int(t),{int(t)}) & eligible) for p,t in zip(newuid,truth)])
            source_corrections.append({'scene':scene,'method':method,'changed_voxels':int(changed.sum()),
                'persistent_changed':int(np.sum(changed&persist)),
                'strict_fixed_voxels':int(np.sum(changed&persist&(olduid!=truth)&(newuid==truth))),
                'strict_damaged_voxels':int(np.sum(changed&persist&(olduid==truth)&(newuid!=truth))),
                'equivalent_fixed_voxels':int(np.sum(changed&persist&~old_allowed&new_allowed)),
                'equivalent_damaged_voxels':int(np.sum(changed&persist&old_allowed&~new_allowed)),
                'scope':'Descriptive correlated voxel counts, not independent trials or physical identity certainty.'})
        for uid in sorted(eligible):
            rr=[r for r in rows if r['scene']==scene and r['period']==2 and r['truth_id']==uid]
            if len({(r['strict_correct'],r['equivalent_correct']) for r in rr})>1:
                changes.append({'scene':scene,'truth_id':uid,'methods':{r['method']:{k:r[k] for k in ['strict_fraction','equivalent_fraction','dominant_id']} for r in rr}})
    save(out/(prefix+'scores.json'),{'rows':rows,'summaries':summaries,'truth_inputs':truth_inputs,'evaluation_source_sha':sha(__file__),
        'readout':('history_supported' if args.history_support else ('model_supported' if args.supported_readout else prediction_key)),'scope':'Two familiar spaces, dev captures; 80% original-instance coverage, not mIoU or displacement; permitted equivalent IDs restricted to initial eligible archives.'})
    save(out/(prefix+'provenance_diagnosis.json'),{'rows':sources,'source_corrections':source_corrections})
    save(out/(prefix+'identity_changes.json'),changes)
    print(__import__('json').dumps([r for r in summaries if r['period']==2],indent=2))


if __name__=='__main__':main()
