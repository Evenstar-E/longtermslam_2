"""Post-seal strict/equivalent initial-archive identity scores, no motion claims."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_rescan_three_period_v1 import verify_runs,run_root
from scripts.run_rescan_third_period_v1 import sha,save
from scripts.prepare_real_temporal_v1 import read_ply


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--progress',action='store_true');args=ap.parse_args()
    base=verify_runs(progress=args.progress)
    if args.progress:
        base=Path('results/rescan_third_period_progress_v1');base.mkdir(exist_ok=True)
    prepared=Path('data/rescan_three_period_native_v1')
    for p,h in read(prepared/'seal.json')['sha256'].items():assert sha(prepared/p)==h
    cfg=read('configs/rescan_third_period_v1.json')
    rows=[];summaries=[];truth_inputs={};native_diagnostics=[]
    for scene in cfg['scenes']:
        historical=Path(cfg['prior_roots'][scene])/'history.ply'
        truthfile=Path('data/rescan_third_capture_v1/evaluation/raw')/scene/'gt_segmentation'/f'{scene}_2.ply'
        equivalence=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_2.txt'
        h,truth=read_ply(historical),read_ply(truthfile)
        for p in [historical,truthfile,equivalence]:truth_inputs[str(p)]=sha(p)
        eligible_history=set()
        for identifier in np.unique(h['instance_idx']):
            mask=h['instance_idx']==identifier
            classes,counts=np.unique(h['class_idx'][mask],return_counts=True)
            if mask.sum()>=128 and int(classes[np.argmax(counts)]) not in cfg['excluded_classes']:eligible_history.add(int(identifier))
        eq={}
        for line in equivalence.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            left,right=line.split('|');eq[int(left)]=[int(x) for x in right.split()]
        for mode in cfg['modes']+['inherited_protected']:
            root=run_root(scene,mode,args.progress)
            pred=read_ply(Path(read(root/'native_groups.json')['prediction']))
            assert len(pred)==len(truth) and all(np.array_equal(pred[f],truth[f]) for f in ['x','y','z'])
            for identifier in np.unique(truth['instance_idx']):
                mask=truth['instance_idx']==identifier
                classes,counts=np.unique(truth['class_idx'][mask],return_counts=True);cls=int(classes[np.argmax(counts)])
                if cls in cfg['excluded_classes'] or mask.sum()<128:continue
                identifier=int(identifier);declared_allowed=eq.get(identifier,[identifier])
                allowed=sorted(set(declared_allowed)&eligible_history)
                p=pred['instance_idx'][mask];ids,n=np.unique(p,return_counts=True);j=int(np.argmax(n))
                strict=float(np.mean(p==identifier));equiv=float(np.mean(np.isin(p,allowed)))
                rows.append({'scene':scene,'mode':mode,'truth_id':identifier,'points':int(mask.sum()),'class':cls,
                    'strict_initial_persistent':identifier in eligible_history,
                    'equivalent_initial_persistent':bool(set(allowed)&eligible_history),
                    'allowed_ids':allowed,'dataset_declared_allowed_ids':declared_allowed,'strict_fraction':strict,'equivalent_fraction':equiv,
                    'strict_correct':strict>=.8,'equivalent_correct':equiv>=.8,
                    'dominant_id':int(ids[j]),'dominant_fraction':float(n[j]/n.sum()),
                    'prediction_distribution':{str(int(i)):int(c) for i,c in zip(ids,n)}})
            rr=[r for r in rows if r['scene']==scene and r['mode']==mode]
            strict=[r for r in rr if r['strict_initial_persistent']];equiv=[r for r in rr if r['equivalent_initial_persistent']]
            summaries.append({'scene':scene,'mode':mode,'strict_correct':sum(r['strict_correct'] for r in strict),'strict_total':len(strict),
                'equivalent_correct':sum(r['equivalent_correct'] for r in equiv),'equivalent_total':len(equiv),
                'not_initial_persistent':len(rr)-len(equiv),'native_groups':read(root/'native_groups.json')['group_count'],
                'runtime_s':sum(r['elapsed_s'] for r in read(root/'run.json')['records'])})
            # Characterize ALL qualifying native masks, not only successful objects.
            for group in read(root/'native_groups.json')['groups']:
                mask=pred['instance_idx']==group['predicted_id'];ids,n=np.unique(truth['instance_idx'][mask],return_counts=True);j=int(np.argmax(n));tid=int(ids[j])
                native_diagnostics.append({'scene':scene,'mode':mode,**group,'dominant_truth_id':tid,
                    'truth_purity':float(n[j]/n.sum()),'dominant_truth_recall':float(n[j]/np.sum(truth['instance_idx']==tid)),
                    'truth_distribution':{str(int(i)):int(c) for i,c in zip(ids,n)}})
    changes=[]
    for scene in cfg['scenes']:
        rr={m:{r['truth_id']:r for r in rows if r['scene']==scene and r['mode']==m and r['equivalent_initial_persistent']} for m in cfg['modes']+['inherited_protected']}
        for uid in rr['initial_archive']:
            records={m:v[uid] for m,v in rr.items()}
            if len({r['dominant_id'] for r in records.values()})>1 or len({r['equivalent_correct'] for r in records.values()})>1:
                changes.append({'scene':scene,'truth_id':uid,'methods':{m:{k:r[k] for k in ['dominant_id','strict_fraction','equivalent_fraction','equivalent_correct']} for m,r in records.items()}})
    save(base/'scores.json',{'summaries':summaries,'rows':rows,'truth_inputs':truth_inputs,'progress_repair_used_in_scene_a':args.progress,
        'metric':'At least80% original GT-instance points covered by strict or allowed initial-archive IDs; not original mIoU. Novel ID numbers are not scored as correct births.',
        'scope':'Two known spaces with unused capture2; correlated objects; no displacement GT, no novel-method claim.',
        'evaluation_source_sha':sha(__file__)})
    save(base/'identity_changes.json',{'changes':changes,'scope':'Paired descriptive outcomes, not significance or isolated causal proof.'})
    save(base/'native_fragment_diagnosis.json',{'groups':native_diagnostics})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
