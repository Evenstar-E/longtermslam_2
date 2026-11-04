"""Evaluate completed author outputs only; no scores for timeouts or partial states."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.prepare_real_temporal_v1 import read_ply

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/rescan_author_v1/inference_long_scene_a_v2');args=parser.parse_args()
    root=Path(args.input);run=root/'run.json'
    if not run.exists():raise RuntimeError('Native run has not finished')
    status=json.loads(run.read_text(encoding='utf-8'));rows=[];scenes=[]
    cfg=json.loads(Path('configs/real_cross_session_v1.json').read_text(encoding='utf-8'))
    for scene in ['scene_a','scene_b','scene_c']:
        stages=[r for r in status['records'] if r['scene']==scene]
        if not any(r['stage']=='segment_transfer' and r['returncode']==0 for r in stages):continue
        candidates=list((root/scene/'prediction').glob('*segmented*.ply'))
        if not candidates:candidates=list((root/scene/'prediction').glob('*.ply'))
        # The author's database explicitly states which saved cloud is the scene output.
        db=(root/scene/'prediction.rsdb').read_text(encoding='utf-8').splitlines()
        lines=[s for s in db if s.startswith('scene ')]
        saved=lines[-1].split()[3]
        path=Path(saved.replace('/mnt/c/','C:/'))
        if not path.exists():raise FileNotFoundError(path)
        pred=read_ply(path)
        raw=Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'
        hist=read_ply(raw/f'{scene}_0.ply');truth=read_ply(raw/f'{scene}_1.ply')
        x=np.column_stack([truth[n] for n in ['x','y','z']]);y=np.column_stack([pred[n] for n in ['x','y','z']])
        same=len(x)==len(y) and np.array_equal(x,y)
        if same:dist=np.zeros(len(x));ix=np.arange(len(x))
        else:dist,ix=cKDTree(y).query(x)
        predicted=pred['instance_idx'][ix].copy();predicted[dist>.002]=-1
        eq={};meta=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if meta.exists():
            for line in meta.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|');eq[int(left)]=[int(v) for v in right.split()]
        eligible_history={int(i) for i in np.unique(hist['instance_idx']) if (hist['instance_idx']==i).sum()>=128}
        for identifier in np.unique(truth['instance_idx']):
            mask=truth['instance_idx']==identifier
            classes,counts=np.unique(truth['class_idx'][mask],return_counts=True);cls=int(classes[np.argmax(counts)])
            if cls in cfg['excluded_classes'] or mask.sum()<128:continue
            identifier=int(identifier);allowed=eq.get(identifier,[identifier]);persistent=bool(set(allowed)&eligible_history)
            unique,count=np.unique(predicted[mask],return_counts=True);k=int(np.argmax(count));dominant=int(unique[k]);purity=float(count[k]/mask.sum())
            correct_fraction=float(np.isin(predicted[mask],allowed).mean())
            correct=correct_fraction>=.8
            archive_prediction=dominant in eligible_history and purity>=.8
            rows.append(dict(scene=scene,instance=identifier,cls=cls,persistent=persistent,raw_points=int(mask.sum()),
                dominant_prediction=dominant,purity=purity,correct_fraction=correct_fraction,correct=correct,
                status='correct' if correct else 'wrong_archive_identity' if archive_prediction else 'mixed_or_unresolved'))
        rr=[r for r in rows if r['scene']==scene and r['persistent']]
        scenes.append(dict(scene=scene,persistent=len(rr),correct=sum(r['correct'] for r in rr),
            wrong_archive_identity=sum(r['status']=='wrong_archive_identity' for r in rr),
            mixed_or_unresolved=sum(r['status']=='mixed_or_unresolved' for r in rr),
            coordinate_order_identical=same,evaluation_nn_missing_fraction=float((dist>.002).mean()),prediction_ply=str(path),
            runtime_s=sum(r['elapsed_s'] for r in stages)))
    report={'scenes':scenes,'rows':rows,'metric':'At least 80% of original current GT-instance points assigned an allowed historical identity; NOT original Rescan mIoU',
        'front_end':'Author pipeline has no current masks/classes; module baseline table has privileged masks/classes. Scores must be interpreted as separate contracts.',
        'motion_accuracy':None,'completed_scenes':len(scenes),'no_scores_for_incomplete':True}
    (root/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(scenes,indent=2))

if __name__=='__main__':main()
