"""Keep every current point; native semantic/mask filtering must not erase unknown geometry."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply
from scripts.run_rescan_third_period_v1 import save,sha
from scripts.prepare_rescan_three_period_v1 import run_root


def main():
    fragments=Path('data/rescan_three_period_native_v1');public=json.loads((fragments/'public.json').read_text(encoding='utf-8'))
    cfg=json.loads(Path('configs/rescan_third_period_v1.json').read_text(encoding='utf-8'))
    out=Path('data/rescan_three_period_residual_v1');out.mkdir(exist_ok=False);records=[]
    for scene in cfg['scenes']:
        for period in ['1','2']:
            if period=='1':source=Path(cfg['prior_roots'][scene])/'predictions/prediction.ply'
            else:source=Path(json.loads((run_root(scene,'initial_archive')/'native_groups.json').read_text(encoding='utf-8'))['prediction'])
            points=read_ply(source);proposal=np.full(len(points),-1,np.int32);tokens=[]
            for r in public[scene][period]:
                if not r['eligible']:continue
                z=np.load(fragments/r['path']);indices=z['point_indices'];assert np.all(proposal[indices]==-1)
                proposal[indices]=len(tokens);tokens.append(r['token'])
            xyz=np.column_stack([points[f] for f in ['x','y','z']]);rgb=np.column_stack([points[f] for f in ['red','green','blue']])
            payload={'xyz':xyz,'rgb':rgb,'proposal_index':proposal,'predicted_class':points['class_idx'],
                     'residual_point_indices':np.flatnonzero(proposal<0)}
            if all(f in points.dtype.names for f in ['nx','ny','nz']):payload['normal']=np.column_stack([points[f] for f in ['nx','ny','nz']])
            dest=out/'public'/scene/(period+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
            z=np.load(dest);assert len(z['xyz'])==len(points) and np.array_equal(z['xyz'],xyz)
            assert len(z['residual_point_indices'])+np.sum(proposal>=0)==len(points)
            assert 'instance_idx' not in z.files
            records.append({'scene':scene,'period':period,'path':str(dest.relative_to(out)),
                'points':len(points),'covered_by_eligible_proposal':int(np.sum(proposal>=0)),
                'residual_points':int(np.sum(proposal<0)),'proposal_tokens':tokens,
                'source':str(source),'source_sha':sha(source),'sha256':sha(dest)})
    save(out/'public.json',{'records':records,'scope':'All current geometry retained. Residual pool includes background, missed objects and small fragments; no GT class/ID selection. Predicted classes are soft metadata, not a mandatory gate. IDs are anonymous fragment indices only.'})
    save(out/'audit.json',{'passed':True,'all_points_retained':True,'current_GT_read':False,
        'source_sha':sha(__file__),'scope':'Development input representation fix after baseline evaluation; no new inference, ID gain or independent holdout claim.'})
    save(out/'seal.json',{'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print([(r['scene'],r['period'],r['points'],r['residual_points']) for r in records])


if __name__=='__main__':main()
