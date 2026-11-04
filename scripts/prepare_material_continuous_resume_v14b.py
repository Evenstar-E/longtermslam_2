"""Generate exact-cache runner/scorer and reconstruct completed public blocks."""
import json
from pathlib import Path
import numpy as np
from scripts.run_textured_material_comparison_v1 import sha


def load_completed(cfg,v,old):
    done={};root=Path(cfg['resume_root']);nextq=0
    for path in sorted(root.glob(f'{v}_block_*.npz')):
        with np.load(path) as f:block=dict(f)
        start=int(path.stem.rsplit('_',1)[1]);assert start==nextq
        assert np.array_equal(block['triples'],old['triples'][start:start+len(block['triples'])])
        for h in range(len(block['triples'])):
            q=start+h;data={k:val for k,val in block.items() if k.startswith(f'{q}_')};assert f'{q}_values' in data
            done[q]=(block['costs'][h],block['transforms'][h],data)
        nextq+=len(block['triples'])
    return done


def resume_protocol(cfg):
    root=Path(cfg['resume_root']);p=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'))
    for name,h in p['sources'].items():assert sha(name)==h,name
    return dict(original_protocol_sha=sha(root/'frozen_protocol.json'),blocks={str(p):sha(p) for p in root.glob('*_block_*.npz')},equivalence_sha=sha('results/material_continuous_cache_equivalence_v14b.json'))


def main():
    source=Path('scripts/run_material_continuous_surface_correspondence_v14.py').read_text(encoding='utf-8')
    source=source.replace('from inspection_decisions.material_continuous_surface_correspondence_v14 import DensePair,ContinuousTriple,fit_pose,spectral_proposals,PAIRS','from inspection_decisions.material_continuous_surface_cache_v14b import DensePair,ContinuousTriple,fit_pose,spectral_proposals,PAIRS\nfrom scripts.prepare_material_continuous_resume_v14b import load_completed,resume_protocol')
    source=source.replace("cfg.update(json.loads(Path('configs/material_continuous_surface_correspondence_v14.json').read_text(encoding='utf-8')));return cfg", "cfg.update(json.loads(Path('configs/material_continuous_surface_correspondence_v14.json').read_text(encoding='utf-8')));cfg.update(json.loads(Path('configs/material_continuous_surface_cache_v14b.json').read_text(encoding='utf-8')));return cfg")
    source=source.replace("triples=old['triples'][:1] if profile else old['triples']", "completed=load_completed(cfg,v,old) if not profile else {}\n    triples=old['triples'][:1] if profile else old['triples']")
    anchor="A,B,C=modes(0,i,j),modes(1,i,k),modes(2,j,k);retained="
    replacement="""A,B,C=modes(0,i,j),modes(1,i,k),modes(2,j,k)
        if q in completed:
            cost,T,data=completed[q];U=data[f'{q}_transforms'];stage=data[f'{q}_stage'];val=data[f'{q}_values'];indices=np.flatnonzero(stage<=1);initial=int(indices[val[indices,3].argmin()])
            for col in [3,6]:
                endpoint,path=fit_pose(sc,U[initial],cfg,col);assert any(np.max(abs(endpoint-p))<1e-9 for p in U),'checkpoint fit mismatch';fits[f'{q}_{col}_initial']=U[initial];fits[f'{q}_{col}_T']=endpoint;fits[f'{q}_{col}_path']=path;counters['continuous_fits']+=1;counters['continuous_improved']+=int(path[-1]<path[0]-1e-10)
            rows.append((cost,T));payload.update(data)
            if q%8==7:
                np.savez_compressed(root/f'{v}_block_{q-7:04}.npz',triples=old['triples'][q-7:q+1],costs=np.array([r[0] for r in rows[-8:]]),transforms=np.array([r[1] for r in rows[-8:]]),**{key:vv for key,vv in payload.items() if q-7<=int(key.split('_')[0])<=q})
                save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,counters=counters,checkpoint_rows_reused=q+1))
            continue
        retained="""
    assert source.count(anchor)==1;source=source.replace(anchor,replacement)
    source=source.replace("'scripts/score_material_continuous_surface_correspondence_v14.py'", "'scripts/score_material_continuous_surface_correspondence_v14b.py','inspection_decisions/material_continuous_surface_cache_v14b.py','configs/material_continuous_surface_cache_v14b.json','scripts/check_material_continuous_cache_v14b.py','scripts/prepare_material_continuous_resume_v14b.py'")
    source=source.replace("scope=cfg['contract']));start=time.perf_counter()", "scope=cfg['contract'],execution_resume=resume_protocol(cfg)));start=time.perf_counter()")
    scorer=Path('scripts/score_material_continuous_surface_correspondence_v14.py').read_text(encoding='utf-8').replace('from scripts.run_material_continuous_surface_correspondence_v14 import','from scripts.run_material_continuous_surface_correspondence_v14b import')
    for path,text in [('scripts/run_material_continuous_surface_correspondence_v14b.py',source),('scripts/score_material_continuous_surface_correspondence_v14b.py',scorer)]:
        compile(text,path,'exec');p=Path(path);assert not p.exists();p.write_text(text,encoding='utf-8')
    print('V14 exact-cache resume runner/scorer generated')


if __name__=='__main__':main()
