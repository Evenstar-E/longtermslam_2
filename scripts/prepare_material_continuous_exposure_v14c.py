"""Versioned corrected run. Reuse geometry only, NEVER old values or fits."""
import json
from pathlib import Path
import numpy as np
from scripts.run_textured_material_comparison_v1 import sha


def geometry_candidates(cfg,v,old):
    result={};nextq=0
    for path in sorted(Path(cfg['geometry_checkpoint_root']).glob(f'{v}_block_*.npz')):
        with np.load(path) as f:
            start=int(path.stem.rsplit('_',1)[1]);assert start==nextq
            assert np.array_equal(f['triples'],old['triples'][start:start+len(f['triples'])])
            for q in range(start,start+len(f['triples'])):
                # Read only transform/stage, never old objective or task metric.
                s=f[f'{q}_stage'];mask=s<=1;result[q]=(f[f'{q}_transforms'][mask],s[mask])
            nextq+=len(f['triples'])
    return result


def geometry_protocol(cfg):
    root=Path(cfg['geometry_checkpoint_root']);protocol=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'))
    for p,h in protocol['sources'].items():assert sha(p)==h,p
    return dict(protocol_sha=sha(root/'frozen_protocol.json'),blocks={str(p):sha(p) for p in root.glob('*_block_*.npz')},reuse='ONLY stage<=1 transforms; all energies and continuous optimization recomputed. Old stage2 endpoints discarded.')


def main():
    source=Path('scripts/run_material_continuous_surface_correspondence_v14.py').read_text(encoding='utf-8')
    source=source.replace('from inspection_decisions.material_continuous_surface_correspondence_v14 import DensePair,ContinuousTriple,fit_pose,spectral_proposals,PAIRS','from inspection_decisions.material_continuous_exposure_v14c import DensePair,ContinuousTriple,fit_pose,spectral_proposals,PAIRS\nfrom scripts.prepare_material_continuous_exposure_v14c import geometry_candidates,geometry_protocol')
    source=source.replace("cfg.update(json.loads(Path('configs/material_continuous_surface_correspondence_v14.json').read_text(encoding='utf-8')));return cfg", "cfg.update(json.loads(Path('configs/material_continuous_surface_correspondence_v14.json').read_text(encoding='utf-8')));cfg.update(json.loads(Path('configs/material_continuous_exposure_v14c.json').read_text(encoding='utf-8')));cfg['contract']+=' '+cfg['exposure_contract'];return cfg")
    source=source.replace("triples=old['triples'][:1] if profile else old['triples']", "geometry=geometry_candidates(cfg,v,old) if not profile else {}\n    triples=old['triples'][:1] if profile else old['triples']")
    anchor="U=unique_poses(list(retained)+new);stage=np.r_[np.zeros(len(retained),int),np.ones(len(U)-len(retained),int)];value=sc.score(U)"
    replacement="""U=unique_poses(list(retained)+new);stage=np.r_[np.zeros(len(retained),int),np.ones(len(U)-len(retained),int)]
        if q in geometry:
            saved,saved_stage=geometry[q]
            assert len(saved)==len(U) and np.max(abs(saved-U))<1e-9 and np.array_equal(saved_stage,stage),'geometric checkpoint mismatch'
            U=saved.copy()
        value=sc.score(U)"""
    assert source.count(anchor)==1;source=source.replace(anchor,replacement)
    source=source.replace("'scripts/score_material_continuous_surface_correspondence_v14.py'", "'scripts/score_material_continuous_surface_correspondence_v14c.py','inspection_decisions/material_continuous_surface_cache_v14b.py','inspection_decisions/material_continuous_exposure_v14c.py','configs/material_continuous_exposure_v14c.json','scripts/check_material_continuous_exposure_v14c.py','scripts/prepare_material_continuous_exposure_v14c.py'")
    source=source.replace("scope=cfg['contract']));start=time.perf_counter()", "scope=cfg['contract'],geometric_reuse=geometry_protocol(cfg)));start=time.perf_counter()")
    scorer=Path('scripts/score_material_continuous_surface_correspondence_v14.py').read_text(encoding='utf-8').replace('from scripts.run_material_continuous_surface_correspondence_v14 import','from scripts.run_material_continuous_surface_correspondence_v14c import')
    for path,text in [('scripts/run_material_continuous_surface_correspondence_v14c.py',source),('scripts/score_material_continuous_surface_correspondence_v14c.py',scorer)]:
        compile(text,path,'exec');p=Path(path);assert not p.exists();p.write_text(text,encoding='utf-8')
    print('V14c corrected runner/scorer generated')


if __name__=='__main__':main()
