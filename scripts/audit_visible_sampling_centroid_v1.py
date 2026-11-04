from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    cfg=read('configs/visible_sampling_centroid_v1.json');out=Path(cfg['output_root']);pool=Path(cfg['pool_root']);checks=0;count=0;maxeq=0.;same=0
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h;checks+=1
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h;checks+=1
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);old=read(pool/pth.parent.name/'predictions.json');assert p['pair']['current_stem'] not in p['source_frames'] and len(p['source_frames'])==3;checks+=1
        assert len(p['rows'])==len(old['rows']);checks+=1
        for r,pr in zip(p['rows'],old['rows']):
            assert r['scores']['pair_only']==pr['base'];checks+=1
            assert np.isfinite(list(r['scores'].values())).all();checks+=1
            d=r['diagnosis'];assert d['rendered_pixels']==d['surface_pixels']+d['free_pixels']+d['unknown_pixels'];checks+=1
            gap=abs(r['scores']['ordinary_rendered_pixels']-r['scores']['visible_centroid_plus_scatter']);maxeq=max(maxeq,gap);assert gap<1e-12;checks+=1
            assert d['moments']['visible_mean']<=d['moments']['pixel']+1e-8;checks+=1;count+=1
        for m,i in p['choices'].items():assert i==min(range(len(p['rows'])),key=lambda j:p['rows'][j]['scores'][m]);checks+=1
        same+=p['choices']['ordinary_rendered_pixels']==p['choices']['visible_centroid']
        assert p['choices']['ordinary_rendered_pixels']==p['choices']['visible_centroid_plus_scatter'];checks+=1
    assert count==3965;checks+=1
    save(out/'audit.json',dict(passed=True,checks=checks,candidates=count,maximum_moment_equivalence_error=maxeq,visible_mean_same_choice_as_pixels_pairs=same))
    save(out/'final_evidence_index.json',dict(files={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='final_evidence_index.json'},analysis_sources={str(p):sha(p) for p in Path('scripts').glob('*visible_sampling_centroid_v1.py')},report_sha=sha('research/visible_sampling_centroid_v1_2026-10-24.md')))
    print('AUDIT_PASS',checks,'CANDIDATES',count,'MEAN_SAME_PIXEL',same,'MAX_EQ',maxeq)


if __name__=='__main__':main()
