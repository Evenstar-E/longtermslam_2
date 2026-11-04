from pathlib import Path
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/material_observation_consistency_v1');checks=0
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h;checks+=1
    poses=0;queries=0;rejected={'raw':0,'stable':0};escape_rejected={'raw':0,'stable':0};changed={}
    for folder in sorted(root.glob('pair_*')):
        p=read(folder/'predictions.json');old=read(Path('results/ray_material_full_resolution_v1')/folder.name/'predictions.json')
        assert len(p['rows'])==len(old['pool']);checks+=1;poses+=len(p['rows'])
        for i,(row,entry) in enumerate(zip(p['rows'],old['pool'])):
            assert row['scores']['nn']==entry['scores']['old_nn'] and row['scores']['visibility']==entry['scores']['ordinary_overlap_rgb'];checks+=1
            for direction,s in row['support'].items():
                assert s['raw_positive']+s['measured_absence']+s['unknown']==s['points'];checks+=1
                assert 0<=s['stable_positive']<=s['raw_positive'];checks+=1;queries+=s['points']
            for gate,field in [('raw','raw_positive'),('stable','stable_positive')]:
                assert row['gates'][gate]==all(v[field]>=1 for v in row['support'].values());checks+=1
                rejected[gate]+=int(not row['gates'][gate]);escape_rejected[gate]+=int(i in old['escape_indices'] and not row['gates'][gate])
        for m in ['nn','visibility']:
            assert p['choices'][m]==min(range(len(p['rows'])),key=lambda i:p['rows'][i]['scores'][m]);checks+=1
            for gate in ['raw','stable']:
                name=m+'_'+gate+'_gate';valid=[i for i,r in enumerate(p['rows']) if r['gates'][gate]]
                expected=min(valid,key=lambda i:p['rows'][i]['scores'][m]) if valid else None
                assert p['choices'][name]==expected;checks+=1;changed[name]=changed.get(name,0)+int(expected!=p['choices'][m])
    assert poses==3941;checks+=1
    result=dict(status='PASS',checks=checks,poses=poses,point_projection_instances=queries,rejected=rejected,escape_rejected=escape_rejected,changed_selection_pairs=changed,
                analysis_sources={str(p):sha(p) for p in [Path(__file__),Path('scripts/check_material_observation_consistency_v1.py'),Path('scripts/score_material_observation_consistency_v1.py')]})
    save(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
