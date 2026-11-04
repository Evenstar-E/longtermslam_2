import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from scripts.run_material_surface_distribution_v15b import configuration
from scripts.prepare_material_surface_fields_v15 import arrays,load_fields,model,public_normalizer
from inspection_decisions.material_surface_distribution_matching_v15 import encode_record,add_area
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);parent=Path(c['parent_root']);p=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'));checks=0
    for group in ['source_hashes','evidence_hashes']:
        for path,h in p[group].items():assert sha(path)==h,path;checks+=1
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
    for path,h in seal['sources'].items():assert sha(path)==h,path;checks+=1
    for path,h in seal['files'].items():assert sha(root/path)==h,path;checks+=1
    field=Path(c['field_cache']);fs=json.loads((field/'field_seal.json').read_text(encoding='utf-8'));assert not fs['private_labels_read'] and not fs['new_training'];checks+=1
    for path,h in fs['sources'].items():assert sha(path)==h,path;checks+=1
    for path,h in fs['files'].items():assert sha(field/path)==h,path;checks+=1
    assert sha(c['checkpoint'])==fs['checkpoint_sha'];checks+=1;torch.set_num_threads(1);torch.set_flush_denormal(True);net=model(c);fc=json.loads(Path(c['feature_config']).read_text(encoding='utf-8'));maxfeature=0.;maxreference=0.;reencoded=0;pixels=0
    for v in c['variants']:
        frames,views,banks,dense=arrays(c,v);load_fields(c,v,banks,dense)
        for e in range(3):
            for i,b in enumerate(dense[e]):
                original=b['area'].copy();add_area(b,views[e][i],i);assert np.max(abs(original-b['area']))<1e-12;checks+=1;pixels+=len(b['xyz'])
            i=(e*7+3)%24
            for record,fr in [(banks[e][i],frames[e]),(dense[e][i],views[e][i])]:
                ix=np.linspace(0,len(record['xyz'])-1,16,dtype=int);sub={k:record[k][ix] for k in ['xyz','uv','frame']};fresh=encode_record(net,sub,fr,fc,c).astype(np.float16).astype(np.float32);fresh/=np.maximum(np.linalg.norm(fresh,axis=1,keepdims=True),1e-8);fresh=fresh.astype(np.float16).astype(np.float32);fresh/=np.maximum(np.linalg.norm(fresh,axis=1,keepdims=True),1e-8);error=float(np.max(abs(fresh-record['desc'][ix])));assert error<.002,error;maxfeature=max(maxfeature,error);reencoded+=len(ix);checks+=1
            f=(e+1)%3;a=banks[e][0];sub=dict(xyz=a['xyz'][:1],desc=a['desc'][:1]);z=public_normalizer(sub,dense[f]);error=float(np.max(abs(z-a[f'normalizer_{f}'][:1])));assert error<1e-6,error;maxreference=max(maxreference,error);checks+=1
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==768 and scores['source_centers_verified']==147456;checks+=1;old='results/surface_measure_first_return_v1/scores.json';assert sha(old)=='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';checks+=1
    report=Path('research/material_surface_distribution_results_v15_2026-10-27.md');assert report.is_file() and (root/'task_comparison.png').is_file();checks+=1
    sources=list(seal['sources'])+['scripts/prepare_material_surface_fields_v15.py','scripts/diagnose_material_surface_distribution_v15.py','scripts/probe_material_surface_seed_failure_v15.py','scripts/analyze_material_surface_retention_v15.py','scripts/report_material_surface_distribution_v15.py','scripts/check_material_surface_distribution_v15.py','scripts/prepare_pointdsc_taskscale_v15.py',__file__];files=[path for path in root.iterdir() if path.is_file() and path.name!='completion_integrity.json']+[report,Path('research/material_surface_distribution_model_v15_2026-10-27.md'),Path('research/material_surface_distribution_decision_v15_2026-10-27.md'),field/'field_seal.json',Path('results/material_surface_region_checks_v15b.json')]
    for folder in ['pointdsc_material_reference_v15','pointdsc_taskscale_reference_v15']:
        completion=Path('results')/folder/'completion_integrity.json';ref=json.loads(completion.read_text(encoding='utf-8'));assert ref['complete'] and ref['official_sources_unchanged'];checks+=1
        proto=json.loads((Path('results')/folder/'frozen_protocol.json').read_text(encoding='utf-8'));assert sha('third_party/PointDSC_official_v1/source_manifest.json')==proto['official_manifest_sha'] and sha(field/'field_seal.json')==proto['field_seal_sha'];checks+=1
        for group in ['source_hashes','evidence_hashes']:
            for path,h in ref[group].items():assert sha(path)==h,path;checks+=1
        files.append(completion)
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=768,native_pixels_area_verified=pixels,reencoded_source_native_queries=reencoded,max_feature_reencode_error=maxfeature,max_global_reference_error=maxreference,old_192_sha=sha(old),new_training=False,old_evidence_unchanged=True,source_hashes={path:sha(path) for path in sources},evidence_hashes={str(path):sha(path) for path in files},scope='Integrity/reconstruction audit, not independent rerun or novelty/generalization confirmation. PNG reviewed separately; PDF exported.'))
    print('V15 COMPLETION',checks,'pixels',pixels,'reencode',reencoded,'error',maxfeature)


if __name__=='__main__':main()
