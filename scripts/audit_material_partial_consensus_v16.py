"""Seal/provenance/budget audit; no redundant old training or area rebuild."""
import json
from pathlib import Path
import numpy as np
from scripts.run_material_partial_consensus_v16 import configuration,load,FAMILIES
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);parent=Path(c['parent_root']);old=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'));checks=0
    for group in ['source_hashes','evidence_hashes']:
        for path,h in old[group].items():assert sha(path)==h,path;checks+=1
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));protocol=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'] and not seal['new_training'];checks+=1
    for path,h in seal['sources'].items():assert sha(path)==h,path;checks+=1
    for path,h in seal['files'].items():assert sha(root/path)==h,path;checks+=1
    assert sha(parent/'completion_integrity.json')==protocol['parent_completion_sha'];checks+=1;field=Path(c['field_cache']);assert sha(field/'field_seal.json')==protocol['field_seal_sha'];checks+=1;fs=json.loads((field/'field_seal.json').read_text(encoding='utf-8'))
    for path,h in fs['sources'].items():assert sha(path)==h,path;checks+=1
    for path,h in fs['files'].items():assert sha(field/path)==h,path;checks+=1
    assert sha(c['checkpoint'])==fs['checkpoint_sha'];checks+=1;paircount=0;rawcounts={f:0 for f in FAMILIES}
    for v in c['variants']:
        z=load(root/f'{v}_pair_proposals.npz');diag=json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8'));prefixes=[key[:-len('_f_global_seeds')] for key in z if key.endswith('_f_global_seeds')];paircount+=len(prefixes);assert len(prefixes)==diag['statistics']['pair_count'];checks+=1
        for prefix in prefixes:
            for direction in ['f','b']:
                for f in ['global','local']:assert len(z[prefix+'_'+direction+'_'+f+'_seeds'])<=c['seed_budget'];checks+=1
            for f in FAMILIES:
                n=len(z[prefix+'_'+f+'_raw']);assert n<=2*c['seed_budget'] and len(z[prefix+'_'+f])<=c['global_keep'];rawcounts[f]+=n;checks+=1
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==768 and scores['source_centers_verified']==147456;checks+=1
    checks+=scores['checks'];diagnosis=json.loads((root/'diagnosis_summary.json').read_text(encoding='utf-8'));checks+=diagnosis['checks'];old192='results/surface_measure_first_return_v1/scores.json';assert sha(old192)=='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';checks+=1
    sources=list(seal['sources'])+['scripts/diagnose_material_partial_consensus_v16.py','scripts/report_material_partial_consensus_v16.py','scripts/check_material_mixture_bias_v16.py',__file__];reports=[Path('research')/f'material_partial_consensus_{name}_v16_2026-10-27.md' for name in ['model','results','decision']]
    for p in reports:assert p.is_file();checks+=1
    files=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+reports+[field/'field_seal.json',parent/'completion_integrity.json',Path('results/material_mixture_bias_check_v16.json')]
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=768,pair_count=paircount,actual_raw_poses=rawcounts,source_centers_verified=147456,old_192_sha=sha(old192),old_evidence_unchanged=True,new_training=False,source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files},scope='Full inference-score-budget-provenance audit with selected-pair reconstruction; not independent rerun, generalization or originality confirmation.'))
    print('V16 COMPLETION',checks,'pairs',paircount,'raw',rawcounts)


if __name__=='__main__':main()
