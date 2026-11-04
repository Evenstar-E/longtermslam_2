"""Parallel four-scene evaluation of the training-only probability repair."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions import textured_material_matching_v1 as base
from inspection_decisions.material_joint_descriptor_v2 import FactoredMaterialDescriptor
from inspection_decisions.material_feature_comparison_v1 import measured_epoch,compare
from inspection_decisions.material_feature_comparison_v2 import learned_epochs
from scripts.run_textured_material_comparison_v1 import save,sha


def job(scene,cfg,bcfg,output):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    root=Path(output)/scene;root.mkdir();training=Path('results/material_joint_descriptor_training_v2');models={}
    for name in ['factorized','balanced']:
        checkpoint=torch.load(training/f'{name}_31.pt',map_location='cpu',weights_only=True)
        model=FactoredMaterialDescriptor(cfg['embedding_dim'],checkpoint['persistence_prior'],checkpoint['balanced'])
        model.load_state_dict(checkpoint['state_dict']);models[name]=model.eval()
    gp=None;start=time.perf_counter()
    for v in cfg['variants']:
        frames=[load_scene_epoch(cfg['dataset'],scene,e,v,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        geom=[base.extract_epoch(f,bcfg,cfg['seed']+e) for e,f in enumerate(frames)]
        if gp is None:gp=base.geometric_pools(geom,bcfg)
        pieces=[measured_epoch(f,cfg) for f in frames];features=learned_epochs(pieces,models,geom)
        for name,model in models.items():
            pred,diag,pool=compare(features[name],frames,bcfg,gp,cfg['seed'],model=model)
            for method,p in pred.items():np.savez_compressed(root/f'{v}_{name}_{method}.npz',**p)
            np.savez_compressed(root/f'{v}_{name}_pools.npz',**pool);save(root/f'{v}_{name}_diagnostics.json',diag)
            save(root/'progress.json',dict(variant=v,model=name,elapsed_s=time.perf_counter()-start))
            print('FACTORED COMPARE',scene,v,name,round(time.perf_counter()-start,1),flush=True)
    return scene


def main():
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'))
    root=Path('results/material_joint_descriptor_comparison_v2');root.mkdir(parents=True,exist_ok=False)
    training=Path('results/material_joint_descriptor_training_v2');seal=json.loads((training/'training_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['checkpoints'].items():assert sha(training/p)==h
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'));bcfg.update(ratio=1.,descriptor_max=2.)
    scenes=[r['scene'] for r in json.loads((Path(cfg['dataset'])/'dataset_index.json').read_text(encoding='utf-8')) if r['split']!='train']
    methods=[f'{model}_{m}' for model in ['factorized','balanced'] for m in ['fusion','visibility','material']]
    source_paths=[__file__,'configs/material_joint_descriptor_v1.json','inspection_decisions/material_joint_descriptor_v1.py',
        'inspection_decisions/material_joint_descriptor_v2.py','inspection_decisions/material_feature_comparison_v1.py',
        'inspection_decisions/material_feature_comparison_v2.py','inspection_decisions/textured_material_matching_v1.py']
    sources={p:sha(p) for p in source_paths}
    save(root/'frozen_protocol.json',dict(config=cfg,base_config=bcfg,scenes=scenes,methods=methods,sources=sources,
        method_features={f'{model}_{m}':model for model in ['factorized','balanced'] for m in ['fusion','visibility','material']},
        visibility_methods=[m for m in methods if not m.endswith('_fusion')],
        paired_controls=[dict(baseline=model+'_fusion',candidate=model+'_'+m) for model in ['factorized','balanced'] for m in ['visibility','material']],
        dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),training_seal_sha=sha(training/'training_seal.json'),
        stage='Old held-out scenes now DEVELOPMENT; repair motivated only by TRAIN calibration',primary_seed=31,
        formula='P(v,j,c)=P(v)P(j|v)P(c|v,j); balanced appearance evidence corrected to train prior at inference'))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,s,cfg,bcfg,str(root)) for s in scenes]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,scenes=scenes,methods=methods,sources=sources,
        elapsed_s=time.perf_counter()-start,heldout_labels_read=False,
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('FACTORED COMPARISON SEALED',flush=True)


if __name__=='__main__':main()
