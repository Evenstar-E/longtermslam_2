"""Frozen learned/handcrafted/official controls on all four old held-out scenes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import json
import time
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions import textured_material_matching_v1 as base
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor
from inspection_decisions.material_pretrained_v1 import PretrainedLocal
from inspection_decisions.material_feature_comparison_v1 import measured_epoch,feature_epochs,official_matches,compare
from scripts.run_textured_material_comparison_v1 import save,sha


METHODS=['metric_rootsift','untrained','contrastive','learned_fusion','learned_visibility','learned_material','superpoint_lightglue']


def main():
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'));torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);root=Path(cfg['comparison_root']);root.mkdir(parents=True,exist_ok=False)
    training=Path(cfg['training_root']);ts=json.loads((training/'training_seal.json').read_text(encoding='utf-8'))
    assert ts['complete'] and not ts['heldout_labels_read']
    for f,h in ts['checkpoints'].items():assert sha(training/f)==h
    pretrained_manifest=json.loads(Path('third_party/material_local_matching_v1/source_manifest.json').read_text(encoding='utf-8'))
    for f,h in pretrained_manifest['files'].items():assert sha(Path('third_party/material_local_matching_v1')/f)==h['sha256']
    models={}
    for name,file in [('untrained','untrained.pt'),('contrastive','contrastive_31.pt'),('full','full_31.pt')]:
        checkpoint=torch.load(training/file,map_location='cpu',weights_only=True)
        model=MaterialDescriptor(cfg['embedding_dim']);model.load_state_dict(checkpoint['state_dict']);models[name]=model.eval()
    pretrained=PretrainedLocal()
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    # Dense normalized embeddings use a common mutual-NN matcher without a ratio
    # gate; official LightGlue retains its pretrained default confidence 0.1.
    bcfg.update(ratio=1.,descriptor_max=2.)
    scenes=[r['scene'] for r in json.loads((Path(cfg['dataset'])/'dataset_index.json').read_text(encoding='utf-8')) if r['split']!='train']
    sources={p:sha(p) for p in ['configs/material_joint_descriptor_v1.json',__file__,
        'inspection_decisions/material_joint_descriptor_v1.py','inspection_decisions/material_feature_comparison_v1.py',
        'inspection_decisions/material_pretrained_v1.py','inspection_decisions/textured_material_matching_v1.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,base_config=bcfg,methods=METHODS,scenes=scenes,sources=sources,
        training_seal_sha=sha(training/'training_seal.json'),dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),
        official_manifest_sha=sha('third_party/material_local_matching_v1/source_manifest.json'),
        primary_seed=31,other_seeds='Training replication; descriptor metrics evaluated separately',
        comparison='All 24 assets and all 3 periods; old held-out scenes already analyzed, development only',
        head_controls='Learned fusion, visibility and material have identical features/pose proposals; geometry coefficient=1 in all three',
        official_control='Official dense SuperPoint descriptors at the common measured samples; original LightGlue on real per-image pixel coordinates, not the full published benchmark'))
    start=time.perf_counter()
    for scene in scenes:
        dest=root/scene;dest.mkdir()
        def progress(msg):
            save(dest/'progress.json',dict(message=msg,elapsed_s=time.perf_counter()-start))
            print('COMPARE',scene,msg,round(time.perf_counter()-start,1),flush=True)
        gp=None
        for variant in cfg['variants']:
            frames=[load_scene_epoch(cfg['dataset'],scene,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
            geom=[base.extract_epoch(f,bcfg,cfg['seed']+e) for e,f in enumerate(frames)]
            if gp is None:gp=base.geometric_pools(geom,bcfg)
            pieces=[measured_epoch(f,cfg) for f in frames]
            feats,lg_frames=feature_epochs(pieces,models,pretrained,cfg,geom)
            progress(variant+' all features ready')
            for feature in ['metric_rootsift','untrained','contrastive','full','superpoint']:
                precomputed=official_matches(lg_frames,pretrained,feats[feature]) if feature=='superpoint' else None
                pred,diag,pool=compare(feats[feature],frames,bcfg,gp,cfg['seed'],model=models['full'] if feature=='full' else None,precomputed=precomputed)
                for method,p in pred.items():
                    name=('learned_'+method if feature=='full' else 'superpoint_lightglue' if feature=='superpoint' else feature)
                    np.savez_compressed(dest/f'{variant}_{name}.npz',**p)
                np.savez_compressed(dest/f'{variant}_{feature}_pools.npz',**pool)
                save(dest/f'{variant}_{feature}_diagnostics.json',diag)
                progress(variant+' '+feature+' complete')
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,scenes=scenes,methods=METHODS,sources=sources,
        elapsed_s=time.perf_counter()-start,heldout_labels_read=False,
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('LEARNED COMPARISON SEALED',len(scenes),'scenes',flush=True)


if __name__=='__main__':main()
