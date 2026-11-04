"""Check geometry, label separation, split leakage, and material correspondences."""
from pathlib import Path
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.textured_material_dataset_v1 import make_scene,view_specs,cast,albedo,inverse,transform
from scripts.build_textured_material_dataset_v1 import read,save,sha
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch,load_training_pair


def main():
    root=Path('data/textured_material_v1');cfg=read(root/'config.json');seal=read(root/'dataset_seal.json')
    assert seal['complete'] and seal['source_hashes_unchanged'];checks=0
    for p,h in seal['sha256'].items():assert sha(root/p)==h;checks+=1
    for p,h in read(root/'source_manifest.json').items():assert sha(p)==h;checks+=1
    pair_seal=read(root/'pair_labels_v2_seal.json')
    for p,h in pair_seal['sha256'].items():assert sha(root/p)==h;checks+=1
    v3=read(root/'pair_labels_v3_seal.json');assert v3['previous_pair_seal_sha256']==sha(root/'pair_labels_v2_seal.json');checks+=1
    for p,h in v3['sha256'].items():assert sha(root/p)==h;checks+=1
    index=read(root/'dataset_index.json');split_keys={}
    for split in cfg['splits']:
        rs=[r for r in index if r['split']==split]
        split_keys[split]={field:set(x for r in rs for x in r[field]) for field in ['shape_keys','texture_keys']}
    splits=list(split_keys)
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            for f in ['shape_keys','texture_keys']:assert not(split_keys[a][f]&split_keys[b][f]);checks+=1
    allowed={'rgb','rgb_flat','rgb_repeated','rgb_changed','depth','K','C','anonymous_slot'}
    max_projection=0.;max_material=0.;max_albedo=0.;noise_max=0.;all_changes=0
    visible=outside=occluded=total=0;rich_std=[];surface_pixels=0;anonymous_kept=0;slot_errors=0;slot_background=0;pair_count=0
    for si,(split,count) in enumerate(cfg['splits'].items()):
        for wi in range(count):
            tag=f'{split}_{wi:03d}';dest=root/tag;scene=make_scene(cfg,si,wi);views=view_specs(scene,cfg)
            frames=read(dest/'frames.json')
            for rec,view in zip(frames,views):
                with np.load(dest/rec['observation']) as f:obs=dict(f)
                with np.load(dest/rec['labels']) as f:truth=dict(f)
                assert set(obs)==allowed;checks+=1
                owner=truth['instance'];fg=owner>=0;surface_pixels+=int(fg.sum())
                kept=fg&(obs['anonymous_slot']>=0);anonymous_kept+=int(kept.sum())
                slot_background+=int(((~fg)&(obs['anonymous_slot']>=0)).sum())
                expected=scene['permutations'][view['epoch']][owner[kept]]
                slot_errors+=int((expected!=obs['anonymous_slot'][kept]).sum())
                noise_max=max(noise_max,float(np.max(abs(obs['depth']-truth['depth_exact']))))
                assert np.max(abs(obs['depth']-truth['depth_exact']))<=cfg['depth_noise_bound_m']+4e-6;checks+=1
                changes=truth['changed_material_mask'];all_changes+=int(changes.sum())
                if view['epoch']<2:assert not changes.any() and np.array_equal(obs['rgb'],obs['rgb_changed']);checks+=1
                assert np.array_equal(obs['rgb'][~changes],obs['rgb_changed'][~changes]);checks+=1
                # Independent projection of exact local material coordinates through GT pose.
                yy,xx=np.nonzero(fg);select=np.linspace(0,len(yy)-1,min(128,len(yy)),dtype=int)
                yy,xx=yy[select],xx[select];ids=owner[yy,xx];p=truth['material_xyz'][yy,xx].astype(float);world=np.zeros_like(p)
                for asset in np.unique(ids):
                    ix=ids==asset;world[ix]=transform(p[ix],scene['poses'][view['epoch'],asset])
                    boxes=scene['geometry'][asset];pr=truth['primitive'][yy[ix],xx[ix]];face=truth['face'][yy[ix],xx[ix]]
                    plane=boxes[pr,face%2,face//2];err=abs(p[ix][np.arange(ix.sum()),face//2]-plane)
                    assert err.max(initial=0)<1e-6;checks+=1
                    predicted=albedo(p[ix],face,scene['texture_keys'][asset]);ref=truth['albedo'][yy[ix],xx[ix]]
                    max_albedo=max(max_albedo,float(abs(predicted-ref).max(initial=0)))
                cam=inverse(world,obs['C']);uv=cam[:,:2]/cam[:,2,None]*[obs['K'][0,0],obs['K'][1,1]]+obs['K'][:2,2]
                max_projection=max(max_projection,float(abs(uv-np.c_[xx,yy]).max(initial=0)))
                assert np.max(abs(uv-np.c_[xx,yy]),initial=0)<1e-3;checks+=1
                if rec['kind']=='detail':
                    visible_ids=np.unique(owner[fg]);asset=int(visible_ids[np.argmax([(owner==i).sum() for i in visible_ids])])
                    mask=owner==asset;rich_std.append(float(truth['albedo'][mask].std(0).mean()))
            for pi,pr in enumerate(read(dest/'supervision/pairs_v3.json')):
                pair_count+=1
                with np.load(dest/pr['labels']) as f:pair=dict(f)
                if pi<18:
                    with np.load(dest/'supervision'/f'pair_{pi:03d}.npz') as f:
                        for key in ['source_uv','instance','material_xyz','status']:
                            assert np.array_equal(pair[key],f[key]);checks+=1
                counts=np.bincount(pair['status'],minlength=3);visible+=int(counts[0]);outside+=int(counts[1]);occluded+=int(counts[2]);total+=int(counts.sum())
                target=views[pr['target']];cam=target['camera'];ii=np.flatnonzero(pair['status']==0)[::7]
                if len(ii):
                    hit=cast(scene,target['epoch'],cam['C'],pair['target_uv'][ii],cam['K'])
                    assert np.array_equal(hit['instance'],pair['instance'][ii]);checks+=1
                    err=np.linalg.norm(hit['material_xyz']-pair['material_xyz'][ii],axis=1);max_material=max(max_material,float(err.max()))
                    assert err.max()<1e-4;checks+=1
    # Whole-scene rigid-coordinate change must preserve material points and z depth.
    s=make_scene(cfg,0,0);s['occluders']=np.empty((0,2,3));v=view_specs(s,cfg)[7];cam=v['camera']
    uv=np.array([[64,64],[128,128],[190,130],[110,180]],float);a=cast(s,v['epoch'],cam['C'],uv,cam['K'])
    G=np.eye(4);G[:3,:3]=Rotation.from_rotvec([.23,-.31,.12]).as_matrix();G[:3,3]=[2,-1,3]
    s2=copy.deepcopy(s);s2['poses']=G@s['poses'];b=cast(s2,v['epoch'],G@cam['C'],uv,cam['K'])
    assert np.array_equal(a['instance'],b['instance']);assert np.allclose(a['depth_exact'],b['depth_exact'],atol=1e-10)
    assert np.allclose(a['material_xyz'],b['material_xyz'],atol=1e-10);checks+=3
    assert all_changes>0 and visible>0 and outside>0 and occluded>0 and min(rich_std)>5;checks+=5
    assert max_albedo<.02 and (a['instance']>=0).any();checks+=2
    packets,_=load_scene_epoch(root,'test_000',1,shuffle_seed=333)
    assert all(set(p)=={'rgb','depth','depth_valid','K','C','anonymous_slot'} for p in packets);checks+=1
    assert all(np.all(p['depth'][~p['depth_valid']]==0) for p in packets);checks+=1
    inputs,labels=load_training_pair(root,'train_000',0)
    assert 'material_xyz' in labels and 'material_xyz' not in inputs['source'];checks+=1
    for split in ['validation','test']:
        try:load_training_pair(root,split+'_000',0)
        except ValueError:checks+=1
        else:raise AssertionError('Holdout labels entered training adapter')
    report=dict(passed=True,checks=checks,scenes=len(index),asset_tasks=sum(r['asset_tasks'] for r in index),
        frames=sum(r['frames'] for r in index),rgb_variants_per_frame=4,
        pair_sets=pair_count,sampled_correspondences=total,
        visible_correspondences=visible,outside_correspondences=outside,occluded_correspondences=occluded,
        max_projection_error_px=max_projection,max_material_correspondence_error_m=max_material,
        max_albedo_reconstruction_error_255=max_albedo,max_depth_noise_m=noise_max,
        changed_pixels=all_changes,detail_albedo_std_min=float(min(rich_std)),detail_albedo_std_median=float(np.median(rich_std)),
        anonymous_slot_foreground_coverage=anonymous_kept/surface_pixels,anonymous_slot_wrong_retained_pixels=slot_errors,
        anonymous_slot_background_pixels=slot_background,
        split_disjoint_geometry_and_texture_instances=True,private_supervision_not_in_observation_npz=True,
        no_model_trained=True,shared_procedural_families_not_unseen_shape_categories=True,
        limitations=cfg['contract'],audit_source_sha256=sha(Path(__file__)),
        model_input_adapter_sha256=sha('inspection_decisions/textured_material_data_io_v1.py'),
        acquisition_group_not_model_input=True,training_adapter_refuses_holdout_labels=True)
    save(root/'audit.json',report);print(report,flush=True)


if __name__=='__main__':main()
