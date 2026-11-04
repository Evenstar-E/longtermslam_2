"""Fixed-state right-action diagnostic, not extra inference or a GT pose test.

Both epoch poses are right-translated by the same canonical displacement.
Source surfaces are rebuilt. Cross-epoch geometry must remain invariant;
immutable history alone can distinguish these shared coordinate shifts.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_provenance_pose_hypotheses_v1 import load,read,save,sha
from inspection_decisions.reversible_surface_v2 import inverse,transform,costs


def main():
    root=Path('results/provenance_pose_hypotheses_v1');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    cfg=read(root/'config.json');rows=[]
    for scene in cfg['scenes']:
        H,O,_,_,_=load(scene,cfg);state=np.load(root/scene/'pool_coupled/state.npz')
        poses=state['poses'];z=[state['labels1'],state['labels2']]
        for i,h in enumerate(H):
            if h['class'] in [0,1,2,22] or h['raw_points']<128:continue
            subsets=[np.flatnonzero(a==i) for a in z]
            def evaluate(pp):
                out=[]
                for t in [0,1]:
                    idx=subsets[t];src=subsets[1-t]
                    obs={k:O[t][k][idx] for k in ['xyz','rgb','class']}
                    q=inverse(O[1-t]['xyz'][src],pp[1-t])
                    inc={'xyz':q,'rgb':O[1-t]['rgb'][src],'source_voxel':src,'tree':cKDTree(q) if len(q) else None}
                    cc,_,_,origin=costs(obs,[pp[t]],[h],[inc],cfg,True)
                    ext=np.full(len(idx),np.nan)
                    if len(src) and len(idx):
                        d,k=inc['tree'].query(inverse(obs['xyz'],pp[t]),workers=1)
                        ext=np.minimum(d/cfg['distance_scale_m'],1)**2+cfg['rgb_weight']*np.mean((obs['rgb']-inc['rgb'][k])**2,axis=1)+cfg['extension_cost']+cfg['semantic_weight']*(obs['class']!=h['class'])
                    out.append({'full':np.minimum(cc[:,0],cfg['null_cost']), 'ext':ext,
                                'extension_wins':origin[0]>=0,'mass':O[t]['mass'][idx],
                                'w':O[t]['mass'][idx]/O[t]['mass'].sum()})
                return out
            pp=poses[:,i].copy();base=evaluate(pp)
            old_anchor=np.array([transform(h['anchor'][None],T)[0] for T in pp])
            for axis in range(3):
                for sign in [-1,1]:
                    G=np.eye(4);G[axis,3]=sign*.02;newpose=np.array([T@G for T in pp]);new=evaluate(newpose)
                    anchor=np.array([transform(h['anchor'][None],T)[0] for T in newpose])
                    ext_errors=[]
                    for b,n in zip(base,new):
                        ok=np.isfinite(b['ext'])
                        if np.any(ok):ext_errors.append(float(np.max(np.abs(b['ext'][ok]-n['ext'][ok]))))
                    error=max(ext_errors,default=0.);assert error<1e-6,(scene,h['uid'],error)
                    rows.append({'scene':scene,'asset_uid':h['uid'],'canonical_axis':axis,'sign':sign,'shift_m':.02,
                        'assigned_voxels':[len(s) for s in subsets],
                        'extension_winning_mass':[float(b['mass'][b['extension_wins']].sum()) for b in base],
                        'extension_supported_mass':[float(b['mass'][b['extension_wins']&(b['full']<cfg['null_cost'])].sum()) for b in base],
                        'history_supported_mass':[float(b['mass'][~b['extension_wins']&(b['full']<cfg['null_cost'])].sum()) for b in base],
                        'assigned_mass':[float(b['mass'].sum()) for b in base],
                        'extension_cost_max_difference':error,
                        'full_energy_delta':float(sum(np.dot(b['w'],n['full']-b['full']) for b,n in zip(base,new))),
                        'maximum_full_point_cost_difference':max([float(np.max(np.abs(n['full']-b['full']))) for b,n in zip(base,new) if len(b['full'])],default=0.),
                        'anchor_position_change_m':np.linalg.norm(anchor-old_anchor,axis=1).tolist(),
                        'anchor_displacement_change_m':float(np.linalg.norm((anchor[1]-anchor[0])-(old_anchor[1]-old_anchor[0])))})
        print('GAUGE DIAGNOSED',scene,flush=True)
    save(root/'anchor_gauge_diagnosis.json',{'rows':rows,'source_sha':sha(__file__),
        'scope':'Post-seal fixed-label diagnostic, no GT used, no candidate selected, no predictions changed. Six common right-translations per eligible asset, not a new solver. Relative-pose invariance is standard gauge algebra; immutable historical surface can break it.'})


if __name__=='__main__':main()
