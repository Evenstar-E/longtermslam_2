"""Numerical model checks and replay of the already diagnosed concavity."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.material_observation_consistency_v1 import conditional_domain_observation,hard_labels_feasible
from inspection_decisions.ray_conditioned_plane_polytope_v1 import fit_patch


def main():
    out=Path('results/material_observation_consistency_v1');checks=0;rng=np.random.default_rng(20261024)
    patch=fit_patch(np.full((3,3),2.),[0,0],200.,[0,0],.001)
    for o,r in [(np.array([.3,0,0]),np.array([-.15,0,1.])),(np.array([.3,0,4.]),np.array([-.15,0,-1.]))]:
        got=conditional_domain_observation(patch,o,[r],[3.],.006)
        assert got['valid'][0] and got['robust_free'][0] and not got['fixed_coordinate'][0] and got['fixed_label'][0]==-1;checks+=1
        # Dense interior parameters verify conditional, moving coordinates.
        weights=rng.dirichlet(np.ones(len(patch['normals'])),100);nn=weights@patch['normals'];z=(1-nn@o)/(nn@r)
        points=o+z[:,None]*r;uv=points[:,:2]/points[:,2,None]*200
        assert np.all(uv>=got['conditional_source_uv'][:,0].min(0)-1e-10)
        assert np.all(uv<=got['conditional_source_uv'][:,0].max(0)+1e-10);checks+=2
        assert np.all(z<3.-.006);checks+=1
    got=conditional_domain_observation(patch,np.zeros(3),[[0,0,1]],[3.],.006)
    assert got['fixed_coordinate'][0] and got['fixed_label'][0]==0;checks+=1
    assert not hard_labels_feasible([[0.,0.],[0.,0.]],[1,0]);checks+=1
    assert hard_labels_feasible([[0.,0.],[0.,0.]],[1,-1]);checks+=1
    assert hard_labels_feasible([[0.,0.],[.001,0.]],[1,0]);checks+=1
    near=conditional_domain_observation(patch,np.zeros(3),[[0,0,1]],[2.],.006)
    assert near['fixed_label'][0]==-1 and not near['robust_free'][0];checks+=1
    flip={**patch,'normals':np.array([[0.,0.,1.],[0.,0.,-1.]])}
    invalid=conditional_domain_observation(flip,np.zeros(3),[[0,0,1]],[3.],.006)
    assert not invalid['valid'][0] and invalid['fixed_label'][0]==-1;checks+=1
    old=read('results/material_domain_observation_v1/mechanism_diagnosis.json');cfg=read('results/material_domain_observation_v1/config.json');holes=[]
    for item in old['concavity_domain_labels']:
        tag=item['scene'];src=Path('results/first_return_material_surface_v1')/tag;i=item['asset']
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path('results/shared_plane_visibility_mainline_v1')/tag/'history_patches.pkl').open('rb') as f:patches=pickle.load(f)
        p=patches[i][0];T=np.array(inp['branches'][i][1][item['true_pair'][1]]['T']);frame=np.load(src/'frame_2_1.npz')
        M=T@np.linalg.inv(inp['T0'][i])@p['source_camera'];relative=np.linalg.inv(M)@frame['C'];o=relative[:3,3]
        u,v=item['target_pixel'];rc=np.array([(u-(cfg['width']-1)/2)/cfg['focal_px'],(v-(cfg['height']-1)/2)/cfg['focal_px'],1.]);r=rc@relative[:3,:3].T
        got=conditional_domain_observation(p,o,[r],[frame['depth'][v,u]],cfg['depth_tolerance_m'])
        assert got['valid'][0] and got['robust_free'][0] and got['fixed_label'][0]==-1 and not got['fixed_coordinate'][0];checks+=1
        holes.append(dict(scene=tag,asset=i,source_pixel=item['source_pixel'],target_pixel=[u,v],robust_free=True,
                          fixed_mean_label='unknown',coordinate_span_px=got['coordinate_span_px'][0].tolist(),
                          retained_rule='For each compatible plane c, material is absent at q(c) if lambda(c)<d-tolerance.',
                          actual_hole_recovered=False,interpretation='Correct conditional event; no complete material-domain inference or task gain claimed.'))
    save(out/'model_checks.json',dict(checks=checks,status='PASS',concavity_replay=holes,old_GT_diagnosis_used_for_replay_only=True,predictions_changed=False,source_sha256=sha(Path(__file__))))
    print({'checks':checks,'status':'PASS','holes':holes},flush=True)


if __name__=='__main__':main()
