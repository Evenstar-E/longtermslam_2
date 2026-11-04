"""Geometric/selection controls, not tests mirroring bookkeeping."""
import itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import save,sha
from scripts.completion_catalog_scene_v1 import transform,inverse
from inspection_decisions.continuous_material_pose_mainline_v1 import select,visibility


def main():
    rng=np.random.default_rng(71503);checks=0
    # Pose acts on a HISTORICAL material coordinate, not on object centroids.
    for _ in range(60):
        T=np.eye(4);T[:3,:3]=Rotation.random(random_state=rng).as_matrix();T[:3,3]=rng.normal(size=3)
        center=rng.normal(size=3);local=T.copy();local[:3,3]-=center
        material=rng.normal(size=(5,3));assert np.allclose(transform(material,T),transform(material,local)+center);checks+=1
        assert np.allclose(inverse(transform(material,T),T),material);checks+=1
        Q=np.eye(4);Q[:3,:3]=Rotation.random(random_state=rng).as_matrix();Q[:3,3]=rng.normal(size=3)
        displacement=transform(material,Q)-transform(material,T)
        explicit=material@(Q[:3,:3]-T[:3,:3]).T+Q[:3,3]-T[:3,3]
        assert np.allclose(displacement,explicit);checks+=1
    # Exhaustive 3-asset assignment: eliminating poses given slots is exact.
    for _ in range(8):
        candidates=[[[{'slot':slot,'source':'catalogue' if code<2 else 'point_to_plane','scores':{'nn':float(rng.uniform(-1,1))}} for slot in range(3) for code in range(3)] for t in range(2)] for i in range(3)]
        for expanded in [False,True]:
            answer=select(candidates,'nn',expanded);brute=0.
            for t in [0,1]:
                best=np.inf
                for slots in itertools.permutations(range(3)):
                    value=sum(min(b['scores']['nn'] for b in candidates[i][t] if b['slot']==slot and (expanded or b['source']=='catalogue')) for i,slot in enumerate(slots));best=min(best,value)
                brute+=best
            assert abs(answer['objective']-brute)<1e-12;checks+=1
    # Physical front/match/occluded/unknown raster states have distinct costs.
    cfg={'width':5,'height':5,'focal_px':2.,'depth_tolerance_m':.006}
    f={'C':np.eye(4),'depth':np.full((5,5),2.),'labels':np.zeros((5,5),int)}
    assert visibility(np.array([[0.,0.,1.]]),[f],0,cfg)['score']==1.;checks+=1
    assert visibility(np.array([[0.,0.,2.]]),[f],0,cfg)['score']==-1.;checks+=1
    behind=visibility(np.array([[0.,0.,3.],[100.,0.,2.]]),[f],0,cfg)
    assert behind['unknown']==2 and behind['absence']==0 and behind['score']==0.;checks+=1
    assert visibility(np.array([[0.,0.,2.]]),[f],1,cfg)['score']==1.;checks+=1
    save(Path('results/continuous_material_pose_mainline_v1/model_checks.json'),{'status':'PASS','checks':checks,'source_sha256':sha(Path(__file__))})
    print({'status':'PASS','checks':checks},flush=True)


if __name__=='__main__':main()
