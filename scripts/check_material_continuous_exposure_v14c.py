"""Exercise FINAL aggregation, including a valid soft match plus exposure."""
import numpy as np
from pathlib import Path
from inspection_decisions.material_continuous_exposure_v14c import directed_match
from scripts.run_material_continuous_surface_correspondence_v14 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();K=np.array([[100.,0,16],[0,100.,16],[0,0,1.]])
    def frame(d):return dict(C=np.eye(4),K=K,depth=np.full((32,32),d),depth_valid=np.ones((32,32),bool),anonymous_slot=np.zeros((32,32),int),rgb=np.full((32,32,3),127.5))
    far=frame(2.);near=frame(1.97);p=np.array([[0.,0.,1.97]]);color=np.full((1,3),.5);checks=0
    for optimize in [False,True]:
        r=directed_match(p,color,[far],0,cfg,optimize)
        assert r['valid'][0] and r['kernel'][0]>0 and abs(r['free'][0]-.01)<1e-12;checks+=1
        both=directed_match(p,color,[far,near],0,cfg,optimize)
        target=.5*np.log1p((.01/cfg['continuous_sigma_m'])**2)
        assert both['valid'][0] and abs(np.log1p((both['free'][0]/cfg['continuous_sigma_m'])**2)-target)<1e-12;checks+=1
        behind=directed_match(np.array([[0.,0.,-1.]]),color,[far],0,cfg,optimize)
        assert not behind['valid'][0] and behind['free'][0]==0;checks+=1
    a=directed_match(p,color,[far,near],0,cfg,False);b=directed_match(p,color,[far,near],0,cfg,True)
    assert np.array_equal(a['free'],b['free']);checks+=1
    save(Path('results/material_continuous_exposure_checks_v14c.json'),dict(passed=True,checks=checks,scope='Final aggregated original-ray evidence survives a soft match and another-camera hit; no behind-camera evidence; no GT, not task efficacy.'))
    print('V14c final exposure aggregation checks',checks)


if __name__=='__main__':main()
