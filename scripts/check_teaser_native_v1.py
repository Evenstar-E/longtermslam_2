"""Known-transform adapter check, not a new real-data performance sample."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.real_temporal_v1 import teaser,moved

root=Path('results/real_temporal_v1/native_wrapper_check');root.mkdir(exist_ok=False)
rng=np.random.default_rng(20261023);a=rng.normal(size=(128,3))
truth=np.eye(4);truth[:3,:3]=Rotation.from_rotvec([.12,-.08,.21]).as_matrix();truth[:3,3]=[.2,-.1,.4]
b=moved(a,truth)+rng.normal(0,.002,size=a.shape);b[80:]=rng.normal(size=(48,3))*3
T,stats=teaser(a,b,{'teaser_noise_bound_m':.02},root)
assert T is not None
translation=float(np.linalg.norm(T[:3,3]-truth[:3,3]));rotation=float(np.degrees(Rotation.from_matrix(T[:3,:3]@truth[:3,:3].T).magnitude()))
assert translation<.02 and rotation<1.
(root/'check.json').write_text(json.dumps({'passed':True,'translation_error_m':translation,'rotation_error_deg':rotation,'inliers_generated':80,'outliers_generated':48,'scope':'Synthetic unit check of wrapper direction/units, not a real experimental sample','stats':stats},indent=2),encoding='utf-8')
print('TEASER WRAPPER PASS',translation,rotation)
