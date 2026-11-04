"""No-GT measured-patch feasibility and independent fractional LP controls."""
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.ray_conditioned_plane_polytope_v1 import fit_patch,intersection_bounds


def fractional_lp(A,b,num_const,num_vec,den_vec,sign):
    # Charnes-Cooper: y=c*t, t=1/(sign*den*c), t>=0.
    matrix=np.c_[A,-b];eq=np.r_[sign*den_vec,0.][None,:]
    objective=np.r_[sign*num_vec,sign*num_const]
    out=[]
    for sense in [1,-1]:
        ans=linprog(sense*objective,A_ub=matrix,b_ub=np.zeros(len(b)),A_eq=eq,b_eq=[1.],bounds=[(None,None)]*3+[(0,None)],method='highs')
        assert ans.status==0,ans.message
        out.append(float(sense*ans.fun))
    return out


def main():
    root=Path('results/first_return_material_surface_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json')
    for path,h in seal['sha256'].items():assert sha(root/path)==h
    import pickle
    records=[];controls=[];examined=0
    for run in read(root/'runs.json'):
        dest=root/run['scene']
        scene_controls=0
        with (dest/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(len(slots))%4
        for path in sorted(dest.glob('frame_*.npz')):
            with np.load(path) as z:frame={k:z[k] for k in ['rgb','depth','C']}
            lab,_=labels(frame,slots,groups,cfg)
            count=valid=0;vertices=[]
            # Deterministic stride and all same-slot 3x3 neighborhoods; no GT.
            for y in range(1,cfg['height']-1,3):
                for x in range(1,cfg['width']-1,3):
                    slot=lab[y,x]
                    if slot<0 or not np.all(lab[y-1:y+2,x-1:x+2]==slot):continue
                    count+=1;examined+=1
                    patch=fit_patch(frame['depth'][y-1:y+2,x-1:x+2],[x,y],cfg['focal_px'],[(cfg['width']-1)/2,(cfg['height']-1)/2],cfg['depth_noise_bound_m'])
                    if patch is None:continue
                    valid+=1;vertices.append(len(patch['coefficients']))
                    if scene_controls<10 and (examined%13==0):
                        principal=patch['principal'];r=np.array([[(x-principal[0])/cfg['focal_px'],(y-principal[1])/cfg['focal_px'],1.]])
                        # Nonzero origin creates a genuinely fractional objective.
                        o=np.array([.01,-.02,.03]);bounds=intersection_bounds(patch,o,r)
                        v=patch['coefficients'];normals=patch['normals']
                        # Normals = coeff @ L.T.
                        L=np.array([[cfg['focal_px'],0,0],[0,cfg['focal_px'],0],[-(x-principal[0]),-(y-principal[1]),1.]])
                        den=L.T@r[0];num=-L.T@o
                        if not bounds['denominator_sign_stable'][0]:continue
                        sign=1 if np.min(normals@r[0])>0 else -1
                        lp=fractional_lp(patch['A'],patch['b'],1.,num,den,sign)
                        computed=[float(bounds['lower'][0]),float(bounds['upper'][0])]
                        error=float(np.max(np.abs(np.array(lp)-computed)));assert error<1e-5,(lp,computed)
                        # Camera-origin ray lies exactly at the patch center.
                        center=intersection_bounds(patch,np.zeros(3),r)
                        assert center['valid'][0]
                        controls.append({'scene':run['scene'],'frame':path.name,'pixel':[x,y],'vertices':len(v),'vertex_bounds':computed,'LP_bounds':lp,'max_error_m':error})
                        scene_controls+=1
            records.append({'scene':run['scene'],'frame':path.name,'eligible_neighborhoods':count,'feasible_planar_patches':valid,
                            'median_polytope_vertices':float(np.median(vertices)) if vertices else None})
        print('PLANE_CHECK',run['scene'],flush=True)
    result={'no_evaluation_truth_read':True,'updates_predictions':False,'status':'PASS','examined':examined,
            'feasible_patches':sum(r['feasible_planar_patches'] for r in records),'LP_controls':len(controls),
            'max_LP_vertex_error_m':max(r['max_error_m'] for r in controls),'records':records,'controls':controls,
            'sources':{p:sha(p) for p in ['inspection_decisions/ray_conditioned_plane_polytope_v1.py',__file__]}}
    save(root/'plane_polytope_prototype.json',result);print({k:v for k,v in result.items() if k not in ['records','controls','sources']})


if __name__=='__main__':main()
