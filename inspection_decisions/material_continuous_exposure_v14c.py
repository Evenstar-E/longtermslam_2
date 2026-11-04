"""Correct original-ray exposure aggregation; old frozen sources stay intact."""
import numpy as np
from inspection_decisions import material_continuous_surface_correspondence_v14 as base
from inspection_decisions.material_continuous_surface_cache_v14b import (
    DensePair, ContinuousTriple, fit_pose, spectral_proposals, PAIRS)


def directed_match(p,color,frames,slot,cfg,optimize):
    shape=p.shape[:-1];best=np.zeros(shape);geometry=np.zeros(shape);xyz=np.zeros_like(p);rgb=np.zeros_like(p);uv=np.zeros((*shape,2));offset=np.zeros_like(uv);xyzJ=np.zeros((*shape,3,2));rgbJ=np.zeros_like(xyzJ);valid=np.zeros(shape,bool);penalty=np.zeros(shape);fi=np.full(shape,-1,int)
    for h,f in enumerate(frames):
        r=base.solve_offset(p,color,f,slot,cfg,optimize);take=r['valid']&((r['kernel']>best)|~valid)
        best[take]=r['kernel'][take];geometry[take]=r['geometry'][take];xyz[take]=r['field']['xyz'][take];rgb[take]=r['field']['rgb'][take];uv[take]=r['uv'][take];offset[take]=r['offset'][take];xyzJ[take]=r['field']['xyzJ'][take];rgbJ[take]=r['field']['rgbJ'][take];fi[take]=h;valid|=r['valid']
        # Every original physical ray contributes independently of whether a
        # nearby continuous correspondence or another camera finds a match.
        _,z=base.project_uv(p,f);g=np.where(np.isfinite(r['free'])&(z>0),r['free'],0.)
        penalty+=np.log1p((g/cfg['continuous_sigma_m'])**2)
    penalty/=max(len(frames),1)
    # Preserve the existing projected() API, exactly encoding the mean loss.
    free=cfg['continuous_sigma_m']*np.sqrt(np.expm1(penalty))
    return dict(kernel=best,geometry=geometry,xyz=xyz,rgb=rgb,uv=uv,offset=offset,xyzJ=xyzJ,rgbJ=rgbJ,valid=valid,free=free,frame=fi)


# Explicit runtime injection: inherited pair/track/pose routines resolve this
# global in base. Neither the original file nor its old predictions change.
base.directed_match=directed_match
