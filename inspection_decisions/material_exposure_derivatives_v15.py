"""Prototype exact smooth-branch original-ray loss gradient (not V14c fit)."""
import numpy as np
from inspection_decisions.material_continuous_surface_correspondence_v14 import measured_field,project_uv


def exposure_loss_gradient(p,frame,slot,cfg):
    uv,z=project_uv(p,frame);field=measured_field(uv,frame,slot,cfg);pc=(p-frame['C'][:3,3])@frame['C'][:3,:3];K=frame['K'];R=frame['C'][:3,:3];safe=np.maximum(z,1e-8)
    J=np.zeros((*z.shape,2,3));J[...,0,0]=K[0,0]/safe;J[...,1,1]=K[1,1]/safe;J[...,0,2]=-K[0,0]*pc[...,0]/safe**2;J[...,1,2]=-K[1,1]*pc[...,1]/safe**2;J=J@R.T
    depthJ=np.einsum('i,...ij->...j',R[:,2],field['xyzJ']);gap=field['depth']-z-cfg['ray_tolerance_m'];active=field['valid']&(z>0)&(gap>0);g=np.where(active,gap,0.);sigma=cfg['continuous_sigma_m'];loss=np.log1p((g/sigma)**2)
    gapJ=np.einsum('...i,...ij->...j',depthJ,J)-R[:,2];gradient=(2*g/(sigma*sigma+g*g))[...,None]*gapJ
    return dict(loss=loss,gradient=gradient,active=active,gap=g,gapJ=gapJ)
