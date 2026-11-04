"""Slant-invariant measured-plane support, no hidden surfaces or GT input."""
import numpy as np
from inspection_decisions.textured_material_matching_v1 import world_points

SOURCE_OFFSETS=np.array([[0,0],[-2,0],[2,0],[0,-2],[0,2]])
SOURCE_DESIGN=np.c_[SOURCE_OFFSETS,np.ones(5)]
SOURCE_PINV=np.linalg.pinv(SOURCE_DESIGN)
TARGET_OFFSETS=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
TARGET_DESIGN=np.c_[TARGET_OFFSETS,np.ones(9)]
TARGET_PINV=np.linalg.pinv(TARGET_DESIGN)

def resample_banks(banks,frames,cfg):
    """Only patch eligibility changes; all512 measured center pixels are retained."""
    for e,row in enumerate(banks):
        for slot,a in enumerate(row):
            eligible=[]
            for h,(p,fi) in enumerate(zip(a['uv'],a['frame'])):
                f=frames[e][int(fi)];pix=p+SOURCE_OFFSETS;H,W=f['depth'].shape
                if (pix<0).any() or (pix[:,0]>=W).any() or (pix[:,1]>=H).any():continue
                x,y=pix.T
                if not (f['depth_valid'][y,x]&(f['anonymous_slot'][y,x]==slot)).all():continue
                z=f['depth'][y,x];coeff=SOURCE_PINV@(1/z);fit=SOURCE_DESIGN@coeff
                if (fit<=0).any() or np.max(abs(1/fit-z))>cfg['plane_residual_m']:continue
                eligible.append(h)
            eligible=np.array(eligible,int)
            if len(eligible)>cfg['patches']:eligible=eligible[np.linspace(0,len(eligible)-1,cfg['patches'],dtype=int)]
            xyz=[];rgb=[];uv=[];fi=[]
            for h in eligible:
                p=a['uv'][h]+SOURCE_OFFSETS;f=frames[e][int(a['frame'][h])];xyz.append(world_points(f,p));rgb.append(f['rgb'][p[:,1],p[:,0]]/255.);uv.append(p);fi.append(a['frame'][h])
            a.update(patch_xyz=np.array(xyz).reshape(-1,5,3),patch_rgb=np.array(rgb).reshape(-1,5,3),patch_uv=np.array(uv).reshape(-1,5,2),patch_frame=np.array(fi,int),patch_center_index=eligible)

def project(world,frame,slot):
    """Fit nine real inverse depths; evaluate an actual plane at subpixel uv."""
    shape=world.shape[:-1];x=world.reshape(-1,3);C,K=frame['C'],frame['K'];p=(x-C[:3,3])@C[:3,:3];z=p[:,2];zz=np.maximum(z,1e-8);u=p[:,0]/zz*K[0,0]+K[0,2];v=p[:,1]/zz*K[1,1]+K[1,2];H,W=frame['depth'].shape;cx=np.clip(np.rint(u).astype(int),1,W-2);cy=np.clip(np.rint(v).astype(int),1,H-2);X=cx[:,None]+TARGET_OFFSETS[:,0];Y=cy[:,None]+TARGET_OFFSETS[:,1];depth=frame['depth'][Y,X];inv=1/np.maximum(depth,1e-8);coeff=inv@TARGET_PINV.T;fitted=coeff@TARGET_DESIGN.T;residual=np.max(abs(1/np.maximum(fitted,1e-8)-depth),axis=1);q=coeff[:,0]*(u-cx)+coeff[:,1]*(v-cy)+coeff[:,2];valid=(z>0)&(u>=1)&(u<W-2)&(v>=1)&(v<H-2)&frame['depth_valid'][Y,X].all(1)&(fitted>0).all(1)&(q>0)&(residual<=.002);D=1/np.maximum(q,1e-8);surface=valid&(frame['anonymous_slot'][Y,X]==slot).all(1)
    # Color still comes from the four native neighbors, not generated texture.
    xx=np.clip(np.floor(u).astype(int),0,W-2);yy=np.clip(np.floor(v).astype(int),0,H-2);du=u-xx;dv=v-yy;weight=np.stack([(1-du)*(1-dv),du*(1-dv),(1-du)*dv,du*dv],1);X4=np.stack([xx,xx+1,xx,xx+1],1);Y4=np.stack([yy,yy,yy+1,yy+1],1);rgb=(weight[:,:,None]*(frame['rgb'][Y4,X4]/255.)).sum(1);normal=np.c_[coeff[:,0]*K[0,0],coeff[:,1]*K[1,1],coeff[:,2]+coeff[:,0]*(K[0,2]-cx)+coeff[:,1]*(K[1,2]-cy)];normal=-normal@C[:3,:3].T;length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length[:,None],1e-12)
    return dict(surface=surface.reshape(shape),valid=valid.reshape(shape),gap=(D-z).reshape(shape),rgb=rgb.reshape(*shape,3),normal=normal.reshape(*shape,3),normal_valid=(length>1e-12).reshape(shape))
