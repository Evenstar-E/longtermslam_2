"""Standard oriented-normal compatibility on actual RGBD material patches."""
import numpy as np
from inspection_decisions.material_surface_patch_registration_v9 import inverse

def orient_banks(banks,frames,cfg):
    for e,row in enumerate(banks):
        for a in row:
            x=a['patch_xyz'];normal=np.cross(x[:,2]-x[:,1],x[:,4]-x[:,3]);length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length[:,None],1e-12);camera=np.array([frames[e][int(fi)]['C'][:3,3] for fi in a['patch_frame']]);normal[np.sum(normal*(camera-x[:,0]),axis=1)<0]*=-1;a['patch_normal']=normal;a['patch_normal_valid']=length>cfg['normal_cross_min']

def project(world,frame,slot):
    """V9 bilinear surface/color, plus exact depth-field normal without color J."""
    shape=world.shape[:-1];x=world.reshape(-1,3);C,K=frame['C'],frame['K'];p=(x-C[:3,3])@C[:3,:3];z=p[:,2];zz=np.maximum(z,1e-8);u=p[:,0]/zz*K[0,0]+K[0,2];v=p[:,1]/zz*K[1,1]+K[1,2];H,W=frame['depth'].shape;inside=(z>0)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1);xx=np.clip(np.floor(u).astype(int),0,W-2);yy=np.clip(np.floor(v).astype(int),0,H-2);du=u-xx;dv=v-yy
    weight=np.stack([(1-du)*(1-dv),du*(1-dv),(1-du)*dv,du*dv],1);X=np.stack([xx,xx+1,xx,xx+1],1);Y=np.stack([yy,yy,yy+1,yy+1],1);depth=frame['depth'][Y,X];valid=inside&frame['depth_valid'][Y,X].all(1)&(np.ptp(depth,axis=1)<.025);surface=valid&(frame['anonymous_slot'][Y,X]==slot).all(1);D=(weight*depth).sum(1);rgb=(weight[:,:,None]*(frame['rgb'][Y,X]/255.)).sum(1)
    dx=(1-dv)*(depth[:,1]-depth[:,0])+dv*(depth[:,3]-depth[:,2]);dy=(1-du)*(depth[:,2]-depth[:,0])+du*(depth[:,3]-depth[:,1]);normal=np.c_[-dx*K[0,0]/zz,-dy*K[1,1]/zz,1+(dx*K[0,0]*p[:,0]+dy*K[1,1]*p[:,1])/zz**2];normal=-normal@C[:3,:3].T;length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length[:,None],1e-12)
    return dict(surface=surface.reshape(shape),valid=valid.reshape(shape),gap=(D-z).reshape(shape),rgb=rgb.reshape(*shape,3),normal=normal.reshape(*shape,3),normal_valid=(length>1e-12).reshape(shape))

def supports(color,target,hit,gap,cfg):
    geometry=np.exp(-.5*(gap/cfg['ray_tolerance_m'])**2)*hit;distance=np.mean((target-color)**2,axis=-1);ind=(geometry*np.exp(-.5*distance/cfg['color_scale']**2)).mean(-1);count=hit.sum(-1);den=np.maximum(count,1)[...,None,None];ma=(color*hit[...,None]).sum(-2,keepdims=True)/den;mb=(target*hit[...,None]).sum(-2,keepdims=True)/den;ac=(color-ma)*hit[...,None];bc=(target-mb)*hit[...,None];sa=(ac*ac).sum((-2,-1));sb=(bc*bc).sum((-2,-1));dot=(ac*bc).sum((-2,-1));corr=np.maximum(0.,dot/np.sqrt(np.maximum(sa*sb,1e-12)));flat=(sa<1e-5)&(sb<1e-5);corr=np.where(flat,np.exp(-.5*((ma-mb)**2).mean((-2,-1))/cfg['color_scale']**2),corr);patch=geometry.mean(-1)*corr*(count>=3);return np.stack([ind,patch],-1)

def terms(world,color,normal,normal_valid,frames,slot,cfg):
    shape=world.shape[:-2];best=np.zeros((*shape,4));minimum=np.full(world.shape[:-1],np.inf);any_hit=np.zeros(world.shape[:-1],bool)
    for f in frames:
        r=project(world,f,slot);raw=r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m']);any_hit|=raw;free=r['valid']&(r['gap']>cfg['ray_tolerance_m']);minimum=np.minimum(minimum,np.where(free,r['gap']-cfg['ray_tolerance_m'],np.inf));cosine=np.sum(normal[...,None,:]*r['normal'],axis=-1);compatible=(cosine>=cfg['normal_cosine_min'])|~normal_valid[...,None]|~r['normal_valid'];gated=raw&compatible
        best[...,:2]=np.maximum(best[...,:2],supports(color,r['rgb'],raw,r['gap'],cfg));best[...,2:]=np.maximum(best[...,2:],supports(color,r['rgb'],gated,r['gap'],cfg))
    minimum=np.where(np.isfinite(minimum)&~any_hit,minimum,0.);F=np.log1p((minimum/cfg['geometry_scale_m'])**2).mean((-1,-2));return best.mean(-2),F

class OrientedScore:
    def __init__(self,a,b,fa,fb,ia,ib,cfg):self.a=a;self.b=b;self.fa=fa;self.fb=fb;self.ia=ia;self.ib=ib;self.cfg=cfg
    def score(self,poses):
        result=[];a,b=self.a,self.b
        if not len(a['patch_xyz']) or not len(b['patch_xyz']):return np.ones((len(poses),4))
        for start in range(0,len(poses),64):
            T=np.asarray(poses[start:start+64]);U=inverse(T);pa=np.einsum('bij,npj->bnpi',T[:,:3,:3],a['patch_xyz'])+T[:,None,None,:3,3];pb=np.einsum('bij,npj->bnpi',U[:,:3,:3],b['patch_xyz'])+U[:,None,None,:3,3];na=np.einsum('bij,nj->bni',T[:,:3,:3],a['patch_normal']);nb=np.einsum('bij,nj->bni',U[:,:3,:3],b['patch_normal']);aa=terms(pa,a['patch_rgb'][None],na,a['patch_normal_valid'][None],self.fb,self.ib,self.cfg);bb=terms(pb,b['patch_rgb'][None],nb,b['patch_normal_valid'][None],self.fa,self.ia,self.cfg);value=1-.5*(aa[0]+bb[0])+.5*self.cfg['free_weight']*(aa[1]+bb[1])[:,None];result.append(value)
        return np.concatenate(result)
