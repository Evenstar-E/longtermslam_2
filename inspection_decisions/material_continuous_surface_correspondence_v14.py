"""All measured centers, continuous target coordinates, shared material tracks.

No true IDs, material coordinates or reference transforms enter inference.
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree
from inspection_decisions.textured_material_matching_v1 import apply,rigid,world_points
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.material_partial_track_transport_v7 import transformed_jacobian
from inspection_decisions.material_correspondence_mode_graph_v6 import retract

PAIRS=[(0,1),(0,2),(1,2)]
_RASTER=OrderedDict()


def raster_planes(frame,cfg):
    key=(id(frame),cfg['surface_edge_limit_m'])
    if key in _RASTER:_RASTER.move_to_end(key);return _RASTER[key]
    d=frame['depth'];valid=frame['depth_valid'];owner=frame['anonymous_slot'];rgb=frame['rgb']/255.;Y,X=np.indices(d.shape);K=frame['K'];xyz=np.stack([(X-K[0,2])*d/K[0,0],(Y-K[1,2])*d/K[1,1],d],-1);inv=1/np.maximum(d,1e-8);out=[]
    for corners in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        pp=[];zz=[];cc=[];vv=[];oo=[]
        for y,x in corners:
            sl=(slice(y,d.shape[0]-1+y),slice(x,d.shape[1]-1+x));pp.append(xyz[sl]);zz.append(inv[sl]);cc.append(rgb[sl]);vv.append(valid[sl]);oo.append(owner[sl])
        edge=np.maximum.reduce([np.linalg.norm(pp[0]-pp[1],axis=-1),np.linalg.norm(pp[0]-pp[2],axis=-1),np.linalg.norm(pp[1]-pp[2],axis=-1)])
        ok=vv[0]&vv[1]&vv[2]&(oo[0]==oo[1])&(oo[0]==oo[2])&(edge<=cfg['surface_edge_limit_m'])
        if len(out)==0:q=np.stack([zz[1]-zz[0],zz[2]-zz[0],zz[0]],-1);color=np.stack([cc[1]-cc[0],cc[2]-cc[0],cc[0]],-1)
        else:q=np.stack([zz[0]-zz[1],zz[0]-zz[2],zz[1]+zz[2]-zz[0]],-1);color=np.stack([cc[0]-cc[1],cc[0]-cc[2],cc[1]+cc[2]-cc[0]],-1)
        out.append(dict(valid=ok,owner=oo[0],inverse=q,color=color))
    _RASTER[key]=out
    while len(_RASTER)>8:_RASTER.popitem(last=False)
    return out


def project_uv(world,frame):
    p=(world-frame['C'][:3,3])@frame['C'][:3,:3];z=p[...,2];uv=p[...,:2]/np.maximum(z[...,None],1e-8)*[frame['K'][0,0],frame['K'][1,1]]+frame['K'][:2,2]
    return uv,z


def measured_field(uv,frame,slot,cfg):
    """Actual3-pixel triangle, perspective-correct depth and real RGB.

An unresolved triangle falls back to ONE actual nearest native pixel, never
deletes a source center. This fallback is discrete, not a smooth surface claim.
"""
    shape=uv.shape[:-1];q=uv.reshape(-1,2);H,W=frame['depth'].shape;u,v=q.T
    inside=(u>=0)&(v>=0)&(u<W-1)&(v<H-1)
    x=np.clip(np.floor(u).astype(int),0,W-2);y=np.clip(np.floor(v).astype(int),0,H-2);du=u-x;dv=v-y;low=du+dv<=1
    plane=raster_planes(frame,cfg);coeff=np.where(low[:,None],plane[0]['inverse'][y,x],plane[1]['inverse'][y,x]);colors=np.where(low[:,None,None],plane[0]['color'][y,x],plane[1]['color'][y,x]);rho=coeff[:,0]*du+coeff[:,1]*dv+coeff[:,2];D=1/np.maximum(rho,1e-8);triangle=inside&np.where(low,plane[0]['valid'][y,x],plane[1]['valid'][y,x])&(rho>0);owner3=np.where(low,plane[0]['owner'][y,x],plane[1]['owner'][y,x])==slot
    K=frame['K'];C=frame['C'];rgb=colors[:,:,0]*du[:,None]+colors[:,:,1]*dv[:,None]+colors[:,:,2];rgbJ=colors[:,:,:2];depthJ=-D[:,None]**2*coeff[:,:2]
    ray=np.c_[(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(len(q))];pc=ray*D[:,None];pj=ray[:,:,None]*depthJ[:,None,:];pj[:,0,0]+=D/K[0,0];pj[:,1,1]+=D/K[1,1]
    xyz=pc@C[:3,:3].T+C[:3,3];xyzJ=np.einsum('ij,njk->nik',C[:3,:3],pj)
    xx=np.clip(np.rint(u).astype(int),0,W-1);yy=np.clip(np.rint(v).astype(int),0,H-1);native_inside=(u>=-.5)&(v>=-.5)&(u<W-.5)&(v<H-.5);native=native_inside&frame['depth_valid'][yy,xx];native_owner=frame['anonymous_slot'][yy,xx]==slot
    atomic=~triangle;dn=frame['depth'][yy,xx];pn=np.c_[(xx-K[0,2])*dn/K[0,0],(yy-K[1,2])*dn/K[1,1],dn];xyz[atomic]=pn[atomic]@C[:3,:3].T+C[:3,3];rgb[atomic]=frame['rgb'][yy[atomic],xx[atomic]]/255.;D[atomic]=dn[atomic];xyzJ[atomic]=0;rgbJ[atomic]=0
    valid=np.where(triangle,True,native);surface=np.where(triangle,owner3,native&native_owner)
    return dict(xyz=xyz.reshape(*shape,3),rgb=rgb.reshape(*shape,3),xyzJ=xyzJ.reshape(*shape,3,2),rgbJ=rgbJ.reshape(*shape,3,2),depth=D.reshape(shape),valid=valid.reshape(shape),surface=surface.reshape(shape),triangle=triangle.reshape(shape))


def photo_kernel(residual,cfg):
    s0=cfg['source_color_sigma'];st=cfg['target_color_sigma'];v=s0*s0+st*st
    return (st*st/v)**1.5*np.exp(-.5*np.sum(residual**2,axis=-1)/v)


def joint_photo(r1,r2,cfg,shared):
    a=cfg['source_color_sigma']**2;b=cfg['target_color_sigma']**2;v=a+b;c=a if shared else 0.;det=v*v-c*c
    quadratic=(v*(r1*r1+r2*r2).sum(-1)-2*c*(r1*r2).sum(-1))/det
    return (b*b/det)**1.5*np.exp(-.5*quadratic)


def residual_energy(field,p,color,offset,cfg):
    d=field['xyz']-p;r=field['rgb']-color;v=cfg['source_color_sigma']**2+cfg['target_color_sigma']**2
    e=np.sum(d*d,-1)/cfg['continuous_sigma_m']**2+np.sum(r*r,-1)/v+np.sum(offset*offset,-1)/cfg['offset_sigma_px']**2
    return np.where(field['surface'],e,np.inf)


def solve_offset(p,color,frame,slot,cfg,optimize):
    uv,z=project_uv(p,frame);zero=np.zeros_like(uv);base=measured_field(uv,frame,slot,cfg);offset=zero.copy();field={k:v.copy() for k,v in base.items()}
    if optimize:
        # Do not require another epoch to have a sample at the same center.
        # A small cross also reaches native branches near unresolved image edges.
        best=residual_energy(field,p,color,offset,cfg)
        for step in [[-1.,0],[1.,0],[0,-1.],[0,1.]]:
            d=np.broadcast_to(step,uv.shape);r=measured_field(uv+d,frame,slot,cfg);e=residual_energy(r,p,color,d,cfg);take=e<best
            if take.any():
                offset[take]=d[take];best[take]=e[take]
                for k in field:field[k][take]=r[k][take]
        for _ in range(cfg['offset_steps']):
            A=field['xyzJ']/cfg['continuous_sigma_m'];r=(field['xyz']-p)/cfg['continuous_sigma_m'];B=field['rgbJ']/np.sqrt(cfg['source_color_sigma']**2+cfg['target_color_sigma']**2);c=(field['rgb']-color)/np.sqrt(cfg['source_color_sigma']**2+cfg['target_color_sigma']**2)
            H=np.einsum('...ci,...cj->...ij',A,A)+np.einsum('...ci,...cj->...ij',B,B)+np.eye(2)/cfg['offset_sigma_px']**2
            g=np.einsum('...ci,...c->...i',A,r)+np.einsum('...ci,...c->...i',B,c)+offset/cfg['offset_sigma_px']**2
            d=-np.linalg.solve(H,g[...,None])[...,0];d*=np.minimum(1.,1./np.maximum(np.linalg.norm(d,axis=-1),1e-12))[...,None];new=offset+d;new*=np.minimum(1.,cfg['offset_limit_px']/np.maximum(np.linalg.norm(new,axis=-1),1e-12))[...,None]
            q=measured_field(uv+new,frame,slot,cfg);e=residual_energy(q,p,color,new,cfg);take=e<best-1e-12
            if not take.any():break
            offset[take]=new[take];best[take]=e[take]
            for k in field:field[k][take]=q[k][take]
    # Free-space evidence stays on the original geometric ray; delta cannot
    # move an inconvenient ray to erase an actual exposure contradiction.
    gap=base['depth']-z;occluded=base['valid']&(gap < -cfg['ray_tolerance_m']);dist2=np.sum((field['xyz']-p)**2,-1);valid=field['surface']&(z>0)&~occluded&(dist2<=cfg['continuous_capture_m']**2)
    geometry=np.exp(-.5*dist2/cfg['continuous_sigma_m']**2-.5*np.sum(offset**2,-1)/cfg['offset_sigma_px']**2)*valid;kernel=photo_kernel(field['rgb']-color,cfg)*geometry
    free=np.where(base['valid']&(gap>cfg['ray_tolerance_m']),gap-cfg['ray_tolerance_m'],np.inf)
    return dict(field=field,uv=uv+offset,offset=offset,valid=valid,geometry=geometry,kernel=kernel,free=free,original_gap=gap)


def directed_match(p,color,frames,slot,cfg,optimize):
    shape=p.shape[:-1];best=np.zeros(shape);geometry=np.zeros(shape);xyz=np.zeros_like(p);rgb=np.zeros_like(p);uv=np.zeros((*shape,2));offset=np.zeros_like(uv);xyzJ=np.zeros((*shape,3,2));rgbJ=np.zeros_like(xyzJ);valid=np.zeros(shape,bool);free=np.full(shape,np.inf);fi=np.full(shape,-1,int)
    for h,f in enumerate(frames):
        r=solve_offset(p,color,f,slot,cfg,optimize);take=r['valid']&((r['kernel']>best)|~valid);best[take]=r['kernel'][take];geometry[take]=r['geometry'][take];xyz[take]=r['field']['xyz'][take];rgb[take]=r['field']['rgb'][take];uv[take]=r['uv'][take];offset[take]=r['offset'][take];xyzJ[take]=r['field']['xyzJ'][take];rgbJ[take]=r['field']['rgbJ'][take];fi[take]=h;valid|=r['valid'];free=np.minimum(free,r['free'])
    free=np.where(np.isfinite(free)&~valid,free,0.)
    return dict(kernel=best,geometry=geometry,xyz=xyz,rgb=rgb,uv=uv,offset=offset,xyzJ=xyzJ,rgbJ=rgbJ,valid=valid,free=free,frame=fi)


class DensePair:
    def __init__(self,a,b,fa,fb,ia,ib,dense_a,dense_b,cfg):
        self.a=a;self.b=b;self.fa=fa;self.fb=fb;self.ia=ia;self.ib=ib;self.da=dense_a;self.db=dense_b;self.ta=cKDTree(dense_a['xyz']);self.tb=cKDTree(dense_b['xyz']);self.cfg=cfg
    def nearest(self,T):
        a,b=self.a,self.b;cfg=self.cfg;x=apply(T,a['xyz']);y=apply(inverse(T),b['xyz']);d,j=self.tb.query(x,k=4);e,i=self.ta.query(y,k=4)
        ca=np.mean((a['rgb'][:,None]-self.db['rgb'][j])**2,axis=-1);cb=np.mean((b['rgb'][:,None]-self.da['rgb'][i])**2,axis=-1);v=cfg['source_color_sigma']**2+cfg['target_color_sigma']**2
        va=np.minimum(d/cfg['continuous_capture_m'],1)**2+np.minimum(ca/v,1);vb=np.minimum(e/cfg['continuous_capture_m'],1)**2+np.minimum(cb/v,1)
        geo=.5*(np.minimum(d[:,0]/cfg['continuous_capture_m'],1).mean()+np.minimum(e[:,0]/cfg['continuous_capture_m'],1).mean());fusion=.5*(va.min(1).mean()+vb.min(1).mean())
        return geo,fusion,dict(forward=(x,self.db['xyz'][j[np.arange(len(j)),va.argmin(1)]]),backward=(y,self.da['xyz'][i[np.arange(len(i)),vb.argmin(1)]]))
    def projected(self,T,optimize):
        a,b=self.a,self.b;forward=directed_match(apply(T,a['xyz']),a['rgb'],self.fb,self.ib,self.cfg,optimize);backward=directed_match(apply(inverse(T),b['xyz']),b['rgb'],self.fa,self.ia,self.cfg,optimize)
        value=1-.5*(forward['kernel'].mean()+backward['kernel'].mean())+self.cfg['free_weight']*.5*(np.log1p((forward['free']/self.cfg['continuous_sigma_m'])**2).mean()+np.log1p((backward['free']/self.cfg['continuous_sigma_m'])**2).mean())
        return value,(forward,backward)


def joint_offsets(r1,r2,p1,p2,color,frames1,frames2,slot1,slot2,T12,cfg,shared,track):
    """Four image-coordinate variables, shared covariance + geometric track.

No loss can be reduced by dropping a currently observed track. Accepted
updates retain both actual surface observations and decrease the exact energy.
"""
    a=cfg['source_color_sigma']**2;b=cfg['target_color_sigma']**2;v=a+b;c=a if shared else 0.;W=np.array([[v,-c],[-c,v]])/(v*v-c*c)
    old1={k:val.copy() for k,val in r1.items()};old2={k:val.copy() for k,val in r2.items()};both=r1['valid']&r2['valid'];N=len(both)
    def energy(A,B):
        d1=(A['xyz']-p1)/cfg['continuous_sigma_m'];d2=(B['xyz']-p2)/cfg['continuous_sigma_m'];c1=A['rgb']-color;c2=B['rgb']-color;value=np.sum(d1*d1+d2*d2,1)+W[0,0]*np.sum(c1*c1+c2*c2,1)+2*W[0,1]*np.sum(c1*c2,1)+np.sum(A['offset']**2+B['offset']**2,1)/cfg['offset_sigma_px']**2
        if track:value+=np.sum((apply(T12,A['xyz'])-B['xyz'])**2,1)/cfg['continuous_sigma_m']**2
        return value
    old=energy(r1,r2);Jg=np.zeros((N,6,4));Jg[:,:3,:2]=r1['xyzJ']/cfg['continuous_sigma_m'];Jg[:,3:,2:]=r2['xyzJ']/cfg['continuous_sigma_m'];rg=np.c_[r1['xyz']-p1,r2['xyz']-p2]/cfg['continuous_sigma_m']
    H=np.einsum('nci,ncj->nij',Jg,Jg)+np.eye(4)/cfg['offset_sigma_px']**2;g=np.einsum('nci,nc->ni',Jg,rg)+np.c_[r1['offset'],r2['offset']]/cfg['offset_sigma_px']**2
    C=np.zeros((N,2,3,4));C[:,0,:,:2]=r1['rgbJ'];C[:,1,:,2:]=r2['rgbJ'];res=np.stack([r1['rgb']-color,r2['rgb']-color],1);H+=np.einsum('neci,ef,nfcj->nij',C,W,C);g+=np.einsum('neci,ef,nfc->ni',C,W,res)
    if track:
        J=np.concatenate([np.einsum('ij,njk->nik',T12[:3,:3],r1['xyzJ']),-r2['xyzJ']],2)/cfg['continuous_sigma_m'];res=(apply(T12,r1['xyz'])-r2['xyz'])/cfg['continuous_sigma_m'];H+=np.einsum('nci,ncj->nij',J,J);g+=np.einsum('nci,nc->ni',J,res)
    d=-np.linalg.solve(H,g[...,None])[...,0];d*=np.minimum(1.,1./np.maximum(np.linalg.norm(d,axis=1),1e-12))[:,None]
    candidates=[]
    for r,frames,slot,part,predicted in [(r1,frames1,slot1,d[:,:2],p1),(r2,frames2,slot2,d[:,2:],p2)]:
        out={k:val.copy() for k,val in r.items()};new=r['offset']+part;new*=np.minimum(1.,cfg['offset_limit_px']/np.maximum(np.linalg.norm(new,axis=1),1e-12))[:,None];q_uv=r['uv']+new-r['offset'];ok=r['valid'].copy()
        for fi,f in enumerate(frames):
            ix=np.flatnonzero(both&(r['frame']==fi))
            if not len(ix):continue
            q=measured_field(q_uv[ix],f,slot,cfg);out['xyz'][ix]=q['xyz'];out['rgb'][ix]=q['rgb'];out['xyzJ'][ix]=q['xyzJ'];out['rgbJ'][ix]=q['rgbJ'];out['uv'][ix]=q_uv[ix];out['offset'][ix]=new[ix];ok[ix]&=q['surface']&(np.sum((q['xyz']-predicted[ix])**2,1)<=cfg['continuous_capture_m']**2)
        candidates.append((out,ok))
    A,oa=candidates[0];B,ob=candidates[1];take=both&oa&ob&(energy(A,B)<old-1e-12)
    for out,new in [(old1,A),(old2,B)]:
        for key in ['xyz','rgb','xyzJ','rgbJ','uv','offset']:out[key][take]=new[key][take]
    return old1,old2,int(take.sum())


class ContinuousTriple:
    def __init__(self,pairs,cfg):self.pairs=pairs;self.cfg=cfg
    def evaluate(self,T,return_state=False):
        U=[T[0],T[1],T[1]@inverse(T[0])];nearest=[p.nearest(u) for p,u in zip(self.pairs,U)];fixed=[p.projected(u,False) for p,u in zip(self.pairs,U)];shift=[p.projected(u,True) for p,u in zip(self.pairs,U)];weights=[1.,1.,.5]
        values=[sum(w*r[col] for w,r in zip(weights,nearest))/2.5 for col in [0,1]]+[sum(w*r[0] for w,r in zip(weights,mode))/2.5 for mode in [fixed,shift]]
        h1,h2=shift[0][1][0],shift[1][1][0];a=self.pairs[0].a;groupstates=[]
        for shared,track in [(False,False),(False,True),(True,True)]:
            A,B,count=joint_offsets(h1,h2,apply(T[0],a['xyz']),apply(T[1],a['xyz']),a['rgb'],self.pairs[0].fb,self.pairs[1].fb,self.pairs[0].ib,self.pairs[1].ib,U[2],self.cfg,shared,track)
            both=A['valid']&B['valid'];g1=np.exp(-.5*np.sum((A['xyz']-apply(T[0],a['xyz']))**2,1)/self.cfg['continuous_sigma_m']**2-.5*np.sum(A['offset']**2,1)/self.cfg['offset_sigma_px']**2);g2=np.exp(-.5*np.sum((B['xyz']-apply(T[1],a['xyz']))**2,1)/self.cfg['continuous_sigma_m']**2-.5*np.sum(B['offset']**2,1)/self.cfg['offset_sigma_px']**2)
            kernel=joint_photo(A['rgb']-a['rgb'],B['rgb']-a['rgb'],self.cfg,shared)*g1*g2
            if track:kernel*=np.exp(-.5*np.sum((apply(U[2],A['xyz'])-B['xyz'])**2,1)/self.cfg['continuous_sigma_m']**2)
            singles=h1['kernel']+h2['kernel'];joint=np.where(both,2*np.sqrt(np.maximum(kernel,0)),singles)
            # ONLY replace the two forward H-source contributions; all reverse
            # evidence and actual exposure penalties stay in the common base.
            value=values[3]+(singles-joint).mean()/5.;values.append(value);groupstates.append((A,B,count))
        state=dict(nearest=nearest,fixed=fixed,shift=shift,groups=groupstates)
        return (np.array(values),state) if return_state else np.array(values)
    def score(self,poses):return np.array([self.evaluate(t) for t in poses])


def fit_pose(scorer,T,cfg,column):
    T=np.array(T);values,state=scorer.evaluate(T,True);value=float(values[column]);path=[value]
    for _ in range(cfg['pose_steps']):
        G=np.r_[np.eye(4)[None],T];AA=[];rr=[]
        for pe,((h,f),p,w) in enumerate(zip(PAIRS,scorer.pairs,[1.,1.,.5])):
            info=state['fixed' if column==2 else 'shift'][pe][1]
            for direction,(e,z,bank,r) in enumerate([(h,f,p.a,info[0]),(f,h,p.b,info[1])]):
                if pe<2 and direction==0 and column>=4:r=state['groups'][column-4][pe]
                good=r['valid']&(r['kernel']>1e-6)
                if not good.any():continue
                x=bank['xyz'][good];y,J=transformed_jacobian(G,e,z,x);res=y-r['xyz'][good];weight=np.sqrt(w*r['kernel'][good])/cfg['continuous_sigma_m'];AA.append((J*weight[:,None,None]).reshape(-1,12));rr.append((res*weight[:,None]).ravel())
        if not AA:break
        A=np.concatenate(AA);b=np.concatenate(rr);d=np.linalg.lstsq(np.r_[A,.01*np.eye(12)],np.r_[-b,np.zeros(12)],rcond=None)[0];d*=min(1,.04/max(np.linalg.norm(d),1e-12));accepted=False
        for alpha in [1,.5,.25]:
            U=retract(T,alpha*d);vv,ss=scorer.evaluate(U,True)
            if vv[column]<value-1e-12:T=U;state=ss;value=float(vv[column]);accepted=True;break
        path.append(value)
        if not accepted:break
    return T,np.array(path)


def spectral_proposals(a,b,cfg):
    """Ordinary spatial-consistency spectral proposals, NOT authorSC2 benchmark."""
    A=a['fpfh'].T.copy();B=b['fpfh'].T.copy();A/=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-8);B/=np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-8)
    D=np.maximum(0,1-A@B.T)+.5*np.minimum(np.mean((a['rgb'][:,None]-b['rgb'][None])**2,2)/.04,1);j=D.argmin(1);rev=D.argmin(0);edges=np.unique(np.r_[np.c_[np.arange(len(j)),j],np.c_[rev,np.arange(len(rev))]],axis=0);x=a['xyz'][edges[:,0]];y=b['xyz'][edges[:,1]]
    dx=np.linalg.norm(x[:,None]-x[None],axis=-1);dy=np.linalg.norm(y[:,None]-y[None],axis=-1);compatible=(abs(dx-dy)<cfg['spectral_distance_m'])&(edges[:,0,None]!=edges[None,:,0])&(edges[:,1,None]!=edges[None,:,1]);C=compatible.astype(np.float32);C2=(C@C)*C;v=np.ones(len(C),np.float32)
    for _ in range(6):v=C2@v;v/=max(np.linalg.norm(v),1e-8)
    order=np.argsort(-v);seeds=[]
    for h in order:
        if not seeds or np.min(np.linalg.norm(x[seeds]-x[h],axis=1))>.04:seeds.append(int(h))
        if len(seeds)>=cfg['spectral_seeds']:break
    poses=[]
    for h in seeds:
        group=np.argsort(-C2[h])[:cfg['spectral_consensus']];weight=np.maximum(C2[h,group],0).astype(float);T=rigid(x[group],y[group],weight)
        if T is not None:poses.append(T)
    return np.array(poses).reshape(-1,4,4),dict(edges=len(edges),seeds=len(seeds),models=len(poses))
