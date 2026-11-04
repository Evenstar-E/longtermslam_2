"""Conditional unknown-face/edge association, full two-pose/face/depth fitting.

Local normal evidence is a geometric prior derived from raw depth, not an
independent calibrated sensor likelihood. Graph topology and spans are fixed.
Physical identities and face births are NOT optimized by this module.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from inspection_decisions.material_supported_continuous_alignment_v1 import right_jacobian,skew
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions


def extract_pixels(frames,raw,slot,center,cfg):
    """Unique source/interior pixels, each with ONE optional latent depth."""
    tokens=set();radius=cfg['patch_radius_px'];pixels=[];lookup={}
    for row in raw:
        v=row['view'];x,y=row['inside_pixel'];frame=frames[v];lab=frame['labels'];h,w=lab.shape
        for yy in range(max(0,y-radius),min(h,y+radius+1)):
            for xx in range(max(0,x-radius),min(w,x+radius+1)):
                if lab[yy,xx]==slot:tokens.add((v,xx,yy))
    for v,x,y in sorted(tokens):
        f=frames[v];h,w=f['depth'].shape;C=f['C'];origin=C[:3,3]-center
        def direction(xx,yy):return np.array([(xx-(w-1)/2)/cfg['focal_px'],(yy-(h-1)/2)/cfg['focal_px'],1.])@C[:3,:3].T
        ray=direction(x,y);z=float(f['depth'][y,x]);point=origin+ray*z;normals=[];sigmas=[]
        for dx in [-radius,radius]:
            for dy in [-radius,radius]:
                xx=x+dx;yy=y+dy
                if not (0<=xx<w and 0<=yy<h and f['labels'][y,xx]==slot and f['labels'][yy,x]==slot):continue
                q=origin+direction(xx,y)*f['depth'][y,xx];r=origin+direction(x,yy)*f['depth'][yy,x];D=np.array([q-point,r-point]);_,s,V=np.linalg.svd(D,full_matrices=True);normal=V[-1]
                if normal@(origin-point)<0:normal=-normal
                sigma=max(cfg['normal_sigma_floor_rad'],min(1.5,4*cfg['depth_noise_bound_m']/max(float(s[-1]),1e-12)))
                normals.append(normal);sigmas.append(sigma)
        lookup[(v,x,y)]=len(pixels);pixels.append({'epoch':raw[0]['epoch'],'view':v,'pixel':[x,y],'origin':origin,'ray':ray,'depth':z,'normals':np.array(normals).reshape(-1,3),'normal_sigma':np.array(sigmas)})
    boundary=[]
    for row in raw:
        copy=dict(row);copy['pixel_index']=lookup[(row['view'],*row['inside_pixel'])];boundary.append(copy)
    return {'pixels':pixels,'boundary':boundary,'unique_depth_variables':len(pixels)}


def axis_evidence(pixels,x,cfg):
    R=Rotation.from_rotvec(x[:3]).as_matrix();weights=[]
    for p in pixels:
        if not len(p['normals']):weights.append(np.zeros(3));continue
        cosine=p['normals']@R;angular=1-cosine*cosine;cost=np.min(angular/p['normal_sigma'][:,None]**2,axis=0);prob=np.exp(-.5*(cost-cost.min()));prob/=prob.sum()
        # Uniform evidence must not count as a detected physical plane.
        weights.append(np.maximum((3*prob-1)/2,0))
    return np.array(weights)


def induce_graph(references,cfg):
    points=[];weights=[];regions=[]
    for x,data in references:
        R=Rotation.from_rotvec(x[:3]).as_matrix();pp=data['pixels'];P=np.array([p['origin']+p['ray']*p['depth'] for p in pp]);points.extend((P-x[3:])@R);weights.extend(axis_evidence(pp,x,cfg));rr,_=all_regions(x,data['boundary'],cfg)
        for r in rr:
            b=data['boundary'][r['ray_id']];r['token']=(b['epoch'],b['view'],*b['inside_pixel']);r['point']=(P[b['pixel_index']]-x[3:])@R
        regions.extend(rr)
    P=np.array(points);W=np.array(weights);faces=[];radius=cfg['face_mode_radius_m'];minimum=cfg['minimum_face_pixels']
    if not len(P):return {'faces':[],'edges':[],'reference_pixels':0}
    for a in range(3):
        order=np.argsort(P[:,a]);v=P[order,a];w=W[order,a].copy();left=np.searchsorted(v,v-radius);right=np.searchsorted(v,v+radius,side='right')
        while w.sum()>=minimum:
            cumulative=np.r_[0.,np.cumsum(w)];gain=cumulative[right]-cumulative[left];j=int(gain.argmax())
            if gain[j]<minimum:break
            ix=np.arange(left[j],right[j]);ww=w[ix];median_index=int(np.searchsorted(np.cumsum(ww),ww.sum()/2));d=float(v[ix[min(median_index,len(ix)-1)]]);use=abs(v-d)<=radius;support=float(w[use].sum())
            if support<minimum-1e-9:w[ix]=0.;continue
            faces.append({'axis':a,'offset':d,'effective_normal_support':support});w[use]=0.
    by_key={}
    for key in [(0,1),(0,2),(1,2)]:
        rr=[r for r in regions if r['key']==key]
        if rr:by_key[key]=(rr,np.array([r['vertices'] for r in rr]))
    edges=[]
    for f,F in enumerate(faces):
        for g in range(f+1,len(faces)):
            G=faces[g];a=F['axis'];k=G['axis'];key=tuple(sorted((a,k)))
            if a==k or key not in by_key:continue
            support=np.array([F['offset'],G['offset']]) if a<k else np.array([G['offset'],F['offset']]);rr,V=by_key[key];E=np.roll(V,-1,axis=1)-V;delta=support-V;cross=E[:,:,0]*delta[:,:,1]-E[:,:,1]*delta[:,:,0];area=abs(np.sum(V[:,:,0]*np.roll(V[:,:,1],-1,axis=1)-V[:,:,1]*np.roll(V[:,:,0],-1,axis=1),axis=1));inside=(area>1e-18)&(np.all(cross>=-1e-12,axis=1)|np.all(cross<=1e-12,axis=1));den=(E*E).sum(2);t=np.clip(np.divide((delta*E).sum(2),den,out=np.zeros_like(den),where=den>1e-28),0,1);dist=np.where(inside,0.,np.linalg.norm(delta-t[:,:,None]*E,axis=2).min(1));witness={}
            for j in np.flatnonzero(dist<=cfg['graph_witness_tolerance_m']):witness[rr[j]['token']]=float(rr[j]['point'][3-a-k])
            values=np.sort(list(witness.values()))
            for part in np.split(values,np.flatnonzero(np.diff(values)>cfg['edge_span_gap_m'])+1):
                if len(part)>=cfg['minimum_edge_pixels']:edges.append({'faces':[f,g],'span_axis':3-a-k,'span':[float(part.min()),float(part.max())]})
    return {'faces':faces,'edges':edges,'reference_pixels':len(P),'normal_conditioned_dictionary_prior':True}


class JointProblem:
    def __init__(self,data,graph,initial,limits,cfg,method):
        self.cfg=cfg;self.graph=graph;self.method=method;self.limits=limits;self.faces=graph['faces'];self.F=len(self.faces);self.initial=np.array(initial,float);self.free_faces=method!='normal_fixed';self.free_depth=method in ['bounded_joint','bounded_no_incidence'];self.incidence=method!='bounded_no_incidence';self.rows=[];self.boundary=[];offset=0
        for t,d in enumerate(data):
            self.rows.extend(d['pixels'])
            for b in d['boundary']:copy=dict(b);copy['pixel_index']+=offset;self.boundary.append(copy)
            offset+=len(d['pixels'])
        self.N=len(self.rows);self.M=len(self.boundary);self.epoch=np.array([p['epoch'] for p in self.rows]);self.O=np.array([p['origin'] for p in self.rows]);self.ray=np.array([p['ray'] for p in self.rows]);self.z=np.array([p['depth'] for p in self.rows]);self.pi=np.array([b['pixel_index'] for b in self.boundary],int);self.ro=np.array([b['ray_outside'] for b in self.boundary]);self.outdepth=np.array([b['outside_depth'] for b in self.boundary]);self.be=self.epoch[self.pi];self.face0=np.array([f['offset'] for f in self.faces]);self.axes=np.array([f['axis'] for f in self.faces],int);self.face_start=12;self.depth_start=12+(self.F if self.free_faces else 0);self.size=self.depth_start+(self.N if self.free_depth else 0)
        self.pweight=np.array([1/np.sqrt(3*max(1,np.sum(self.epoch==t))) for t in self.epoch]);self.bweight=np.array([1/np.sqrt(3*max(1,np.sum(self.be==t))) for t in self.be]);self.normal=np.zeros((self.N,4,3));self.sigmas=np.full((self.N,4),np.inf)
        for j,p in enumerate(self.rows):
            n=len(p['normals']);self.normal[j,:n]=p['normals'];self.sigmas[j,:n]=p['normal_sigma']

    def start(self):return np.r_[self.initial,self.face0 if self.free_faces else [],self.z if self.free_depth else []]

    def project(self,p):
        p=p.copy()
        for j in [0,6]:
            for off,limit in [(0,self.limits[0]),(3,self.limits[1])]:
                v=p[j+off:j+off+3];n=np.linalg.norm(v)
                if n>limit:p[j+off:j+off+3]*=limit/n
        if self.free_faces:p[12:12+self.F]=np.clip(p[12:12+self.F],-self.cfg['face_workspace_bound_m'],self.cfg['face_workspace_bound_m'])
        if self.free_depth:p[self.depth_start:]=np.clip(p[self.depth_start:],self.z-self.cfg['depth_noise_bound_m'],self.z+self.cfg['depth_noise_bound_m'])
        return p

    def geometry(self,p):
        rotations=np.array([np.eye(3)]+[Rotation.from_rotvec(p[j:j+3]).as_matrix() for j in [0,6]]);Jr=np.array([np.zeros((3,3))]+[right_jacobian(p[j:j+3]) for j in [0,6]]);u=np.array([np.zeros(3),p[3:6],p[9:12]]);R=rotations[self.epoch];O=np.einsum('ni,nij->nj',self.O-u[self.epoch],R);ri=np.einsum('ni,nij->nj',self.ray,R);ro=np.einsum('ni,nij->nj',self.ro,rotations[self.be]);z=p[self.depth_start:] if self.free_depth else self.z;Q=O+ri*z[:,None];d=p[12:12+self.F] if self.free_faces else self.face0
        return rotations,Jr,R,O,ri,ro,z,Q,d

    def evaluate(self,p,jacobian=True):
        cfg=self.cfg;cap=cfg['support_match_cap_m'];rotations,Jr,R,O,ri,ro,z,Q,d=self.geometry(p);pixel=np.zeros((self.N,4));pixel[:,0]=cap;br=np.zeros((self.M,5));br[:,0]=cap if self.incidence else 0.;face=np.full(self.N,-1,int);normal_id=np.zeros(self.N,int);edge_id=np.full(self.M,-1,int);best=np.full(self.N,np.inf);all_normal={}
        if self.F:
            for f,a in enumerate(self.axes):
                axis_world=R[:,:,a];cross=np.cross(self.normal,axis_world[:,None,:]);scaled=cross*cfg['normal_residual_scale_m']/self.sigmas[:,:,None];ncost=(scaled*scaled).sum(2);missing=~np.isfinite(self.sigmas);ncost[missing&(~missing).any(1)[:,None]]=np.inf;choice=ncost.argmin(1);nres=scaled[np.arange(self.N),choice];pr=np.c_[Q[:,a]-d[f],nres];pcost=np.minimum((pr*pr).sum(1),cap*cap);bc=np.full(self.M,cap*cap) if self.incidence else np.zeros(self.M);bv=np.zeros((self.M,5));bv[:,0]=cap if self.incidence else 0.;ei=np.full(self.M,-1,int)
                if self.incidence:
                    for e,edge in enumerate(self.graph['edges']):
                        if f not in edge['faces']:continue
                        g=edge['faces'][1] if edge['faces'][0]==f else edge['faces'][0];k=self.axes[g];h=edge['span_axis'];B=ro[:,a];A=ri[self.pi,a];stable=(abs(B)>cfg['denominator_tolerance'])&(abs(A)>cfg['denominator_tolerance']);lam=np.divide(d[f]-O[self.pi,a],B,out=np.full(self.M,-1.),where=stable);ratio=np.divide(A,B,out=np.zeros(self.M),where=stable);step=ratio*ro[:,k]-ri[self.pi,k];s=np.sign(step);valid=stable&(ratio>0)&(lam>0)&(abs(step)>cfg['denominator_tolerance']);outside=O[self.pi,k]+ro[:,k]*lam;lo,hi=edge['span'];margin=cfg['edge_span_margin_m'];values=np.c_[np.maximum(s*(Q[self.pi,k]-d[g]),0),np.maximum(s*(d[g]-outside),0),abs(B)*np.maximum(lam-self.outdepth+cfg['depth_noise_bound_m'],0),np.maximum(lo-margin-Q[self.pi,h],0),np.maximum(Q[self.pi,h]-hi-margin,0)];value=np.minimum((values*values).sum(1),cap*cap);value[~valid]=cap*cap;use=value<bc;bc[use]=value[use];bv[use]=values[use];ei[use]=e
                total=pcost*self.pweight**2;np.add.at(total,self.pi,bc*self.bweight**2);use=total<best;best[use]=total[use];face[use]=f;pixel[use]=pr[use];normal_id[use]=choice[use]
                take=use[self.pi];br[take]=bv[take];edge_id[take]=ei[take]
        # Capped blocks retain all observations; saturated blocks have zero gradient.
        pnorm=np.linalg.norm(pixel,axis=1);brr=np.linalg.norm(br,axis=1);pactive=pnorm<cap;bactive=(brr<cap)&(edge_id>=0);pixel[~pactive]=[cap,0,0,0];br[brr>=cap]=[cap,0,0,0,0];r=np.r_[(pixel*self.pweight[:,None]).ravel(),(br*self.bweight[:,None]).ravel()]
        diag={'pixel_face':face,'edge':edge_id,'normal_id':normal_id,'surface_saturated':int((~pactive).sum()),'incidence_saturated':int((~bactive).sum()) if self.incidence else 0}
        if not jacobian:return r,None,diag
        rows=[];cols=[];values=[]
        def add(row,col,value):
            row,col,value=np.broadcast_arrays(row,col,value);keep=(col>=0)&(col<self.size)&(abs(value)>0);rows.extend(row[keep].ravel());cols.extend(col[keep].ravel());values.extend(value[keep].ravel())
        def qjac(indices,axis,coords):
            ea=np.eye(3)[axis];rot=np.einsum('ni,nij->nj',np.cross(ea,coords),Jr[self.epoch[indices]]);trans=-R[indices,:,axis];return np.c_[rot,trans]
        def pose_add(base,J,epochs,scale):
            col=6*(epochs-1)[:,None,None]+np.arange(6)[None,None,:];weight=scale[:,None,None]*(epochs>0)[:,None,None];add(base[:,:,None],col,J*weight)
        use=np.flatnonzero((face>=0)&pactive)
        if len(use):
            f=face[use];a=self.axes[f];base=4*use[:,None]+np.arange(4)[None];Jp=np.zeros((len(use),4,6));Jp[:,0]=qjac(use,a,Q[use]);n=self.normal[use,normal_id[use]];sc=cfg['normal_residual_scale_m']/self.sigmas[use,normal_id[use]];da=np.array([-R[j]@skew(np.eye(3)[aa])@Jr[self.epoch[j]] for j,aa in zip(use,a)]);Jp[:,1:,:3]=np.cross(n[:,None,:],da.transpose(0,2,1)).transpose(0,2,1)*sc[:,None,None];pose_add(base,Jp,self.epoch[use],self.pweight[use])
            if self.free_faces:add(base[:,0],12+f,-self.pweight[use])
            if self.free_depth:add(base[:,0],self.depth_start+use,ri[use,a]*self.pweight[use])
        for f in np.unique(face[self.pi][bactive]):
            for e in np.unique(edge_id[bactive&(face[self.pi]==f)]):
                ix=np.flatnonzero(bactive&(face[self.pi]==f)&(edge_id==e));pp=self.pi[ix];edge=self.graph['edges'][e];g=edge['faces'][1] if edge['faces'][0]==f else edge['faces'][0];a=self.axes[f];k=self.axes[g];h=edge['span_axis'];B=ro[ix,a];A=ri[pp,a];lam=(d[f]-O[pp,a])/B;s=np.sign(A/B*ro[ix,k]-ri[pp,k]);Jo_a=qjac(pp,np.full(len(ix),a),O[pp]);Jo_k=qjac(pp,np.full(len(ix),k),O[pp]);Jq_k=qjac(pp,np.full(len(ix),k),Q[pp]);Jq_h=qjac(pp,np.full(len(ix),h),Q[pp]);Jb=np.zeros_like(Jo_a);Jb[:,:3]=np.einsum('ni,nij->nj',np.cross(np.eye(3)[a],ro[ix]),Jr[self.be[ix]]);Jro_k=np.zeros_like(Jo_a);Jro_k[:,:3]=np.einsum('ni,nij->nj',np.cross(np.eye(3)[k],ro[ix]),Jr[self.be[ix]]);Jl=(-Jo_a-lam[:,None]*Jb)/B[:,None];Jout=Jo_k+lam[:,None]*Jro_k+ro[ix,k,None]*Jl;J=np.zeros((len(ix),5,6));J[:,0]=s[:,None]*Jq_k;J[:,1]=-s[:,None]*Jout;J[:,2]=abs(B)[:,None]*Jl+np.maximum(lam-self.outdepth[ix]+cfg['depth_noise_bound_m'],0)[:,None]*np.sign(B)[:,None]*Jb;J[:,3]=-Jq_h;J[:,4]=Jq_h;active=br[ix]>0;J*=active[:,:,None];base=4*self.N+5*ix[:,None]+np.arange(5)[None];pose_add(base,J,self.be[ix],self.bweight[ix])
                if self.free_faces:
                    d_f=np.c_[np.zeros(len(ix)),-s*ro[ix,k]/B,np.sign(B),np.zeros((len(ix),2))]*active;d_g=np.c_[-s,s,np.zeros((len(ix),3))]*active;add(base,12+f,d_f*self.bweight[ix,None]);add(base,12+g,d_g*self.bweight[ix,None])
                if self.free_depth:
                    Jz=np.c_[s*ri[pp,k],np.zeros((len(ix),2)),-ri[pp,h],ri[pp,h]]*active;add(base,self.depth_start+pp[:,None],Jz*self.bweight[ix,None])
        J=coo_matrix((values,(rows,cols)),shape=(len(r),self.size)).tocsr();return r,J,diag

    def fit(self):
        p=self.project(self.start());r,J,diag=self.evaluate(p);cost=float(r@r);initial_cost=cost;trace=[]
        for it in range(self.cfg['joint_iterations']):
            step=lsmr(J,-r,damp=self.cfg['lsmr_damping'],atol=1e-7,btol=1e-7,maxiter=self.cfg['lsmr_iterations'])[0];accepted=False
            for back in range(self.cfg['backtracking_steps']):
                candidate=self.project(p+step*.5**back);rr,_,_=self.evaluate(candidate,False);value=float(rr@rr)
                if value<cost-self.cfg['descent_tolerance']:
                    trace.append({'iteration':it,'before':cost,'after':value,'backtracks':back});p=candidate;r,J,diag=self.evaluate(p);cost=value;accepted=True;break
            if not accepted:break
        return p,{'initial_cost':initial_cost,'final_cost':cost,'accepted_steps':trace,'surface_saturated':diag['surface_saturated'],'incidence_saturated':diag['incidence_saturated'],'pixels':self.N,'boundary_pairs':self.M,'faces':self.F,'edges':len(self.graph['edges']),'depths_at_bounds':int(np.sum(abs(p[self.depth_start:]-self.z)>=self.cfg['depth_noise_bound_m']-1e-9)) if self.free_depth else 0,'identities_fixed':True,'graph_topology_fixed':True}
