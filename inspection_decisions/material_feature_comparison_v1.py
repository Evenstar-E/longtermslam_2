"""Common measured sampling, trained descriptors, and matched-budget inference."""
import numpy as np
import cv2
import torch
from inspection_decisions.material_joint_descriptor_v1 import sample_frame,metric_patches
from inspection_decisions import textured_material_matching_v1 as base


def rootsift(patches):
    # Handcrafted control at precisely the same metric-normalized sample centers.
    gray=patches[:,:3].mean(1)
    sift=cv2.SIFT_create();out=[]
    for p in gray:
        img=np.clip((p+3)*255/6,0,255).astype(np.uint8)
        img=cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR)
        _,d=sift.compute(img,[cv2.KeyPoint(15.5,15.5,8.,angle=0.)])
        d=np.zeros(128) if d is None else d[0]
        out.append(np.sqrt(d/max(float(d.sum()),1e-10)))
    return np.asarray(out,np.float32)


def measured_epoch(frames,cfg):
    counts=np.zeros(24,int)
    for frame in frames:
        for slot in np.unique(frame['anonymous_slot']):
            if slot>=0:counts[int(slot)]+=1
    pieces=[]
    for frame in frames:
        # Equal per-view quota prevents a close-up from consuming every feature.
        selected=sample_frame(frame,cfg['features_per_slot'])
        uv=[];slots=[]
        for slot,(u,p) in selected.items():
            quota=max(1,cfg['features_per_slot']//max(counts[slot],1))
            ix=np.linspace(0,len(u)-1,min(quota,len(u)),dtype=int)
            uv.append(u[ix]);slots.extend([slot]*len(ix))
        uv=np.concatenate(uv) if uv else np.zeros((0,2))
        patch,xyz=metric_patches(frame,uv,cfg)
        pieces.append(dict(frame=frame,uv=uv,xyz=xyz,slots=np.array(slots,int),patch=patch))
    return pieces


@torch.inference_mode()
def feature_epochs(pieces,models,pretrained,cfg,geometry_epochs):
    names=['metric_rootsift']+list(models)+['superpoint']
    result={name:[] for name in names};lg_frames=[]
    for e,views in enumerate(pieces):
        buckets={name:[dict(xyz=[],desc=[]) for _ in range(24)] for name in names}
        local=[];offset=np.zeros(24,int)
        for v in views:
            x=torch.from_numpy(v['patch']);ds={'metric_rootsift':rootsift(v['patch'])}
            for name,model in models.items():
                ds[name]=np.concatenate([model(z).numpy() for z in x.split(512)]) if len(x) else np.zeros((0,64),np.float32)
            ds['superpoint']=pretrained.descriptors(v['frame'],v['uv']) if len(x) else np.zeros((0,256),np.float32)
            indices=np.zeros(len(v['uv']),int)
            for slot in np.unique(v['slots']):
                take=v['slots']==slot;n=int(take.sum());indices[take]=np.arange(n)+offset[slot];offset[slot]+=n
                for name in names:
                    buckets[name][slot]['xyz'].append(v['xyz'][take]);buckets[name][slot]['desc'].append(ds[name][take])
            h,w=v['frame']['depth'].shape
            local.append(dict(uv=v['uv'].astype(np.float32),desc=ds['superpoint'],size=(w,h),slots=v['slots'],indices=indices))
        lg_frames.append(local)
        for name in names:
            epoch=[]
            for slot in range(24):
                b=buckets[name][slot];a=geometry_epochs[e][slot]
                epoch.append(dict(points=a['points'],tree=a['tree'],
                    xyz=np.concatenate(b['xyz']) if b['xyz'] else np.zeros((0,3)),
                    desc=np.concatenate(b['desc']) if b['desc'] else np.zeros((0,256 if name=='superpoint' else 128 if name=='metric_rootsift' else 64),np.float32)))
            result[name].append(epoch)
    return result,lg_frames


def official_matches(lg_frames,pretrained,epochs,progress=None):
    out={}
    for e,f in [(0,1),(0,2),(1,2)]:
        raw=[[[] for _ in range(24)] for _ in range(24)]
        for va in lg_frames[e]:
            for vb in lg_frames[f]:
                pairs=pretrained.match(va,vb)
                for ia,ib in pairs:
                    i,j=int(va['slots'][ia]),int(vb['slots'][ib])
                    raw[i][j].append((int(va['indices'][ia]),int(vb['indices'][ib])))
        mat=[]
        for i in range(24):
            row=[]
            for j in range(24):
                pair=np.unique(np.array(raw[i][j],int).reshape(-1,2),axis=0)
                ia,ib=pair.T
                dd=np.linalg.norm(epochs[e][i]['desc'][ia]-epochs[f][j]['desc'][ib],axis=1)
                row.append(dict(ia=ia,ib=ib,distance=dd))
            mat.append(row)
        out[e,f]=mat
        if progress:progress('official LightGlue '+str((e,f)))
    return out


def head_weight(a,b,match,model):
    if not len(match['ia']):return np.zeros(0)
    with torch.inference_mode():
        aa=torch.from_numpy(a['desc']).float();bb=torch.from_numpy(b['desc']).float()
        m=model.context_logits(aa,bb).sigmoid().numpy()[match['ia']]
        c=model.persistence_logits(aa[match['ia']],bb[match['ib']]).sigmoid().numpy()
    # Appearance change is not physical disappearance: c never deletes geometry.
    return m*(.5+.5*c)


def compare(epochs,frames,cfg,gp,seed,model=None,precomputed=None,progress=None):
    methods=['fusion'] if model is None else ['fusion','visibility','material']
    costs={m:{} for m in methods};single={m:{} for m in methods};pools={};matches={}
    def score(m,e,f,i,j,T):
        a,b=epochs[e][i],epochs[f][j];mm=matches[e,f][i][j]
        g=base.geometry_loss(a,b,T,cfg);t=base.texture_loss(a,b,mm,T,cfg)
        if m=='fusion':return g+t
        vis=base.visibility_loss(a,b,T,frames[e],frames[f],i,j,cfg)
        if m=='visibility':return g+t+.5*vis
        # Same geometry coefficient as ordinary fusion. Heads are soft evidence,
        # and are not a hard rejection of all visible appearance-changed points.
        return g+base.texture_loss(a,b,mm,T,cfg,mm['head_weight'])+.5*vis
    for e,f in [(0,1),(0,2),(1,2)]:
        matches[e,f]=[];pools[e,f]=[]
        for m in methods:costs[m][e,f]=np.zeros((24,24));single[m][e,f]=np.zeros((24,24,4,4))
        for i,a in enumerate(epochs[e]):
            mr=[];pr=[]
            for j,b in enumerate(epochs[f]):
                mm=precomputed[e,f][i][j] if precomputed is not None else base.matches(a,b,cfg)
                if model is not None:mm={**mm,'head_weight':head_weight(a,b,mm,model)}
                mr.append(mm)
                T=base.ransac(a,b,mm,cfg,seed+e*10000+f*1000+i*24+j)
                pool=list(gp[e,f][i][j])
                if T is not None:pool += [T,base.icp(a,b,T,cfg)]
                pr.append(pool)
            matches[e,f].append(mr);pools[e,f].append(pr)
        for m in methods:
            for i in range(24):
                for j in range(24):
                    pool=pools[e,f][i][j];values=[score(m,e,f,i,j,T) for T in pool];k=int(np.argmin(values))
                    costs[m][e,f][i,j]=values[k];single[m][e,f][i,j]=pool[k]
        if progress:progress('shared proposals '+str((e,f)))
    predictions={}
    for m in methods:
        p1,p2=base.solve_assignments(costs[m],cfg);Ts=[];objectives=[]
        for i,(j,k) in enumerate(zip(p1,p2)):
            best=None
            for T1 in pools[0,1][i][j]:
                for T2 in pools[0,2][i][k]:
                    value=score(m,0,1,i,j,T1)+score(m,0,2,i,k,T2)+cfg['cycle_weight']*score(m,1,2,j,k,T2@np.linalg.inv(T1))
                    if best is None or value<best[0]:best=(value,T1,T2)
            objectives.append(best[0]);Ts.append([best[1],best[2]])
        predictions[m]=dict(target_slots=np.c_[p1,p2],transforms=np.array(Ts),triplet_counts=np.zeros(24,int),objectives=np.array(objectives))
    packed=np.tile(np.eye(4),(3,24,24,4,1,1));counts=np.zeros((3,24,24),int)
    for pe,key in enumerate([(0,1),(0,2),(1,2)]):
        for i in range(24):
            for j in range(24):
                values=pools[key][i][j];counts[pe,i,j]=len(values);packed[pe,i,j,:len(values)]=values
    return predictions,dict(features=[[len(a['desc']) for a in ep] for ep in epochs],
        matches={f'{e}-{f}':[[len(mm['ia']) for mm in row] for row in mat] for (e,f),mat in matches.items()},
        pool_sizes={f'{e}-{f}':[[len(p) for p in row] for row in mat] for (e,f),mat in pools.items()}),dict(transforms=packed,counts=counts)
