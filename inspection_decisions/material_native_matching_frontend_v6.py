"""Native official detection/matching on observed ROIs, measured pixel support."""
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from inspection_decisions.material_pretrained_v1 import PretrainedLocal
from inspection_decisions.textured_material_matching_v1 import world_points


class NativeMatching:
    def __init__(self,cfg):
        self.cfg=cfg;self.native=PretrainedLocal()
        self.native.extractor.conf.max_num_keypoints=cfg['detector_keypoints']
        for k in ['depth_confidence','width_confidence','filter_threshold']:setattr(self.native.matcher.conf,k,cfg[k])

    @torch.inference_mode()
    def extract(self,frame,slot,frame_index,epoch):
        h,w=frame['depth'].shape;y,x=np.nonzero((frame['anonymous_slot']==slot)&frame['depth_valid'])
        if len(x)<16:return None
        margin=self.cfg['roi_margin_px'];x0=max(0,int(x.min())-margin);x1=min(w,int(x.max())+1+margin);y0=max(0,int(y.min())-margin);y1=min(h,int(y.max())+1+margin)
        rgb=frame['rgb'][y0:y1,x0:x1];scale=self.cfg['roi_long_side']/max(rgb.shape[:2]);hh=max(8,int(round(rgb.shape[0]*scale)));ww=max(8,int(round(rgb.shape[1]*scale)))
        image=cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_LINEAR);image=torch.from_numpy(image.transpose(2,0,1).copy()).float()[None]/255
        image=F.pad(image,(0,(-ww)%8,0,(-hh)%8),mode='replicate');out=self.native.extractor({'image':image})
        uv=out['keypoints'][0].numpy();quality=out['keypoint_scores'][0].numpy();desc=out['descriptors'][0].numpy()
        # OpenCV resize is half-pixel centered. Convert the extractor's declared
        # coordinate origin before returning to original measured pixel centers.
        center=uv-float(self.native.extractor.coords_offset)
        original=(center+.5)/np.array([ww/(x1-x0),hh/(y1-y0)])-.5+[x0,y0]
        pixel=np.rint(original).astype(int);inside=(center[:,0]>=0)&(center[:,0]<ww)&(center[:,1]>=0)&(center[:,1]<hh)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
        ii=np.flatnonzero(inside);p=pixel[ii];ii=ii[(frame['anonymous_slot'][p[:,1],p[:,0]]==slot)&frame['depth_valid'][p[:,1],p[:,0]]]
        # One native keypoint per ACTUAL measured depth pixel. Upsampling is not
        # a new measurement and cannot multiply material support.
        order=ii[np.argsort(-quality[ii],kind='stable')];seen=set();keep=[]
        for k in order:
            key=tuple(pixel[k])
            if key in seen:continue
            seen.add(key);keep.append(k)
            if len(keep)>=self.cfg['keypoints_per_view']:break
        keep=np.array(keep,int);pixel=pixel[keep];node=epoch*100000000+frame_index*1000000+pixel[:,1]*w+pixel[:,0]
        return dict(uv=uv[keep].astype(np.float32),desc=desc[keep].astype(np.float32),quality=quality[keep],size=(ww,hh),
            original_uv=original[keep],pixel=pixel,xyz=world_points(frame,pixel),node=node.astype(np.int64),frame=frame_index,slot=slot,epoch=epoch,
            crop=np.array([x0,y0,x1,y1]),original_size=(w,h),origin_offset=float(self.native.extractor.coords_offset))

    @torch.inference_mode()
    def match(self,a,b):
        if a is None or b is None or not len(a['uv']) or not len(b['uv']):return np.zeros((0,2),int),np.zeros(0),0
        def item(x):return dict(keypoints=torch.from_numpy(x['uv'])[None],descriptors=torch.from_numpy(x['desc'])[None],image_size=torch.tensor([x['size']],dtype=torch.float32))
        out=self.native.matcher(dict(image0=item(a),image1=item(b)))
        return out['matches'][0].numpy(),out['scores'][0].numpy(),int(out['stop'])


def selected_views(frames,slot,count):
    sizes=[int(((f['anonymous_slot']==slot)&f['depth_valid']).sum()) for f in frames]
    return [int(i) for i in np.argsort(-np.array(sizes),kind='stable')[:count] if sizes[i]>=16]


def extract_epochs(all_frames,matcher,cfg):
    epochs=[]
    for e,frames in enumerate(all_frames):
        groups=[]
        for slot in range(24):
            views=[matcher.extract(frames[fi],slot,fi,e) for fi in selected_views(frames,slot,cfg['views_per_slot'])]
            for a in views:
                if a is not None:a['raw_node']=a['node'].copy();a['raw_xyz']=a['xyz'].copy()
            # Same-epoch views have known camera extrinsics and a static object.
            # Merge only actual native matches witnessed by measured 3D proximity.
            if len(views)==2 and all(a is not None for a in views):
                a,b=views;pair,_,_=matcher.match(a,b)
                for ia,ib in pair:
                    if np.linalg.norm(a['raw_xyz'][ia]-b['raw_xyz'][ib])>cfg['same_epoch_gate_m']:continue
                    node=min(a['raw_node'][ia],b['raw_node'][ib]);point=a['raw_xyz'][ia] if a['quality'][ia]>=b['quality'][ib] else b['raw_xyz'][ib]
                    a['node'][ia]=node;b['node'][ib]=node;a['xyz'][ia]=point;b['xyz'][ib]=point
            groups.append(views)
        epochs.append(groups)
    return epochs


def view_pairs(a,b):
    if len(a)<2 or len(b)<2:return [(aa,bb) for aa in a for bb in b]
    def affinity(x,y):
        if x is None or y is None or not len(x['desc']) or not len(y['desc']):return -1.
        d=x['desc']@y['desc'].T
        return float(.5*(np.mean(d.max(1))+np.mean(d.max(0))))
    diagonal=affinity(a[0],b[0])+affinity(a[1],b[1]);cross=affinity(a[0],b[1])+affinity(a[1],b[0])
    # Two view-pairs cover both observed views, chosen without identity labels.
    return [(a[0],b[0]),(a[1],b[1])] if diagonal>=cross else [(a[0],b[1]),(a[1],b[0])]


def pair_edges(epochs,matcher,progress=None):
    records={};layers=[];calls=0
    for e,f in [(0,1),(0,2),(1,2)]:
        values=[]
        for i in range(24):
            for j in range(24):
                xx=[];yy=[];source=[];target=[];weights=[]
                for a,b in view_pairs(epochs[e][i],epochs[f][j]):
                        matches,confidence,stop=matcher.match(a,b);calls+=1
                        if stop:layers.append(stop)
                        if not len(matches):continue
                        ia,ib=matches.T;xx.extend(a['xyz'][ia]);yy.extend(b['xyz'][ib]);source.extend(a['node'][ia]);target.extend(b['node'][ib]);weights.extend(confidence)
                values.append(dict(i=i,j=j,x=np.array(xx).reshape(-1,3),y=np.array(yy).reshape(-1,3),source=np.array(source,np.int64),target=np.array(target,np.int64),weight=np.array(weights)))
            if progress:progress(f'{e}-{f} source {i+1}/24')
        records[e,f]=values
    return records,dict(matcher_calls=calls,mean_layers=float(np.mean(layers)) if layers else 0,empty_calls=calls-len(layers))
