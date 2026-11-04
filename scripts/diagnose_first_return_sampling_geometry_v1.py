"""Exact GT-ray versus raster neighborhood, and phantom interpolation diagnosis."""
from pathlib import Path
import pickle
import numpy as np
import open3d as o3d
from scipy.ndimage import minimum_filter
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import render,transform,inverse,ray_box
from inspection_decisions.first_return_material_surface_v1 import build_patches,project


def main():
    root=Path('results/first_return_material_surface_v1');cfg=read(root/'config.json');rows=[];witnesses=[]
    for p,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/p)==h
    for run in read(root/'runs.json'):
        dest=root/run['scene']
        with (dest/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        geometry=geometry_for(run['seed'],cfg['objects'],run['geometry']);slots=inp['T0'][:,:3,3];groups=np.arange(cfg['objects'])%4
        frames=[]
        for t in range(3):
            ff=[]
            for v,path in enumerate(sorted(dest.glob(f'frame_{t}_*.npz'))):
                C=np.load(path)['C'];poses=inp['T0'] if t==0 else truth['poses'][t-1]
                frame,_=render(geometry,poses,groups,C,dict(cfg,depth_noise_bound_m=0.),np.random.default_rng(run['seed']+10000*(t+1)+v))
                frame['min3']=minimum_filter(frame['depth'],size=3,mode='constant',cval=0.);ff.append(frame)
            frames.append(ff)
        patches=[build_patches(ff,slots,groups,cfg) for ff in frames]
        for i in range(cfg['objects']):
            ids=[i,int(truth['slot1'][i]),int(truth['slot2'][i])];Ts=[inp['T0'][i],truth['poses'][0,i],truth['poses'][1,i]]
            count=exactbad=surfacebad=phantom=0;maxexact=0.;maxsurface=0.
            for s,t in [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]:
                patch=patches[s][ids[s]];M=Ts[t]@np.linalg.inv(Ts[s]);world=transform(patch.vertices,M)
                for view,frame in enumerate(frames[t]):
                    cam,uv=project(world,frame['C'],cfg);pix=np.rint(uv).astype(int)
                    valid=(cam[:,2]>0)&(pix[:,0]>=0)&(pix[:,0]<cfg['width'])&(pix[:,1]>=0)&(pix[:,1]<cfg['height'])
                    delta=frame['min3'][pix[valid,1],pix[valid,0]]-cam[valid,2];count+=int(np.sum(delta>cfg['depth_tolerance_m']))
                    # Exact continuous projected ray through each point, not rounded.
                    origin=inverse(frame['C'][:3,3][None],Ts[t])[0]
                    directions=((world-frame['C'][:3,3])/cam[:,2,None])@Ts[t][:3,:3]
                    exact=np.min([ray_box(origin,directions,lo,hi) for lo,hi in geometry[i]],axis=0)
                    gap=exact-cam[:,2];exactbad+=int(np.sum(gap>cfg['depth_tolerance_m']));maxexact=max(maxexact,float(np.max(gap,initial=0.)))
                    if patch.scene is None:continue
                    low=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);high=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[cfg['width']-1,cfg['height']-1])
                    if (low>high).any():continue
                    yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
                    rc=np.c_[(xx.ravel()-(cfg['width']-1)/2)/cfg['focal_px'],(yy.ravel()-(cfg['height']-1)/2)/cfg['focal_px'],np.ones(xx.size)]
                    target_dirs=rc@frame['C'][:3,:3].T
                    ro=inverse(frame['C'][:3,3][None],M)[0];rd=target_dirs@M[:3,:3]
                    rays=np.c_[np.broadcast_to(ro,rd.shape),rd].astype(np.float32)
                    ans=patch.scene.cast_rays(o3d.core.Tensor(rays),nthreads=1);z=ans['t_hit'].numpy()
                    gap=frame['depth'][yy.ravel(),xx.ravel()]-z;bad=gap>cfg['depth_tolerance_m']
                    surfacebad+=int(bad.sum())
                    if not bad.any():continue
                    point=frame['C'][:3,3]+z[bad,None]*target_dirs[bad];canonical=inverse(point,Ts[t])
                    inside=np.any([np.all((canonical>=lo-1e-6)&(canonical<=hi+1e-6),axis=1) for lo,hi in geometry[i]],axis=0)
                    phantom+=int(np.sum(~inside));maxsurface=max(maxsurface,float(gap[bad].max()))
                    if len(witnesses)<20 or (gap.max()>10 and not any(w.get('gap_m',0)>10 for w in witnesses)):
                        j=int(np.argmax(gap));primitive=int(ans['primitive_ids'].numpy()[j]);triangle=patch.vertices[patch.triangles[primitive]]
                        witnesses.append({'scene':run['scene'],'asset':i,'periods':[s,t],'target_view':view,'pixel':[int(xx.ravel()[j]),int(yy.ravel()[j])],
                            'gap_m':float(gap[j]),'triangle_source_canonical':inverse(triangle,Ts[s]).tolist()})
            rows.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'noiseless_point_neighborhood_conflicts':count,
                         'noiseless_point_exact_ray_conflicts':exactbad,'max_exact_ray_gap_m':maxexact,'noiseless_mesh_conflicts':surfacebad,
                         'mesh_conflicts_outside_true_solid':phantom,'max_mesh_gap_m':maxsurface})
        print('EXACT_RAY_DIAG',run['scene'],flush=True)
    summary={mode:{key:sum(r[key] for r in rows if r['geometry']==mode) for key in ['noiseless_point_neighborhood_conflicts','noiseless_point_exact_ray_conflicts','noiseless_mesh_conflicts','mesh_conflicts_outside_true_solid']} for mode in cfg['geometry_modes']}
    save(root/'sampling_geometry_diagnosis.json',{'GT_assisted':True,'updates_predictions':False,'summary':summary,'rows':rows,'witnesses':witnesses,'script_sha256':sha(Path(__file__))})
    print(summary)


if __name__=='__main__':main()
