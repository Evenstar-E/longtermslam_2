"""Post-seal exact-ray visibility versus the inherited raw depth-span stencil gate."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.run_material_oriented_surface_control_v10 import configuration
from scripts.run_material_surface_patch_registration_v9 import setup
from inspection_decisions.textured_material_dataset_v1 import cast,transform,inverse
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=configuration();root=Path(cfg['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));truth['poses']=np.array(truth['poses']);truth['geometry']=[np.array(a) for a in truth['geometry']];truth['occluders']=np.array(truth['occluders']);perm=np.array(truth['permutations']);records={r['index']:r for r in json.loads((scene/'frames.json').read_text(encoding='utf-8'))};_,frames,ids,_,_,_,views,banks,_=setup(cfg,'rich');labels={};summary=[];details=[];offsets=np.array([[0,0],[-2,0],[2,0],[0,-2],[0,2]]);design=np.c_[offsets,np.ones(5)];pinv=np.linalg.pinv(design)
    for i,a in enumerate(banks[0]):
        N=len(a['xyz']);q=np.zeros((N,3));source_old=np.zeros(N,bool);source_plane=np.zeros(N,bool);source_exact_slant=np.zeros(N,bool)
        for fi in np.unique(a['frame']):
            index=np.flatnonzero(a['frame']==fi);p=a['uv'][index];f=frames[0][int(fi)];rid=int(ids[0][fi])
            if rid not in labels:
                with np.load(scene/records[rid]['labels']) as packed:labels[rid]=dict(material_xyz=packed['material_xyz'],instance=packed['instance'])
            q[index]=labels[rid]['material_xyz'][p[:,1],p[:,0]];assert (labels[rid]['instance'][p[:,1],p[:,0]]==i).all();pix=p[:,None]+offsets;H,W=f['depth'].shape;inside=(pix[...,0]>=0)&(pix[...,0]<W)&(pix[...,1]>=0)&(pix[...,1]<H);ok=inside.all(1);xx=np.clip(pix[...,0],0,W-1);yy=np.clip(pix[...,1],0,H-1);z=f['depth'][yy,xx];ok&=(f['depth_valid'][yy,xx]&(f['anonymous_slot'][yy,xx]==i)).all(1);span=np.ptp(z,axis=1);source_old[index]=ok&(span<=.025);inv=1/np.maximum(z,1e-8);fitted=(inv@pinv.T)@design.T;depth_res=np.max(abs(1/np.maximum(fitted,1e-8)-z),axis=1);source_plane[index]=ok&(depth_res<=.002)
            if ok.any():
                kk=np.flatnonzero(ok);hit=cast(truth,0,f['C'],pix[kk].reshape(-1,2),f['K']);owner=hit['instance'].reshape(-1,5);primitive=hit['primitive'].reshape(-1,5);face=hit['face'].reshape(-1,5);same=(owner==i).all(1)&(primitive==primitive[:,:1]).all(1)&(face==face[:,:1]).all(1);source_exact_slant[index[kk]]=same&(span[kk]>.025)
        k=int(perm[2,i]);world=transform(q,truth['poses'][2,i]);physical=np.zeros(N,bool);physical2=np.zeros(N,bool);quad=np.zeros(N,bool);quad2=np.zeros(N,bool);slant=np.zeros(N,bool);slant2=np.zeros(N,bool);selected={id(f) for f in views[2][k]}
        for f in frames[2]:
            p=inverse(world,f['C']);z=p[:,2];uv=p[:,:2]/np.maximum(z[:,None],1e-8)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2];H,W=f['depth'].shape;inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<W-1)&(uv[:,1]>=0)&(uv[:,1]<H-1);ii=np.flatnonzero(inside)
            if not len(ii):continue
            hit=cast(truth,2,f['C'],uv[ii],f['K']);seen=(hit['instance']==i)&(abs(hit['depth_exact']-z[ii])<2e-5);ii=ii[seen];primitive=hit['primitive'][seen];face=hit['face'][seen]
            if not len(ii):continue
            physical[ii]=True
            if id(f) in selected:physical2[ii]=True
            xy=np.floor(uv[ii]).astype(int);corner=xy[:,None]+[[0,0],[1,0],[0,1],[1,1]];xx,yy=corner[...,0],corner[...,1];ok=(f['depth_valid'][yy,xx]&(f['anonymous_slot'][yy,xx]==k)).all(1);quad[ii[ok]]=True
            if id(f) in selected:quad2[ii[ok]]=True
            if ok.any():
                jj=np.flatnonzero(ok);chit=cast(truth,2,f['C'],corner[jj].reshape(-1,2),f['K']);same=(chit['instance'].reshape(-1,4)==i).all(1)&(chit['primitive'].reshape(-1,4)==primitive[jj,None]).all(1)&(chit['face'].reshape(-1,4)==face[jj,None]).all(1);bad=np.ptp(f['depth'][yy[jj],xx[jj]],axis=1)>=.025;slant[ii[jj[same&bad]]]=True
                if id(f) in selected:slant2[ii[jj[same&bad]]]=True
        row=dict(asset=i,rear=int(k%3==0),source_queries=N,exact_visible_all=int(physical.sum()),exact_visible_top2=int(physical2.sum()),exact_visible_source_old_eligible=int((physical&source_old).sum()),exact_visible_source_inverse_plane_eligible=int((physical&source_plane).sum()),exact_visible_source_same_plane_depthspan_rejected=int((physical&source_exact_slant).sum()),exact_visible_target_quad_owned_all=int(quad.sum()),exact_visible_target_quad_owned_top2=int(quad2.sum()),exact_same_plane_target_depthspan_rejected_all=int(slant.sum()),exact_same_plane_target_depthspan_rejected_top2=int(slant2.sum()));summary.append(row)
        for h in range(N):details.append(dict(asset=i,query=h,source_frame=int(a['frame'][h]),source_u=int(a['uv'][h,0]),source_v=int(a['uv'][h,1]),exact_visible_all=int(physical[h]),exact_visible_top2=int(physical2[h]),source_old_eligible=int(source_old[h]),source_inverse_plane_eligible=int(source_plane[h]),source_exact_same_plane_span_rejected=int(source_exact_slant[h]),target_quad_owned=int(quad[h]),target_exact_same_plane_span_rejected=int(slant[h])))
    write_csv(root/'dense_stencil_visibility.csv',summary);write_csv(root/'dense_stencil_queries.csv',details);back=[r for r in summary if r['rear']];aggregate={key:sum(r[key] for r in back) for key in back[0] if key not in ['asset','rear']};u=np.array([-2,0,2.]);depth=1/(.5+.005*u);save(root/'dense_stencil_summary.json',dict(rear_totals=aggregate,analytic_plane_counterexample=dict(inverse_depth_affine=True,depth_span_m=float(np.ptp(depth)),old25mm_gate_accepts=bool(np.ptp(depth)<=.025),inverse_depth_second_difference=float(depth[0]**-1+depth[2]**-1-2/depth[1])),scope='POST-SEAL exact raycaster/GT geometry diagnosis of actual512 historical source pixels per object. Geometry identical across four appearances, evaluated once on rich. No new inference or new dataset; no claim all physically visible points are resolvable at native pixels. Inverse-plane 2mm check is a diagnostic, not a tuned predictive method.'));print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
