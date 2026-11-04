"""Post-seal core-mechanism and actual wrong-quantification diagnostics."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import inverse,transform,ray_box
from scripts.completion_catalog_scene_v1 import observed_slots
from scipy.spatial import cKDTree
from inspection_decisions.first_return_material_surface_v1 import material_envelope
from inspection_decisions.first_return_material_surface_v1 import project
from inspection_decisions.ray_conditioned_plane_polytope_v1 import intersection_bounds


def physical_planes(patch,T0,boxes,cfg):
    C=patch['source_camera'];o=inverse(C[:3,3][None],T0)[0]
    u,v=patch['pixel'];f=cfg['focal_px'];cx=(cfg['width']-1)/2;cy=(cfg['height']-1)/2
    corners=np.array([[(u+dx-cx)/f,(v+dy-cy)/f,1.] for dx in [-1,1] for dy in [-1,1]])
    output=[]
    for box_index,(lo,hi) in enumerate(boxes):
        for axis in range(3):
            for bound in [lo[axis],hi[axis]]:
                offset=bound-o[axis]
                if abs(offset)<1e-12:continue
                normal=(C[:3,:3].T@T0[:3,:3][:,axis])/offset
                coeff=np.array([normal[0]/f,normal[1]/f,normal[2]+normal[0]/f*(u-cx)+normal[1]/f*(v-cy)])
                residual=patch['A']@coeff-patch['b'];compatible=bool(np.max(residual)<=1e-10)
                den=corners@normal
                if np.any(den<=0):continue
                canonical=inverse(transform(corners/den[:,None],C),T0)
                contained=bool(np.all((canonical>=lo-1e-7)&(canonical<=hi+1e-7)))
                output.append({'box':box_index,'axis':axis,'bound':float(bound),'compatible_with_measured_polytope':compatible,
                               'entire_quad_inside_one_true_box':contained,'max_inverse_depth_constraint_violation':float(np.max(residual))})
    # Count distinct true planes on the NINE actual source pixel rays.
    yy,xx=np.mgrid[v-1:v+2,u-1:u+2];rc=np.c_[(xx.ravel()-cx)/f,(yy.ravel()-cy)/f,np.ones(9)]
    rays=rc@C[:3,:3].T@T0[:3,:3];depth=np.min([ray_box(o,rays,lo,hi) for lo,hi in boxes],axis=0);points=o+depth[:,None]*rays
    identities=[]
    for point in points:
        candidates=[]
        for lo,hi in boxes:
            if np.all(point>=lo-1e-7) and np.all(point<=hi+1e-7):
                for axis in range(3):
                    for bound in [lo[axis],hi[axis]]:
                        if abs(point[axis]-bound)<1e-7:candidates.append((axis,round(float(bound),9)))
        identities.append(candidates)
    common=set(identities[0])
    for item in identities[1:]:common&=set(item)
    return {'generator_plane_candidates':output,'all_9_source_returns_share_one_true_plane':bool(common),'source_return_planes':identities,
            'common_source_planes':[list(v) for v in sorted(common)],
            'compatible_continuous_generator_plane_exists':any(p['compatible_with_measured_polytope'] and p['entire_quad_inside_one_true_box'] for p in output)}


def false_plane_hit_witnesses(patch,T0,Ts,frames,boxes,common,cfg):
    """Does the true plane itself hit nonexistent surface between source pixels?"""
    output=[];C=patch['source_camera'];source_o=inverse(C[:3,3][None],T0)[0]
    canonical_from_source=np.linalg.inv(T0)@C
    for axis,bound in common:
        normal=(C[:3,:3].T@T0[:3,:3][:,axis])/(bound-source_o[axis])
        for t,T in enumerate(Ts):
            M=T@np.linalg.inv(T0)@C;world=transform(patch['corner_points'],M)
            for v,frame in enumerate(frames[t]):
                _,uv=project(world,frame['C'],cfg);low=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);high=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[cfg['width']-1,cfg['height']-1])
                if (low>high).any():continue
                yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
                rc=np.c_[(xx.ravel()-(cfg['width']-1)/2)/cfg['focal_px'],(yy.ravel()-(cfg['height']-1)/2)/cfg['focal_px'],np.ones(xx.size)]
                relative=np.linalg.inv(M)@frame['C'];o=relative[:3,3];r=rc@relative[:3,:3].T
                support=intersection_bounds(patch,o,r)['valid'];depth=(1-normal@o)/(r@normal)
                bad=support&(depth<frame['depth'][yy.ravel(),xx.ravel()]-cfg['depth_tolerance_m'])
                for j in np.flatnonzero(bad):
                    point=transform((o+depth[j]*r[j])[None],canonical_from_source)[0]
                    inside=any(np.all(point>=lo-1e-7) and np.all(point<=hi+1e-7) for lo,hi in boxes)
                    output.append({'epoch':t+1,'view':v,'target_pixel':[int(xx.ravel()[j]),int(yy.ravel()[j])],
                        'false_depth_gap_m':float(frame['depth'][yy.ravel()[j],xx.ravel()[j]]-depth[j]),'canonical_hit':point.tolist(),
                        'inside_true_solid':inside,'true_plane_axis':int(axis),'true_plane_offset':float(bound)})
    return output


def main():
    root=Path('results/shared_plane_visibility_mainline_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);seal=read(root/'prediction_seal.json')
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    scores=read(root/'scores.json');true_diag=read(root/'true_branch_diagnosis.json');failures=[];quant=[];mechanism=[]
    for run in read(root/'runs.json'):
        dest=root/run['scene'];src=base/run['scene'];pred=read(dest/'predictions.json');records=read(dest/'records.json');masks=np.load(dest/'feasible_masks.npz')
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (dest/'history_patches.pkl').open('rb') as f:patches=pickle.load(f)
        geometry=geometry_for(run['seed'],cfg['objects'],run['geometry'])
        mechanism.append({'scene':run['scene'],'active_joint_patch_checks':run['active_joint_patch_checks'],'joint_only_rejections':run['joint_only_rejections'],
                          'nominal_vs_independent_added_pairs':sum(r['independent_pairs']-r['nominal_pairs'] for r in records),
                          'independent_pairs':sum(r['independent_pairs'] for r in records),'shared_pairs':sum(r['shared_pairs'] for r in records),
                          'queried_rays':sum(r['queried_rays'] for r in records),'supported_rays':sum(r['supported_rays'] for r in records),
                          'effective_rows':sum(r['effective_rows'] for r in records)})
        current_frames=[]
        for t in [1,2]:
            ff=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:ff.append({k:z[k] for k in ['rgb','depth','C']})
            current_frames.append(ff)
        groups=np.arange(cfg['objects'])%4;slots=inp['T0'][:,:3,3];history_frames=[]
        for path in sorted(src.glob('frame_0_*.npz')):
            with np.load(path) as z:history_frames.append({k:z[k] for k in ['rgb','depth','C']})
        historical_slots=[observed_slots(f,slots,groups,cfg) for f in history_frames]
        current_slots=[[observed_slots(f,slots,groups,cfg) for f in ff] for ff in current_frames]
        for d in [v for v in true_diag if v['scene']==run['scene'] and not v['retained']['independent_period_planes']]:
            i=d['asset'];bad=sorted(set(j for vv in d['true_unary_failures'].values() for j in vv))
            for j in bad:
                physical=physical_planes(patches[i][j],inp['T0'][i],geometry[i],cfg)
                hits=false_plane_hit_witnesses(patches[i][j],inp['T0'][i],truth['poses'][:,i],current_frames,geometry[i],physical['common_source_planes'],cfg)
                failures.append({'scene':run['scene'],'asset':i,'patch_index':j,'source_view':patches[i][j]['view'],
                    'source_pixel':patches[i][j]['pixel'],**physical,'false_true_plane_hit_witnesses':hits})
        for row in scores:
            if row['scene']!=run['scene'] or row['method']!='shared_period_plane' or not row['resolved'] or not row['both_ids'] or row['identity_and_material_5cm']:continue
            i=row['asset'];bb=inp['branches'][i];correct=next(v['correct_pair'] for v in true_diag if v['scene']==run['scene'] and v['asset']==i)
            selected=next(v for v in pred['shared_period_plane']['selection'] if v[0]==i);_,k,l=selected
            mask=masks[f'shared_period_plane_{i}'].copy()
            idmask=np.array([b['slot']==truth['slot1'][i] for b in bb[0]])[:,None]&np.array([b['slot']==truth['slot2'][i] for b in bb[1]])[None,:]
            remaining=mask&idmask
            env=material_envelope(bb,inp['anchors'][i],inp['historical_rays'][i],remaining,cfg)
            selectedcost=bb[0][k]['cost']+bb[1][l]['cost'];truecost=bb[0][correct[0]]['cost']+bb[1][correct[1]]['cost']
            cloud=np.concatenate([inverse(a[i]['points'],inp['T0'][i]) for a in historical_slots]);tree=cKDTree(cloud);decomposition=[]
            for t,chosen in enumerate([k,l]):
                slot=bb[t][chosen]['slot'];points=np.concatenate([a[slot]['points'] for a in current_slots[t]])
                dt=tree.query(inverse(points,bb[t][correct[t]]['T']),workers=1)[0]
                ds=tree.query(inverse(points,bb[t][chosen]['T']),workers=1)[0]
                lt=np.minimum(dt/.08,1)**2;ls=np.minimum(ds/.08,1)**2;saturated=dt>=.08
                assert abs(float(np.mean(lt))-bb[t][correct[t]]['cost'])<1e-12
                assert abs(float(np.mean(ls))-bb[t][chosen]['cost'])<1e-12
                decomposition.append({'epoch':t+1,'points':len(points),'true_loss_saturated_point_fraction':float(np.mean(saturated)),
                    'selected_minus_true_total':float(np.mean(ls-lt)),
                    'selected_minus_true_from_true_saturated':float(np.sum((ls-lt)[saturated])/len(points)),
                    'selected_minus_true_from_true_unsaturated':float(np.sum((ls-lt)[~saturated])/len(points))})
            quant.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'physical_error_m':row['physical_anchor_error_m'],
                'true_pair':correct,'selected_pair':[k,l],'true_codes':[bb[t][correct[t]]['code'] for t in [0,1]],
                'selected_codes':[bb[0][k]['code'],bb[1][l]['code']],'true_pair_retained':bool(mask[tuple(correct)]),
                'same_identity_feasible_pose_pairs':int(remaining.sum()),'fixed_true_identity_local_material_envelope':env,
                'selected_NN_cost':selectedcost,'true_NN_cost':truecost,'selected_minus_true_NN_cost':selectedcost-truecost,
                'NN_saturation_decomposition':decomposition,
                'classification_limit':'Saturated means >=0.08m from sampled historical cloud at evaluator true pose; not a physical unseen-material or visibility certificate.'})
    summary={'joint_only_rejections':sum(v['joint_only_rejections'] for v in mechanism),'active_joint_patch_checks':sum(v['active_joint_patch_checks'] for v in mechanism),
        'independent_pairs':sum(v['independent_pairs'] for v in mechanism),'nominal_vs_independent_added_pairs':sum(v['nominal_vs_independent_added_pairs'] for v in mechanism),
        'bad_true_patches':len(failures),'bad_true_patches_with_certifiable_generator_plane':sum(f['compatible_continuous_generator_plane_exists'] for f in failures),
        'bad_true_patches_source_9_returns_nonplanar':sum(not f['all_9_source_returns_share_one_true_plane'] for f in failures),
        'false_true_plane_hits':sum(len(f['false_true_plane_hit_witnesses']) for f in failures),
        'false_true_plane_hits_outside_true_solid':sum(not w['inside_true_solid'] for f in failures for w in f['false_true_plane_hit_witnesses']),
        'right_identity_wrong_material_cases':len(quant),'all_true_pairs_retained_in_wrong_material_cases':all(q['true_pair_retained'] for q in quant),
        'wrong_material_cases_selected_NN_cost_not_above_true':sum(q['selected_minus_true_NN_cost']<=1e-10 for q in quant)}
    summary['wrong_material_cases_saturated_group_favors_selected']=sum(sum(d['selected_minus_true_from_true_saturated'] for d in q['NN_saturation_decomposition'])<-1e-10 for q in quant)
    summary['wrong_material_cases_unsaturated_group_favors_true']=sum(sum(d['selected_minus_true_from_true_unsaturated'] for d in q['NN_saturation_decomposition'])>1e-10 for q in quant)
    save(root/'mechanism_diagnosis.json',{'GT_assisted_for_true_failure_and_quantification':True,'updates_predictions':False,'summary':summary,
        'scene_mechanism':mechanism,'bad_true_patches':failures,'correct_identity_wrong_material':quant,'script_sha256':sha(Path(__file__))})
    print(summary,flush=True)


if __name__=='__main__':main()
