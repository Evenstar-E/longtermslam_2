"""Post-seal evidence of conflating different parallel material faces."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES


def main():
    root=Path('results/material_supported_continuous_alignment_v2');cfg=read(root/'config.json');diagnosis=read(root/'mechanism_diagnosis.json');history=[]
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for run in read(root/'runs.json'):
        with (root/run['scene']/'features.pkl').open('rb') as f:features=pickle.load(f)
        for i,ff in enumerate(features[0]):
            for k in sorted({f['axis'] for f in ff}):
                group=[f for f in ff if f['axis']==k];q=[float(AXES[k]@f['centroid']) for f in group];angles=[float(np.degrees(np.arccos(np.clip(AXES[k]@f['normal'],-1,1)))) for f in group]
                if max(q)-min(q)>.012:history.append({'scene':run['scene'],'asset':i,'axis':k,'views':[f['view'] for f in group],'observed_offsets_m':q,'assigned_reference_mean_m':float(np.mean(q)),
                    'offset_span_m':max(q)-min(q),'normal_errors_deg':angles,'fit_rms_m':[f['rms_m'] for f in group],
                    'clean_parallel_face_alias':max(q)-min(q)>.03 and max(angles)<2 and max(f['rms_m'] for f in group)<.002})
    current=[f for f in diagnosis['true_pose_feature_consistency'] if f['history_reference_available'] and abs(f['offset_minus_history_m'])>.03 and f['normal_error_deg']<2 and f['measured_fit_rms_m']<.002]
    summary={'history_direction_groups_offset_span_gt_12mm':len(history),'clean_historical_parallel_face_alias_groups':sum(h['clean_parallel_face_alias'] for h in history),
        'clean_current_reference_mismatch_gt_30mm':len(current)}
    save(root/'face_alias_diagnosis.json',{'summary':summary,'historical_aliases':history,'clean_current_mismatches':current,'postseal':True,'predictions_changed':False,
        'historical_checks_GT_free':True,'current_checks_use_GT_pose_only_after_seal':True,'source_sha256':sha(Path(__file__))})
    print(summary,flush=True);print('CLEAN_HISTORY',[h for h in history if h['clean_parallel_face_alias']][:4],flush=True)


if __name__=='__main__':main()
