"""Raw-point descriptive confidence gap on the past development capture."""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha
from scripts.prepare_real_temporal_v1 import read_ply


def main():
    out=[]
    for stem in ['shared_surface_identity_likelihood_v3','shared_surface_identity_map_control_v3']:
        root=Path('results')/stem
        for method in read(root/'config.json')['methods']:
            n=wrong=0;cs=0.
            for scene in ['scene_a','scene_c']:
                assets=read(root/scene/'assets.json');eligible={r['uid'] for r in assets if r['class'] not in [0,1,2,22] and r['raw_points']>=128}
                truth=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply')['instance_idx'];eq={}
                f=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
                if f.exists():
                    for line in f.read_text(encoding='utf-8').splitlines():
                        if line.strip():
                            a,b=line.split('|');eq[int(a)]={int(x) for x in b.split()}
                p=np.load(root/scene/method/'prediction1.npz');a=p['identity'];c=p['confidence'];use=np.isin(truth,list(eligible))&(a>=0)&(c>=.9);good=np.zeros(len(a),bool)
                for uid in np.unique(truth[use]):good[truth==uid]=np.isin(a[truth==uid],list(eq.get(int(uid),{int(uid)})&eligible))
                n+=int(use.sum());wrong+=int(np.sum(use&~good));cs+=float(c[use].sum())
            out.append({'method':method,'known_id_confidence_ge_point9_n':n,'wrong':wrong,'mean_confidence':cs/n,'accuracy':1-wrong/n})
    save(Path('results/shared_surface_identity_likelihood_v3/confidence_diagnosis.json'),{'rows':out,'source_sha':sha(__file__),'scope':'Exact raw-point source-period descriptive confidence gap on familiar DEV scenes, excluding outlier decisions. No independent calibration guarantee or significance test.'})
    print('CONFIDENCE DIAGNOSIS SAVED')


if __name__=='__main__':main()
