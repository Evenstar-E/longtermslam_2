from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save


def main():
    root=Path('results/identity_material_support_v1');cfg=read(root/'config.json');rows=[]
    for method in cfg['methods']:
        row={'method':method}
        for prefix in ['','supported_','nn_']:
            rr=[r for r in read(root/(prefix+'scores.json'))['summaries'] if r['period']==2 and r['method']==method]
            row[prefix+'strict']=sum(r['strict_correct'] for r in rr);row[prefix+'equivalent']=sum(r['equivalent_correct'] for r in rr)
        prov=read(root/'provenance_diagnosis.json');cor=[r for r in prov['source_corrections'] if r['method']==method]
        for key in ['equivalent_fixed_voxels','equivalent_damaged_voxels']:row[key]=sum(r[key] for r in cor)
        mass=[]
        for scene in cfg['scenes']:
            assets=read(root/scene/'assets.json');eligible={r['uid'] for r in assets if r['class'] not in [0,1,2,22] and r['raw_points']>=128}
            ss=[r for r in prov['rows'] if r['scene']==scene and r['method']==method and r['target_period']==2 and r['archive_id'] in eligible]
            mass.extend(ss)
        row['source_allowed_mass']=sum(r['allowed_mass'] for r in mass);row['source_disallowed_mass']=sum(r['disallowed_mass'] for r in mass)
        row['source_admitted_voxels']=sum(r['admitted_voxels'] for r in mass);rows.append(row)
    save(root/'summary.json',{'rows':rows,'scope':'Same-start one-sweep development refinement, not independent test; mass uses voxel representative truth times raw counts.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');x=np.arange(3);names=['Warm start','Tie preservation','Material admission']
        ax[0].bar(x-.18,[r['strict'] for r in rows],.34,label='Identity',color='#0072B2')
        ax[0].bar(x+.18,[r['supported_strict'] for r in rows],.34,label='Identity + geometric support',color='#E69F00',hatch='//')
        ax[0].set_ylim(0,37);ax[0].set_ylabel('Successful assets / 37');ax[0].legend(fontsize=8)
        ax[1].bar(x,[r['source_disallowed_mass'] for r in rows],color='#CC79A7',hatch='xx');ax[1].set_ylabel('Disallowed admitted source mass')
        for a in ax:a.set_xticks(x,names,rotation=15)
        fig.suptitle('Identity / material admission separation | Two familiar development spaces')
        fig.supxlabel('Source mass is a voxel-truth proxy; no independent displacement truth.')
        fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
    print(__import__('json').dumps(rows,indent=2))


if __name__=='__main__':main()
