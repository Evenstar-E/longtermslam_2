"""Plot completed native outcomes and all four posthoc residual diagnoses."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('results/rescan_author_v1')
a=json.loads((root/'inference_long_scene_a_v2/scores.json').read_text(encoding='utf-8'))['scenes'][0]
c=json.loads((root/'inference_memory_port/scores.json').read_text(encoding='utf-8'))['scenes'][0]
diagnostic=json.loads((root/'inference_long_scene_a_v2/failure_diagnostic.json').read_text(encoding='utf-8'))['rows']
fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
bottom=np.zeros(2)
for field,label,color in [('correct','Correct identity','#009E73'),('wrong_archive_identity','Wrong identity','#D55E00'),('mixed_or_unresolved','Mixed / unresolved','#999999')]:
    values=np.array([a[field],c[field]]);ax[0].barh([0,1],values,left=bottom,color=color,label=label);bottom+=values
ax[0].set(yticks=[0,1],yticklabels=['scene_a (n=20)','scene_c (n=14)'],xlabel='Persistent instances',xlim=(0,21),title='A. Completed author first updates')
ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=8)
x=np.arange(4)
for i,r in enumerate(diagnostic):
    aa=r['best_correct_identity_proposal_fit']['cost'];bb=r['final_true_id_template_fit_to_actual_instance']['cost']
    ax[1].plot([i,i],[aa,bb],color='#999999',lw=1)
    ax[1].scatter(i,aa,color='#0072B2',marker='o',s=48,label='Best retained correct-ID candidate' if i==0 else None)
    ax[1].scatter(i,bb,color='#D55E00',marker='x',s=55,label='Final true-ID template at actual target' if i==0 else None)
ax[1].set(xticks=x,xticklabels=[str(r['instance']) for r in diagnostic],xlabel='All failing scene_a instance IDs',ylabel='Shared surface cost (0–1)',ylim=(-.03,1.12),title="B. Candidate coverage vs final placement")
ax[1].legend(loc='lower right',fontsize=7)
fig.suptitle('Native Rescan memory port; no current masks/classes. Identity metric: >=80% points.\nPanel B is GT-conditioned diagnosis, not author energy or motion error.',fontsize=11)
fig.savefig(root/'author_comparison.png',dpi=180);fig.savefig(root/'author_comparison.svg')
