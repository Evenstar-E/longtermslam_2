"""Descriptive development figures; no independence or statistical significance claim."""
import json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    out=Path('results/livingscenes_real_v1'); native=Path('results/livingscenes_native_masks_v1')
    methods=['more_sequential','more_classwise_sequential','more_classwise_hungarian','more_position_hungarian']
    labels=['Published sequential','Sequential + class gate','Hungarian + class gate','Shape + position control']
    summaries=[read(out/'scores.json')['summary'],read(native/'scores.json')['summary']]
    data=[]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
        for ax,summary,title,n in zip(axes,summaries,['A  Provided GT masks/classes (3 spaces)','B  Actual Rescan masks/classes (2 spaces)'],[62,34]):
            rows=[next(r for r in summary if r['scene']=='all' and r['method']==m) for m in methods]
            good=np.array([r['correct'] for r in rows]);wrong=np.array([r['wrong'] for r in rows]);other=n-good-wrong
            y=np.arange(4)
            ax.barh(y,good,color='#0072B2',label='Correct')
            ax.barh(y,wrong,left=good,color='#D55E00',hatch='//',label='Wrong identity')
            ax.barh(y,other,left=good+wrong,color='#CCCCCC',hatch='..',label='Unresolved')
            for i,g in enumerate(good):ax.text(g+.25,i,str(g),va='center',fontsize=9)
            ax.set(yticks=y,yticklabels=labels,xlim=(0,n+2),xlabel=f'Persistent objects (n={n})',title=title);ax.invert_yaxis()
            for r in rows:data.append({'contract':title,**r})
            if n==34:
                ax.axvline(30,color='black',linestyle='--',label='Original Rescan: 30/34')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=9)
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)
        fig.suptitle('Pretrained learned matching transfer: separate input contracts; no equal-compute claim',fontsize=12)
        for ext in ['png','pdf']:fig.savefig(out/f'learned_comparison.{ext}',dpi=200)
        plt.close(fig)
        caps=[r for r in read(native/'scores.json')['fixed_mask_caps'] if r['scene']=='scene_a' and r['instance'] in [7,12,17,23]]
        caps=sorted(caps,key=lambda r:r['instance']);fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        values=[100*r['strict_one_group_coverage_cap'] for r in caps]
        ax.bar([str(r['instance']) for r in caps],values,color=['#D55E00' if x<80 else '#0072B2' for x in values],edgecolor='black')
        for i,v in enumerate(values):ax.text(i,v+1,f'{v:.2f}%',ha='center')
        ax.axhline(80,color='black',linestyle='--',label='Identity acceptance threshold: 80%')
        ax.set(ylim=(0,110),ylabel='Maximum point coverage (%)',xlabel='Previously failed physical object ID',title='Fixed native masks + one-to-one association: evaluator upper bound')
        ax.legend(loc='lower right');fig.savefig(out/'fixed_mask_cap.png',dpi=200);fig.savefig(out/'fixed_mask_cap.pdf');plt.close(fig)
    (out/'figure_source_data.json').write_text(json.dumps({'comparison':data,'fixed_mask_caps':caps},indent=2),encoding='utf-8')
    (out/'figure_provenance.json').write_text(json.dumps({'audience':'Provisional internal research review, no publisher specified','aggregation':'Raw object counts in observed development spaces; no resampling, no CI, scenes are not independent replicates of every object','unresolved_A':'unmatched','unresolved_B':'mixed or unresolved point labels','fixed_mask_cap':'Maximum GT object points in one fixed disjoint predicted mask / all GT object points, diagnostic only','not_claimed':['Original MORE2 benchmark ranking','equal runtime','independent confirmation','motion accuracy'],'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'scores.json',native/'scores.json']}},indent=2),encoding='utf-8')

if __name__=='__main__':main()
