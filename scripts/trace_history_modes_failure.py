"""Outcome-selected diagnostic trace; not a representative efficacy sample."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.history_modes_v6 import infer,clusters

def main():
    base=Path('results/history_modes_v1');plan=json.loads((base/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());found=None;scanned=0
    for level in [2,3]:
        for cell in [4,5,6,7,12,13,14,15]:
            if found:break
            for k in range(8):
                sid=cell*16+k;path=base/'cases'/f'{level}_{sid:04d}.npz';a=np.load(path);obs={key:a[key] for key in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};p=infer(obs,cfg['algorithm'],sampler,plan['modes'],'v6_modes',plan['seed_base']+sid+400000);scanned+=1
                modes=[(j,g) for j in range(24) for g in clusters(obs['history_positions'][:,j],plan['modes']['cluster_radius_m'])];selected_modes=p['solutions'][p['selected_index']]['mode']
                for i in range(len(p['assignment'])):
                    if not a['truth_changed'][i] or p['changed'][i] or p['assignment'][i]!=a['truth_assignment'][i]:continue
                    j,g=modes[int(selected_modes[i])];provenance=a['truth_provenance'][g,j]
                    if np.all(provenance==j):continue
                    found=dict(stress=plan['stresses'][level]['name'],base_scene=sid,current_index=i,historical_id=j,mode_frames=g,mode_physical_ids=provenance.tolist(),estimated_reference=p['reference_points'][i].tolist(),clean_reference=a['truth_reference'][j].tolist(),current_position_in_map=(obs['current_positions'][i]@p['T'][:3,:3].T+p['T'][:3,3]).tolist(),selected_reference_residual_m=float(p['residual_m'][i]),clean_reference_residual_m=float(np.linalg.norm(obs['current_positions'][i]@p['T'][:3,:3].T+p['T'][:3,3]-a['truth_reference'][j])),reported_changed=False,evaluator_changed=True,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    break
                if found:break
        if found:break
    out=base/'failure_trace.json';out.write_text(json.dumps(dict(scanned_cases=scanned,diagnostic_selection='First matching missed-change/wrong-reference case, not representative sample',case=found,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    a=np.load(base/'cases'/'2_0000.npz');p=a['history_positions'][:,0];groups=clusters(p,plan['modes']['cluster_radius_m']);fig,ax=plt.subplots(figsize=(6,7),layout='constrained')
    for group in groups:
        center=p[group].mean(axis=0);ax.scatter(p[group,0],p[group,1],s=65);physical=np.unique(a['truth_provenance'][group,0]).tolist();text='Frames '+','.join(map(str,group))+'\nPhysical IDs '+','.join(map(str,physical));offset=(30,15) if 0 in group else (30,-35);ax.annotate(text,center[:2],xytext=offset,textcoords='offset points',arrowprops=dict(arrowstyle='->'),fontsize=10)
    mean=p.mean(axis=0);ax.scatter(mean[0],mean[1],marker='x',s=140,c='red');ax.annotate('Mean: not an observed component',mean[:2],xytext=(-100,-45),textcoords='offset points',arrowprops=dict(arrowstyle='->'),fontsize=9);ax.scatter(p[0,0],p[0,1],marker='*',s=180,c='black');ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Fixed scene0 / nominal track0\nTwo real positions are mixed into one false mean',xlim=(-.9,1.2),ylim=(-.3,2.5));ax.set_aspect('equal');ax.grid(alpha=.15);fig.savefig(base/'memory_example_v2.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(scanned=scanned,case=found),indent=2))

if __name__=='__main__':main()
