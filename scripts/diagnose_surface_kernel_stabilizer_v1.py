"""No-GT group-symmetry certificate on the actual 192 fitted dictionaries.

Only the new SURFACE COMPONENT is invariant. The voxel component and full RGB-D
can break the symmetry; this is not a claim of task unobservability.
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from inspection_decisions.surface_measure_first_return_v1 import ray_data,local_data,surface_terms


def main():
    root=Path('results/surface_measure_first_return_v1');limits=read(root/'shard_0/manifest.json')['motion_limits'];rows=[];checks=[]
    for ordinal,run in enumerate(read('results/first_return_gamma_material_v5/runs.json')):
        tag=run['scene'];src=root/f'shard_{ordinal%3}'/tag;cfg=read(src/'scene_config.json')
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        for i in range(24):
            record=read(src/f'material_{i:02d}.json');model=record['model'];F=len(model['center']);row=dict(scene=tag,asset=i,history_views=run['history_views'],surface_kernels=F,certificate='not_single_radial_kernel' if F!=1 else 'surface_component_SO2_invariance')
            if F==1:
                c=np.array(model['center'][0]);n=np.array(model['normal'][0]);radius=float(np.linalg.norm((inp['anchors'][i]-c)-n*(n@(inp['anchors'][i]-c))));center_radius=float(np.linalg.norm(c-n*(n@c)));phi=min(limits[0],2*np.arcsin(min(1.,limits[1]/max(2*center_radius,1e-15))))
                Rplus=Rotation.from_rotvec(n*phi).as_matrix();Rminus=Rotation.from_rotvec(-n*phi).as_matrix();up=c-Rplus@c;um=c-Rminus@c
                error=float(np.linalg.norm((Rplus@inp['anchors'][i]+up)-(Rminus@inp['anchors'][i]+um)))
                assert np.linalg.norm(up)<=limits[1]+1e-10 and np.linalg.norm(um)<=limits[1]+1e-10
                row.update(feasible_spin_rad=phi,anchor_axis_radius_m=radius,single_pose_anchor_separation_m=error,
                           two_period_delta_diameter_m=2*error,surface_only_minimax_error_bound_m=error,
                           possible_delta_difference_over_5cm=bool(2*error>.05),surface_only_minimax_bound_over_5cm=bool(error>.05),
                           history_surface_hit_fraction=float((np.load(src/f'posterior_{i:02d}.npz')['a'][model['V']:]-np.array(model['surface_mass'])).sum()/max(record['history']['history_hits'],1)))
                if i==0:
                    for key,shape in [('center',(-1,3)),('normal',(-1,3)),('precision',(-1,3,3))]:model[key]=np.array(model[key]).reshape(shape)
                    frames,_,_=load_frames(Path(cfg['base_root']),tag,inp['T0'][:,:3,3],cfg);d=ray_data(frames[1],cfg);center=inp['T0'][i,:3,3];Tp=np.eye(4);Tm=np.eye(4);Tp[:3,:3]=Rplus;Tm[:3,:3]=Rminus;Tp[:3,3]=center+up;Tm[:3,3]=center+um
                    Kp,Ap=surface_terms(local_data(d,Tp,i),model,cfg);Km,Am=surface_terms(local_data(d,Tm,i),model,cfg);dk=float(np.max(abs(Kp-Km),initial=0));da=float(np.max(abs(Ap-Am),initial=0));assert dk<1e-6 and da<1e-6
                    checks.append(dict(scene=tag,asset=i,max_surface_logK_difference=dk,max_surface_exposure_difference=da,anchor_motion_difference_m=error))
            rows.append(row)
    summary=[]
    for h in [None,1,3]:
        rr=[r for r in rows if h is None or r['history_views']==h];single=[r for r in rr if r['surface_kernels']==1]
        summary.append(dict(history_views=h,n=len(rr),single_radial_surface=len(single),
                            possible_two_period_delta_difference_over_5cm=sum(r['possible_delta_difference_over_5cm'] for r in single),
                            surface_only_minimax_bound_over_5cm=sum(r['surface_only_minimax_bound_over_5cm'] for r in single)))
    result=dict(GT_read=False,NOT_full_model_unobservability=True,NOT_task_accuracy=True,rows=rows,summary=summary,actual_ray_invariance_checks=checks,script_sha256=sha(Path(__file__)))
    save('research/surface_kernel_stabilizer_diagnostic_v1_2026-10-26.json',result)
    lines=['# 连续表面核的任务敏感对称性（真实192字典，无当前真值）','','对单个径向切向 Gaussian 表面核，令 Sφ(x)=c+R_n(φ)(x−c)，方向同步旋转。由于 Q=ε⁻²nnᵀ+ℓ⁻²(I−nnᵀ)，有 k(Sφx,R_nφu)=k(x,u)。因此这个表面分量对围绕自身法向、穿过中心的旋转完全不变，射线曝光也不变。','','令 r=‖(I−nnᵀ)(p−c)‖。单个位姿的锚点分离为 d=2r sinφ；对本任务的两期位移 Δ，交换两期的(+φ,−φ)与(−φ,+φ)得到两个表面观测等价状态，位移分离为 DΔ=2d=4r sinφ。必须区分：DΔ>5cm 表示可能发生超过容差的位移差；只有DΔ>10cm时，由三角不等式才可说任意仅用此表面分量的估计器不能同时保证两状态误差≤5cm。后者的最坏情况误差下界为DΔ/2=d，且仅针对本构造。','','计算同时检查两侧平移 c−R±c 均在原0.16米约束内，旋转在原公共约束内。','','| 历史视角 | 条件数 | 单径向核 | 两期位移分离>5cm | 表面分量最坏误差下界>5cm |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['history_views'] or '全部'} | {r['n']} | {r['single_radial_surface']} | {r['possible_two_period_delta_difference_over_5cm']} | {r['surface_only_minimax_bound_over_5cm']} |")
    lines += ['',f'每场景首资产共{len(checks)}次在真实当前射线上独立核对两侧 K 和曝光相同；最大logK误差{max(x["max_surface_logK_difference"] for x in checks):.3g}，曝光误差{max(x["max_surface_exposure_difference"] for x in checks):.3g}。这些是表示性质检查，不是新增任务成功数。','','**范围必须分清：**混合模型中的未知体格会打破这种对称性，原始RGB-D的轮廓也可能包含旋转信息。因此这不证明完整模型或真实任务不可观测，也不能把所有H3错误归因于该性质。它证明的是：当前新增连续表面分量没有保留这类本可由有限轮廓提供的旋转约束，困难仍交给粗体格处理。','','这属于表示建模问题；增加同类种子或更准确地求同一Gaussian线积分不会消除它。下一项值得比较的是具有各向异性/非凸有限材料范围的连续支持，仍须配同表示独立与普通表面对照；单纯更换椭圆核是标准表示修复，不能自动宣布论文创新。']
    lines += ['', '特别限制：本构造中 H3 的强最坏情况证书只有2项，不能把它直接当成旧22项净差距的解释。应结合全任务身份/量化错误与池内排序，决定这种对称性是否值得作为下一算法的主攻点；不能将漂亮的群论分析替代实际任务收益。']
    Path('research/surface_kernel_stabilizer_diagnostic_v1_2026-10-26.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
