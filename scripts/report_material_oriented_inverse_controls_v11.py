"""Report both full fixed-pool experiments and exact source-support diagnosis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    roots=[Path('results/material_oriented_surface_control_v10'),Path('results/material_inverse_plane_control_v11')];dd=[json.loads((r/'scores.json').read_text(encoding='utf-8')) for r in roots];methods=['raw_points','raw_patch','normal_points','normal_patch'];variants=['flat','rich','repeated','changed'];labels=['Point projection','Patch projection','Point + normals','Patch + normals'];cn=['逐点投影','小块投影','逐点＋法向','小块＋法向'];colors=['#E69F00','#009E73','#0072B2','#D55E00'];hatches=['','//','..','xx'];table=[];figure_rows=[]
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True);fig.subplots_adjust(left=.06,right=.99,bottom=.2,top=.76,wspace=.1)
        for stage,(root,d,ax) in enumerate(zip(roots,dd,axes)):
            for c,m in enumerate(methods):
                vals=[next(r['qualified'] for r in d['summary'] if r['variant']==v and r['method']==m) for v in variants];bars=ax.bar(np.arange(4)+(c-1.5)*.19,vals,.18,color=colors[c],hatch=hatches[c],edgecolor='black',linewidth=.5,label=labels[c]);table.append('|'+['原表面块 V10','逆深度修复 V11'][stage]+'|'+cn[c]+'|'+'|'.join(map(str,vals))+'|');figure_rows.extend(dict(stage=stage,method=m,variant=v,qualified=n,total=24) for v,n in zip(variants,vals))
                for bar,n in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,n+.35,str(n),ha='center',fontsize=8)
            ax.set(xticks=np.arange(4),xticklabels=['Flat','Rich','Repeated','Repainted'],ylim=(0,24),yticks=np.arange(0,25,4),title=['V10: standard normal compatibility','V11: slant-invariant plane support'][stage]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylabel('Qualified tasks / 24');fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=2,frameon=False);fig.text(.5,.1,'One outcome-informed TRAIN scene; objects / appearances correlated. No independent confirmation or CI.',ha='center');fig.text(.5,.045,'Identical V9 pose pool and finite identity states; V11 changes patch locations / target interpolation, not poses.',ha='center',fontsize=9);fig.savefig(roots[1]/'controlled_task_comparison.png',dpi=180,facecolor='white');fig.savefig(roots[1]/'controlled_task_comparison.pdf',facecolor='white');plt.close(fig)
    save(roots[1]/'figure_provenance.json',dict(source_hashes={str(r/'scores.json'):sha(r/'scores.json') for r in roots},rows=figure_rows,dimensions_inches=[12,5.4],dpi=180,uncertainty='None; one TRAIN, correlated variants/objects',meaning='All4methods/all4appearances/bothstages; same0-24axes; actual success only. Internal research figure, journal requirements unspecified.'))
    dense=json.loads((roots[0]/'dense_stencil_summary.json').read_text(encoding='utf-8'))['rear_totals'];support=[json.loads((r/'view_support_summary.json').read_text(encoding='utf-8'))['summary'][1] for r in roots];runtime=[json.loads((r/'prediction_seal.json').read_text(encoding='utf-8'))['elapsed_s'] for r in roots]
    text='''# V10–V11：有向表面控制、提取缺陷定位与逆深度修复

2026-10-27。已完成两阶段全量推断，每阶段24对象×4外观×4方法=384主要行，合计768。一个已见TRAIN，无新训练、无新候选位姿、无独立确认。旧V9及以前源/结果均保留。

## 实际任务结果

每格分母24，仍要求两期身份正确且三个时间段固定历史材料锚点P95位移误差均≤5cm。

|阶段|方法|无纹理|丰富纹理|重复纹理|局部重涂|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

![完整受控比较](../results/material_inverse_plane_control_v11/controlled_task_comparison.png)

八个背面对象在两个阶段、所有方法、全部外观中仍均0项合格。修复没有转成任务成功，不能用支持数改善替代实际收益。V10的未加法向两列与V9预测逐字段相同；V11的四列全部在新表面块/新投影场上重算，不能冒称旧控制不变。

## 为什么先修法向，又转向提取条件

V10只增加标准法向同半球兼容：源五像素几何估计相机定向法向，目标法向由实际深度场导数估计；对应要求$(R n_i)^\\top n_j\\ge0$。退化法向保持原处理。相反朝向的深度命中不再提供材料对应支持，但不凭空变成自由空间。模型不加位姿，也不改变V9身份状态图。

法向过滤没有追回背面，rich身份从24降至19；剩下三个身份正确的背面解全部旋转误差超过90°。这说明排除一类反面证据后，系统可换成错身份或大角度解释。标准过滤不是完整解法，也不算创新。[Open3D早已有法向对应检查器](https://open3d.org/docs/latest/python_api/open3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal.html)。

随后检查所有现有视图：对旧48块，top2和全部视图的背面法向兼容命中均只有4像素，均无对象有≥3个支持块。因此不能先认定“换两张图就行”。再对原始512中心查询做精确射线核验，发现真正的提取瓶颈。

## 同一组查询的精确证据

私有几何和精确射线仅用于封存后的诊断。背面8对象共4096个真实历史源像素中心：

'''+f"- 当前所有视图中精确可见：{dense['exact_visible_all']}；其中原top2已经可见{dense['exact_visible_top2']}。\n- 这些可见中心通过旧源小块条件：{dense['exact_visible_source_old_eligible']}；通过逆深度平面条件：{dense['exact_visible_source_inverse_plane_eligible']}。\n- {dense['exact_visible_source_same_plane_depthspan_rejected']}个位置的五像素由精确射线确认属于同一平面，却被旧25mm深度跨度条件拒绝。\n- 目标侧另有{dense['exact_same_plane_target_depthspan_rejected_all']}个同平面四像素足迹因深度跨度而被拒（原top2中{dense['exact_same_plane_target_depthspan_rejected_top2']}）。\n"+'''
上述统计来自同4096查询，几何对四外观相同，因此只在rich执行一次。物理可见不代表原生像素足迹可解析，更不等于自动对应可识别；这里没有据此宣称原始数据完全可观测。

## 修复的数学依据与实现

旧条件把小块深度跨度≤25mm当作连续表面证据，但针孔相机下，一个倾斜平面也能有很大深度跨度。平面应满足

$$1/D(u,v)=a u+b v+c.$$

一个精确平面的示例深度跨度约80mm，会被旧条件拒绝，逆深度却严格共面。V11的源端用相同512个实际中心候选，拟合五像素逆深度平面，以深度重建残差≤2mm筛选，再均匀保留48块。目标端用3×3真实深度拟合逆深度平面，连续像素位置在该平面上求深度与有向法向；颜色仍由四个真实邻像素插值。跨槽位或非平面残差超限保持未知，不生成隐藏纹理。

2mm是本实验固定残差门限，数据公开配置的深度扰动界也是±2mm，但该拟合门限不是严格概率置信界。源/目标场修复是标准平面几何的正确性工作，不宣称原创。目标从2×2增加到3×3像素读取，虽然源仍是48五像素块，也不能声称计算量完全相同。

核心源码：[法向控制](../inspection_decisions/material_oriented_surface_control_v10.py)、[逆深度修复](../inspection_decisions/material_inverse_plane_control_v11.py)。两阶段均保留完整V9候选和同一有限身份图，V8耦合端点仍成对使用；非负能量下界仅跳过不可能胜出的计算，不任意裁剪姿态。

## 修复是否真的恢复了材料观测

'''+f"在真实位姿下，背面top2的法向兼容RGB-D命中从{support[0]['rear_top2_hit_pixels']}提升到{support[1]['rear_top2_hit_pixels']}个采样像素；有≥3支持块的对象从{support[0]['rear_top2_has3_supported_patches']}/8变为{support[1]['rear_top2_has3_supported_patches']}/8。用所有视图则为{support[1]['rear_all_hit_pixels']}个命中、{support[1]['rear_all_has3_supported_patches']}/8对象。\n"+'''
这是新旧不同48块查询在真实位姿下的表示诊断，不是推断成功数，也不是同查询漏斗。严格同查询反事实是上一节4096中心的0→175。两种证据不能混用。

## 对主线与创新的判断

链路推进：两阶段全对象实验完成，找到并实际修复了会丢掉倾斜共同材料支持的观测条件。证据不支持“当前输入根本没有共同材料”，也不支持把此前表面块失败全部归于算法概念无效。

实际收益：本轮没有超过V9最强对照，无纹理逐点从11降至修复后的8；加法向版本从10降至9；其他三外观合格数不变。所有背面仍失败。保留这些负结果，不选最好列拼系统。当前新增工作均属标准物理/几何修复，论文创新性尚未建立。

当前仍固定在旧候选池，rich/changed只有1/8背面有合格候选，另外7项无法仅靠评分追回。下一步优先让恢复的倾斜材料支持**生成连续位姿**，包含从C1–C2对应传播到H–C2的普通控制，比较普通逆深度配准与共享持续材料模型。仍评价全24×4，要求依次达到新增合格候选、转成实际成功、超越同表示普通控制。不能继续只加过滤门限或只在已有候选上换评分。

旧GN的近似中心化和局部求解不足仍可能存在；本轮没有证明连续模型已被最优求解。独立场景、现代强3D完整方法和真实跨期确认仍待后续执行。

## 验证与复现

'''+f"- V10运行{runtime[0]:.2f}秒、评分核对{dd[0]['checks']}项；V11运行{runtime[1]:.2f}秒、评分核对{dd[1]['checks']}项。两次预测先封存再评分。\n- 选中目标重建最大差异分别{dd[0]['max_selected_objective_error']:.3g}、{dd[1]['max_selected_objective_error']:.3g}；完整枚举抽查分别{sum(r['pairs'] for r in dd[0]['brute_force'])}、{sum(r['pairs'] for r in dd[1]['brute_force'])}个候选对，最大目标差异均小于1e-8。\n"+'''
- 11项法向/退化/遮挡/旧核一致性检查、7项倾斜平面/阶跃边缘检查通过；精确平面深度误差约机器精度。源补丁像素和保存的输入另做来源核验。
- 没有重新训练网络，没有生成新相机图像，没有用GT生成推断候选或选择视图。精确光线、真实位姿和私有面编号只进入封存后的诊断。

```powershell
.venv-real/Scripts/python.exe -m scripts.run_material_oriented_surface_control_v10
.venv-real/Scripts/python.exe -m scripts.score_material_oriented_surface_control_v10
.venv-real/Scripts/python.exe -m scripts.run_material_inverse_plane_control_v11
.venv-real/Scripts/python.exe -m scripts.score_material_inverse_plane_control_v11
```

封存目录拒绝覆盖，复现须复制配置到新根。完整表格包括两阶段task_metrics.csv、same_pose_counterfactual.csv、view_support.csv，以及V10的dense_stencil_queries.csv精确查询明细。所有旧证据保持不变。一个已见TRAIN，外观、对象和像素相关，不作独立样本显著性或泛化推断。
'''
    Path('research/material_oriented_inverse_results_v11_2026-10-27.md').write_text(text,encoding='utf-8');print('V10/V11 REPORT AND FIGURE WRITTEN')

if __name__=='__main__':main()
