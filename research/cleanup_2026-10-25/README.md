# 安全清理记录（2026-10-25）

已完成：删除 **13,953 个缓存文件、1,753 个空目录**，失败项 **0**。

删除文件的逻辑大小合计 **414.38 MiB**。根据删除前实际硬链接数估算，最后一个链接被删除的载荷约 **205.59 MiB**；这不是磁盘净释放量，未扣除审计记录，也未计入分配簇、目录元数据及系统同时发生的变化。硬链接计数使用 Path.stat 校正，详见计划中的 windows_link_count_method。

| 类别 | 删除文件数 | 文件大小合计 |
|---|---:|---:|
| 有源码的 Python 字节码 | 10,863 | 187.96 MiB |
| uv 索引与解释器缓存 | 149 | 17.24 MiB |
| uv 已核验安装副本的包缓存 | 2,940 | 209.04 MiB |
| Matplotlib 字体索引 | 1 | 0.14 MiB |

## 保留范围

- 数据集全部保留：data 下 2,093 个文件、9,694,322,610 字节。
- 实验载荷全部保留：results 下 44,990 个文件、15,389,378,267 字节。包括 v2/v4/v5 依赖、根目录与分片封存文件、预测、原始观测、失败记录。
- 原始源码、配置、第三方实现、模型权重、两个虚拟环境的安装文件均保留；删除的 uv 包文件均有已核验副本或保存的安装记录。
- `.cache` 中的图实验输入/输出，以及数据目录中的中断下载与分段载荷保留。下载脚本和采集记录仍引用这些路径，本轮不将它们视为可直接删除对象。
- `python-dateutil` 的一个小标记内容哈希命中了历史证据哈希，保守保留该包完整缓存；所有重解析点均保留且未跟随。

## 核验结果

1. 删除前检查 14,691 份文本中的引用，并记录全部保留文件的 SHA256。
2. 实际删除前逐项检查目标位于项目内、路径不经过重解析点、大小与 SHA256 一致。使用 PowerShell 的精确文件删除和空目录删除，无递归整目录清空。
3. 删除后 **80,724 个保留文件的 SHA256 全部一致**，缺失/变化/意外新增均为 0；全部删除目标均已移除，重解析点清单不变。
4. 主实验 v5 证据索引、封存清单及直接 sources/inputs 共 **94 项 SHA256 全部通过**。这不是对历史所有封存文档的重新有效性认证；全量前后比对证明本轮没有改变所保留的实验与数据内容。
5. 两个虚拟环境的 NumPy/SciPy/OpenCV 导入成功，当前两个算法模块导入成功；未重跑研究实验。
6. 清理完成核验后，仅向 `.agent/HANDOFF.md` 追加维护链接。其原始字节完整备份于 [HANDOFF_before_maintenance.md](HANDOFF_before_maintenance.md)，修改前后哈希见 [maintenance_update.json](maintenance_update.json)。原始代码、数据集、实验产物没有后续改动。若重新运行一次性核验脚本，应结合此已登记的文档变更解释 HANDOFF 的差异。

## 重建与审计文件

- Python 字节码在下次正常导入时从保留源码重建；字体索引由 Matplotlib 字体管理器重建。
- 删除的 2,940 个 uv 包文件具有逐文件同哈希重建来源，见 plan.json 的 uv_archive_rebuild_map；原 wheel RECORD 保存在 uv_wheel_records。索引缓存由 uv 正常生成。
- [plan.json](plan.json)：精确目标、SHA256、分类、重建来源、保留原因。
- [deletion_receipt.json](deletion_receipt.json)：实际删除清单、空目录、失败项、执行方式。
- [retained_before.json](retained_before.json)：删除前完整保留文件哈希快照（包含追加维护说明前的 HANDOFF）。
- [verification.json](verification.json)：全量删除后核验及已登记的维护文档变更。
- [primary_evidence_postcheck.json](primary_evidence_postcheck.json)、[runtime_smoke.json](runtime_smoke.json)：封存与环境检查。

原始慢速删除前检查在尚未删除文件时中断；实际执行使用同一计划，并复用已验证父目录的检查结果。目标范围与逐文件哈希校验未减少。
