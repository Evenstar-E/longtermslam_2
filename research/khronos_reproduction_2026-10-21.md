# Khronos ROS1 官方样例复现记录

**完成状态：21个包构建通过，完整官方bag回放至末尾，结束服务与进程正常退出，地图可重新加载；33项证据审计通过。** 仅为官方样例运行复现，没有论文精度指标或真实巡检效果结论。

## 最终运行与地图验收

[运行记录](../results/khronos_reproduction_v1/run_001/result.json)：`complete=true`，末次clock=91.4341秒，与bag结束时间一致；数据节点正常结束，结束／保存服务返回0，roslaunch返回0，实验日志带`Experiment Finished Cleanly`。回放墙钟256.641秒，含保存与清理总计279.096秒。目标0.5倍回放的实际耗时受本机调度／I/O等影响，不将此作为实时性能测量。

保存`run_001/algorithm/final.4dmap`共541,122,476字节，SHA-256 `ea4ab0f6856e2ed897dce715cb9f6264dce321f120f83ffe8ae72e5f72fdb084`。通过独立C++读取器调用官方公开加载API再次读取：[地图内容](../results/khronos_reproduction_v1/run_001/map_content.json)含4个时序状态，覆盖4.204107999至91.054109999秒；最终查询状态有190个非网格图节点、83条边、69,700个网格顶点和117,142个面。仅证明非空且可读取，不证明对象关联或变化判断正确。

处理计时记录显示active window有863项，frontend／backend各176项，时间覆盖输入末端。官方配置输入间隔0.099秒，20Hz原图不会逐帧全部进入算法；时序输出也经过节流。不能把完整回放等同于1,745帧全部处理，额外丢帧数尚未单独审核。

[33项通过的审计](../results/khronos_reproduction_v1/audit_v2.json)核对构建、结束时间、进程／服务返回、bag与地图哈希、固定源码、地图内容与处理时间覆盖；[产物及源文件清单](../results/khronos_reproduction_v1/manifest.json)记录事后哈希。首次审计在Windows读取WSL日志`latest`符号链接时中断，证据检查已写入`audit.json`；修正清单遍历以记录不可访问链接后，`audit_v2.json`完整完成。真实日志文件仍被收录。

运行保留重复TF时间戳警告。官方ROS参数导出使用shell，含`&`路径导出失败；使用独立argv-safe命令保存为`run_001/supervisor_rosparams.yaml`。原日志的commit为空（编译副本未带`.git`），实际固定版本另行核验。另有负的`extract_object`计时值，因此本轮不作逐模块延迟结论。所有这些问题均保留披露，没有修改原始日志。

## 范围

目标是验证已有强时序系统能在本机运行，为后续公平的真实序列任务比较建立基础。本记录针对官方旧版ROS1实现与官方TESSE apartment合成样例，不是论文数字复现，不代表OpenLORIS适配完成。

官方样例使用真值位姿与真值语义。这些特权输入必须明确披露，不能与只获得RGB-D图像的两帧PnP／3D配准表混为一个排行榜。最新主线是另行保存的ROS2源码，本轮不宣称运行了最新主线。

## 已验证的构建与输入

- 官方ROS1根版本：`4e484c4879a76a37e4bf6304b5c58132dc11b279`。
- 17项直接依赖引用已固定；实际构建使用10个核心依赖和单独的`semantic_inference_msgs`消息包。没有启用神经语义推理或VIO。
- 独立WSL工作区：`/home/tong/uav_khronos_ros1_ws_20261021`；Ubuntu20.04、ROSNoetic，Release编译，2个编译作业、1个包并行。
- [build_007](../results/khronos_reproduction_v1/build_007_result.json)：21个所需包全部成功。评估包`khronos_eval`未构建，不能据此声称完成官方精度评测。
- [实际环境](../results/khronos_reproduction_v1/environment.json)：完整dpkg版本、编译器、catkin配置及5个构建过程中拉取的Git依赖版本。Windows侧36个包通过依赖兼容检查，独立冻结记录另存。
- 官方apartment bag：2,975,059,732字节，SHA-256 `7fe54d69808acfb91aab9b72ac5e458c50e6d3c02baacb76d3877dc087653f97`；[下载记录](../data/khronos_official_v1/acquisition.json)。完整文件本地哈希已记录，但未获得作者发布的对照哈希。
- bag共99,069条消息，记录时间4.204107999至91.4341秒，跨度87.229992001秒；RGB／深度各1,745帧。完整主题表在运行目录`bag_info.json`。

## 适配与失败记录

所有原始固定版本源码保留。仅独立编译副本的`khronos_ros/package.xml`增加漏声明的`semantic_inference_msgs`构建依赖，补丁前后哈希与消息文件哈希见[适配记录](../results/khronos_reproduction_v1/message_dependency_adaptation.json)。没有修改建图算法。

关闭未使用的Python绑定、测试、文档等构建目标；安装catkin、GTSAM、glog、json、yaml-cpp和RViz依赖。关闭显示不会消除原库编译时对RViz头文件的依赖。

使用官方launch的单独副本：选择apartment数据、模拟时间、暂停启动、0.5倍回放和无GUI。算法参数保持官方值。[launch变更](../results/khronos_reproduction_v1/launch_adaptation.json)。构造相机模型先等待CameraInfo，早于结束服务注册；监督器在暂停期间仅重发bag中第一条原始CameraInfo作为锁存标定，不注入图像或新位姿。

历史失败日志全部保存：路径`&`在WSL传参问题改用`--exec`；WSL`/tmp`不可持久，迁至用户home；缺yaml-cpp、RViz、可选pybind11和漏声明消息包逐项处理。`build_001`至`build_006`不代表成功，最终构建证据是`build_007`。样例最终执行验收单独记录，不用编译通过替代运行通过。

## 复现命令

现有工作区增量构建与新运行目录：

```powershell
$env:PYTHONUTF8='1'
& '.\.venv\Scripts\python.exe' -m scripts.build_khronos_wsl --label build_new
& '.\.venv\Scripts\python.exe' -m scripts.launch_khronos_example --run-name run_new
```

首次新建工作区需读取准备脚本并使用`--prepare`，随后补齐记录中的消息包和声明补丁；脚本拒绝覆盖现有目录。当前脚本针对本机路径和WSL发行版，不是通用安装器。下载工具的新增依赖版本见`results/khronos_reproduction_v1/windows_requirements_freeze.txt`。
