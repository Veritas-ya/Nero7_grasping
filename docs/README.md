# Nero7 + Gemini 335L 传送带连续抓取系统

松灵 Nero7（7 关节 + AGX 二指夹爪）+ Orbbec Gemini 335L（eye-to-hand 固定俯视）
在连续运动传送带上抓取物块放入料箱。抓取位姿由 GR-ConvNet（免模型 RGB-D
抓取热力图）生成；带速由纯视觉跟踪估计；动态抓取采用"抓取闸门"策略
（预测到达时间 + 提前量触发下降 + 相机确认）。

控制逻辑与感知延迟、测速和 ID 稳定性修复及现场复测方法见
[logic_and_tracking_fixes.md](logic_and_tracking_fixes.md)。正式模式均默认执行关闭；
显式 execution_enabled=true 且 dry_run=false 才会发送运动与夹爪命令。

## 包结构

| 包 | 内容 |
|---|---|
| nero7_msgs | BlockTrack / BlockTrackArray / GraspCandidate |
| nero7_perception | block_detector（带面 RANSAC 分割 + 跟踪 + 测速）+ 标定脚本 |
| nero7_control | grasp_controller（FSM + 臂控制 + GR-ConvNet 异步推理）+ 工具脚本 |
| nero7_bringup | launch（camera_only / arm_only / bringup / dry_run）+ config |
| nero7_sim | Gazebo RGB-D 传送带示例，复用正式视觉与单次抓取控制器 |

Gazebo 示例的启动、真机参数对应关系及验证方法见
[nero7_sim/README.md](../src/nero7_sim/README.md)。

视觉感知改进的[代码审查](perception_review.md)、[实现与实验报告](perception_improvement_report.md)
和[只读采集/验证方法](perception_validation.md)已提供。新模式包括基座系固定高度、
三维平面和 RGB/深度融合；默认保留 legacy_depth 对照，新模式需先校核空带标定文件。

当前现场使用的原尺寸测速画面、稳定 ID 配置及启动命令见
[camera_speed_preview.md](camera_speed_preview.md)。

## 依赖（已就绪的机器环境）

- ROS2 Humble + orbbec_camera（apt）+ cv_bridge/tf2
- ~/agx_arm_ws（agx_arm_ctrl 驱动、agx_arm_msgs，已编译）
- ~/nero_ws（nero_description URDF，已编译）
- torch 2.12+cu130（RTX 5070）、opencv、skimage
- GR-ConvNet 预训练权重：~/grasp/robotic-grasping/trained-models/

## 构建

```bash
cd ~/Nero7_grasping
source /opt/ros/humble/setup.bash
source ~/agx_arm_ws/install/setup.bash   # 提供 agx_arm_msgs
colcon build
source install/setup.bash
```

## M0 环境验证（先插好相机、CAN 线，臂上电）

```bash
# 0) 激活 CAN（1Mbps）
bash ~/agx_arm_ws/src/agx_arm_ros/scripts/can_activate.sh can0 1000000

# 1) 相机验证
ros2 launch nero7_bringup camera_only.launch.py
# 另开终端：
ros2 topic hz /camera/color/image_raw
ros2 topic echo /camera/color/camera_info | head    # 记录内参
ros2 topic echo /camera/depth/camera_info | head    # 深度内参（官方标定）
# ⚠️ 确认对齐深度话题名（depth_registration:=true 时），默认
#    /camera/aligned_depth_to_color/image_raw；若无此话题，把
#    perception.yaml / controller.yaml 的 alignment_mode 改为 points
#    （并确认 depth_topic 改为 /camera/depth/image_raw）

# 2) 机械臂验证（含 TF）
ros2 launch nero7_bringup arm_only.launch.py
ros2 run nero7_control arm_test --ros-args -p mode:=basic     # enable→home→move_p
ros2 run nero7_control arm_test --ros-args -p mode:=gripper   # 夹爪开合
ros2 run tf2_tools view_frames    # 检查 TF 树：world→base_link→link1..7 无断链
# 夹爪 teaching 标定（首次使用必须，SDK 直连方式；此时不能同时跑 agx_arm_ctrl 节点）：
ros2 run nero7_control gripper_calib --ros-args -p backend:=sdk
```

## M1 标定（触碰法：固定相机 + 固定标定板）

眼在手外标定用触碰法（标定板固定在带面上，不装末端，避免晃动）：

```bash
# 0) 前置：arm_only（CAN 驱动 + TF）与相机节点都在运行；标定板平放固定在
#    相机视野内的带面/台面上（姿态随意，板完整可见）
# 1) TCP 标定（夹爪捏住探针；没有探针则跳过，tcp 按 [0,0,0] 用法兰中心，
#    但建议打印一个尖头探针，精度高很多）
ros2 run nero7_perception touch_calib --ros-args -p phase:=tcp
#    把探针尖触碰桌面任一固定尖点，换 4 个不同姿态，每次 Enter 记录，最后按 s

# 2) 板角点触碰求解
ros2 run nero7_perception touch_calib --ros-args -p phase:=board
#    x/X y/Y z/Z 微动（步长 [ / ] 调节），按 Enter 依次记录 4 个角点+中心，
#    最后按 s 求解，结果写入 config/calibration.yaml

# 3) 指哪打哪验证（传送带上放 4 个静止物块）
ros2 run nero7_perception verify_calib
# 通过标准：所有点位偏差 < 10mm

# 4) 传送带方向标定（带运行时放一个块）
ros2 run nero7_perception belt_calib

# 备用：棋盘格装法兰的传统 AX=XB 路线（handeye_collect / handeye_solve）
# 仍然保留，装法稳定时可用
```

## M2 静态抓取（带停）

正式控制器支持 `static_once`：在相机、机械臂驱动、相机到基座 TF 和
`block_detector` 已运行时，选择静止目标，用 GR-ConvNet 的 RGB-D 结果计算
抓取点、夹爪方向和宽度，再结合标定 TCP 与 URDF 检查逆解和接近路径。
该模式执行一次夹取、抬升、放回、松爪、退回，然后停在 `STATIC_DONE`。
它不调用临时抓取脚本，也不使用几何估计替代网络候选。

```bash
# 预览：仅计算候选，不运动；execution_enabled 默认 false
ros2 run nero7_control grasp_controller --ros-args -p mode:=static_once
# 单次实机执行（避免同时启动另一个控制器）
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=static_once -p execution_enabled:=true \
  -p static_infer_timeout:=60.0 -p result_max_age:=0.5
```

当前相机启用深度对齐时，实际深度话题为 `/camera/depth/image_raw`。
静态模式默认 ROI 为 `[180,355,1060,475]`，需与物体所在带面一致。
外参与 TCP 从 `nero7_bringup/config/calibration.yaml` 加载。
若要按深度检测的物块中心夹取，可显式加
`-p static_center_source:=track_centroid`：夹取点采用已确认目标的 RGB-D 质心，
方向和闭合宽度仍须来自有效 GR-ConvNet 候选；默认 `network` 使用网络选点。
`infer_target_radius_px` 默认 60，可按物块在图像中的大小限制候选搜索半径。
候选记录保存到 `/tmp/nero7_formal_static_plan.json`；该文件每次规划覆盖。
目标移动或候选过期时重新推理；无合格候选、不可达或反馈异常时中止。

2026-10-09 实机记录：GR-ConvNet 分数 0.253，预测宽度 24.13 mm，
闭合目标 20.51 mm，夹后反馈 23.98 mm；完成抬升、放回并退回。
抬升图像 `/tmp/nero7_formal_lift.png` 确认物体被抓起，末态 `STATIC_DONE`，
机械臂 `motion_status=0`、`err_status=0`。这是一轮静止物体验证，
不代表传送带连续抓取模式已完成实机验证。

下面是原有整套系统启动方式（默认连续传送带模式）：

```bash
ros2 launch nero7_bringup bringup.launch.py execution_enabled:=false
# 连续执行需先填写有效 bin_pre_joints/bin_drop_pose，再显式 execution_enabled:=true
# 抓取任务会自动打开 /perception/debug_image 图像窗口；状态见 /grasp/state
# 模型 A/B 对比（物块摆好后运行，Ctrl+C 出统计，图存 /tmp/grasp_benchmark）
ros2 run nero7_control benchmark_models
# 状态：ros2 topic echo /grasp/state；卡死后复位：ros2 service call /grasp/reset std_srvs/srv/SetBool "{data: true}"
```

## M3 视觉卡尔曼跟踪与单次运动抓取

`block_detector` 用对齐的 RGB-D 分割带面上的物体，通过相机外参把各帧
位置转换到 `base_link`。每个目标维护 `[x,y,vx,vy]` 的常速度卡尔曼状态，
持续更新位置、速度及协方差。传送带速度来自物体的视觉轨迹，无需输入电机
设定速度。速度沿 `belt.yaml` 中的带方向投影，单位 m/s。

默认要求最近 1.2 秒内至少 8 次有效观测，跨度至少 0.3 秒，速度标准差
不超过 0.010 m/s，最后观测不超过 0.15 秒。异常位置通过创新门限拒绝，
重复时间戳不增加观测数。短时丢失仅作预测，保持最后观测时间戳并增大
协方差；丢失超过新鲜度门限后 `speed_valid=false`。无效速度发布 0，
静止目标也可能为 0，消费者必须检查 `speed_valid`。

启动抓取任务时默认自动打开 `rqt_image_view`，覆盖 `grasp_controller` 的
`conveyor`、`conveyor_once`、`static_once` 模式及旧的 `static_grasp` 命令。
每个物块旁显示独立的 ID 和实时速度大小，例如
`ID 7 speed=0.028 m/s`。速度大小为视觉卡尔曼 XY 速度的模，与控制消息中
沿带方向的有符号 `speed` 不同。未完成测速显示 `WAITING`，目标丢失显示
`LOST`，均不显示有效数值；空带显示速度不可用。控制器退出时关闭它打开的
窗口，手动关闭窗口不会停止抓取。

可加 `-p visualization_enabled:=false` 关闭自动弹窗；自定义图像话题时设置
`-p visualization_topic:=对应话题`，并与检测器的 `debug_topic` 保持一致。
无桌面显示环境时跳过弹窗，图像仍发布到 `/perception/debug_image`。
Gazebo 示例自动使用 `/sim/perception/debug_image`，`gui:=false` 同时关闭测速窗口。

`BlockTrack` 区分观测时间 `stamp` 和状态时间 `motion_stamp`，并发布原始
观测点、滤波状态、4×4 协方差和过程噪声。`BlockTrackArray.header` 记录
检测图像时间，空目标列表也能检查是否过期。这些消息接口已经改变，需要
重新编译并重启所有发布者和订阅者，不能混用旧进程。

```bash
colcon build --packages-select nero7_msgs nero7_perception nero7_control nero7_bringup
source install/setup.bash
```

正式控制器的 `conveyor_once` 在上游完成多帧网络候选和逆解检查，保持同一个
推理目标；随后等待目标进入计算出来的启动区域。启动预算包含启用及开爪
（默认 1.5 秒）、接近（默认 2 秒）、实测下降和闭合到接触的时间，另留
0.8 秒启动余量。规划还额外预留 2.5 秒。抓取线保持固定；启动区域位于
上游，其距离随视觉估计的速度和运动耗时计算。

启动前检查整个预测时段的协方差、带向和横向误差，以及夹爪开口能否容纳
预测位置范围。连续 3 个不同观测时间戳合格后才允许启动。到达预抓位置后
继续跟踪，目标进入下降时间窗口且预测合格才下降。下降期间仅容许不超过
0.15 秒的短时目标丢失。目标太晚、反馈故障或预测范围越界时跳过或急停。
机械臂起始位姿发生变化时，原接近路径失效，必须重新规划。

```bash
# 相机、TF、驱动和 block_detector 已运行；避免同时运行其他抓取控制器。
# 默认 execution_enabled=false：只规划、记录预测，不发送运动或夹爪命令。
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=conveyor_once -p workspace.x_min:=-0.50
```

当前门线及 TCP 补偿需要这一轮命令的 `workspace.x_min=-0.50`；控制器仍会
逐点验证工作区、逆解和接近路径。该设置不表示其他点位都可到达。

实机执行需要近期、同一 TCP、带几何、夹爪姿态及驱动速度下的空中下降/
空夹时序记录。普通抓取模式不会自动执行标定；记录缺失、未验证或过期时
进入 `CONVEYOR_NEED_CALIBRATION`，不运动。标定必须单独指定
`conveyor_calibration_only=true`，并清空视觉带面 ROI 及抓取区；控制器检查
图像和检测输出新鲜，连续空区至少 0.6 秒。标定时若进入物体或视觉失效，
立即中止。标定下降低点在带面上方约 44 mm，行程为 78 mm。

```bash
# 清空物体后，显式执行一次非接触时序标定；使用独立文件保存结果。
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=conveyor_once -p workspace.x_min:=-0.50 \
  -p conveyor_calibration_only:=true -p execution_enabled:=true \
  -p conveyor_timing_file:=/tmp/nero7_conveyor_verified_timings.json

# 标定结束并退出该进程，再启动单次运动抓取。
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=conveyor_once -p workspace.x_min:=-0.50 \
  -p execution_enabled:=true \
  -p conveyor_load_timings_file:=/tmp/nero7_conveyor_verified_timings.json
```

标定记录默认有效 3600 秒，启动时再次检查；改变驱动速度或夹爪方向后需要
重新测量。规划输出 `/tmp/nero7_conveyor_plan.json`，运行事件、预测及结果
输出 `/tmp/nero7_conveyor_timings.json`。加载的标定文件应与运行输出分开，
避免被试抓日志覆盖。预览会使用估计耗时，预览通过不能替代实测时序。

当前单次试抓限于视觉带速 0.008–0.040 m/s。候选夹爪方向与标定姿态差异
须不超过 0.15 rad，抓取点距标定点不超过 50 mm，下降行程须保持 78 mm。
其他方向可以先查看预览计划的 `yaw`（弧度），在标定和抓取命令中同时设置
`-p conveyor_calibration_yaw:=对应值`，重新测量时序；不能用旧时序直接套用。

2026-10-09 的旧运动试抓在下降阶段因预测偏移中止，未完成夹取。
当前卡尔曼及分阶段启动流程已加入软件测试，尚未通过新的实机运动抓取验证。

测速表示物体沿带面的运动速度，前提是物体随带运动；发生滑动时测到的是
物体实际速度。测速需要可跟踪物体，空带不能用这条物体跟踪链路测量。
`belt.yaml` 的方向与抓取线是几何参数，感知和控制器应加载同一个文件。
方向可通过物体运动轨迹标定，速度在运行时持续由视觉更新。
完成 TCP、运动预测和时序验证前，不应把测速通过当作动态抓取通过。

以下是旧的 `mode=conveyor` 连续 FSM 的提前量工具，未接入上面的单次时序
标定流程；`k_lead` 不控制 `conveyor_once` 的启动区域。

```bash
# 带速固定（如 0.05 m/s），先扫描提前量 k_lead（每档喂 5 个块）：
ros2 run nero7_control lead_calib --ros-args -p k_leads:="[0.0, 0.05, 0.1, 0.15]" -p trials:=5
# 该工具自动把最优 k_lead 写回 controller.yaml；也可运行时微调：
ros2 param set /grasp_controller k_lead 0.08
```

## 干跑模式

```bash
ros2 launch nero7_bringup dry_run.launch.py   # 只读预览，不发送运动/夹爪命令
```

`execution_enabled=false` 或 `dry_run=true` 在所有正式模式下都阻止运动和夹爪指令；
执行途中关闭执行开关会请求停止。首次执行前请核对现场标定与示教配置。

## 安全要点

- 旧连续 FSM 的失败走 RECOVER，连续 miss ≥3 次进入 HALT；需 `/grasp/reset` 复位
- `conveyor_once` 的视觉预测或反馈故障急停并终止；不会自动重试或回 home
- 反馈看门狗：/feedback/joint_states 断流 200ms → 停发指令
- 工作区软围栏 + 目标步长 ≤0.35m 校验在控制器内强制
- 紧急情况：`ros2 service call /emergency_stop std_srvs/srv/Empty` 或物理急停

## 关键配置

- perception.yaml：相机话题/QoS/ROI/分割阈值/跟踪与测速参数
- controller.yaml：模型路径/FSM 参数/安全限位/料箱位姿（bin_pre_joints、bin_drop_pose，M2 示教填入）
- calibration.yaml：眼在手外结果（handeye_solve 生成）
- belt.yaml：带方向/原点/gate_s（belt_calib 生成）
