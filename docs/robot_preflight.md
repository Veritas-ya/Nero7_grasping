# 实机复测步骤

使用 PR #1 的 codex/fix-control-and-vision-latency 分支。main 尚未合并这些修复。
保留现场标定、已验证的感知参数和空带平面文件；构建后核对实际加载的文件。

## 获取、构建和重启

```bash
cd ~/Nero7_grasping
git fetch origin
git switch codex/fix-control-and-vision-latency
git pull --ff-only
git rev-parse --short HEAD

source /opt/ros/humble/setup.bash
source ~/agx_arm_ws/install/setup.bash
source ~/nero_ws/install/setup.bash
colcon build --packages-select nero7_msgs nero7_perception nero7_control nero7_bringup
source install/setup.bash
```

退出旧相机、旧检测器和旧控制器后再启动这一版。每个新终端都应 source 对应环境。
不要把 CI 用的消息测试替身生成或安装到机器人工作区。

## 第一阶段：相机、测速、ID，仅观察

以下命令启动相机、相机 TF、检测和显示，不启动机械臂驱动或抓取控制器：

```bash
ros2 launch nero7_bringup perception_preview.launch.py \
  start_camera:=true publish_camera_tf:=true detection_mode:=legacy_depth
```

legacy_depth 是默认对照模式。若现场已验证 depth_plane / rgb_depth，则沿用该模式、
plane_mode、plane_file 和 ROI 参数；旧外参对应的平面不能用于新外参。
roi_px 应只覆盖传送带；2026-10-12 静态记录使用 [300,350,900,450]，仅在相机位置、
分辨率和物体位置仍对应时沿用。

另开终端：

```bash
ros2 topic hz /perception/tracks
ros2 topic hz /perception/debug_image
ros2 topic echo /perception/diagnostics --once
ros2 param get /block_detector belt_file
ros2 param get /block_detector calibration_file
```

先放一个物块，在完整 ROI 内匀速通过，再测试多物块和短时遮挡：
- 同一个块应持续保持 ID；超过 0.5 秒未观测会重新分配 ID，这是预期行为。
- 收到独立观测后，速度才从 WAITING 变为数值；观察 qualification_reason、
  accepted_recent 和 velocity_std_m_s。丢帧或旧图像不能补成新的测速样本。
- debug 最高默认 10 Hz；数值低于这个上限不自动表示失败。关注实际处理帧率、
  process_ms、capture_age_ms、last_error 和 rejected_frames 是否持续恶化。
- 模型给出的速度是物块实际速度。用已知距离和独立计时核对其精度，
  不以传送带电机设定值代替真值。

若检测/注册/TF 校验报错或 ID 持续变化，先保留诊断输出和启动参数，
不要把异常测速用于运动抓取。

## 第二阶段：静态预览和单次夹取

保持带停止；相机、TF、检测器运行。单独启动机械臂驱动（若已运行则不重复启动）：

```bash
ros2 launch nero7_bringup arm_only.launch.py
```

先只计算计划：

```bash
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=static_once -p execution_enabled:=false
```

确认 /grasp/state 能到 STATIC_READY，计划中的抓取点、方向、宽度、TCP 和接近姿态
与实际物块对应。参数 calibration_file / urdf_path 如与现场目录不同，应显式覆盖。
配置路径、注册深度或模型错误时不应发送运动命令。

退出预览控制器，再执行一个完整的静态循环：

```bash
ros2 run nero7_control grasp_controller --ros-args \
  -p mode:=static_once -p execution_enabled:=true \
  -p static_infer_timeout:=60.0 -p result_max_age:=0.5
```

所有正式模式默认执行关闭；dry_run=true 始终阻止运动。
执行途中可关闭开关并观察是否进入 ABORT/停止：

```bash
ros2 param set /grasp_controller execution_enabled false
```

结合夹爪反馈和抬升图像判断是否真的夹住。仅状态 DONE 不代替物理结果确认。

## 运动抓取

静态和测速均通过后，再按 README 的 conveyor_once 流程进行空区非接触时序标定、
预览和一次运动抓取。新版本使用确定性的夹爪姿态对齐；若旧时序记录的姿态不兼容，
控制器会要求重新标定，不能绕过该检查。workspace.x_min=-0.50 只是已有试验记录
中的配置；不能据此扩大其他区域或推断所有路径可达。

conveyor 连续模式还要求有效的 7 元素 bin_pre_joints 和 6 元素 bin_drop_pose。
默认这两个配置为空，因此会拒绝执行；不应直接把一次静态验证当作连续抓取验证。
连续模式复位仅允许 HALT 且最新反馈确认停止、故障已清除；单次模式 ABORT 后须退出重启。
缺少 home/enable 服务会直接报错，不再阻塞控制循环等待。

## 验证范围

自动检查覆盖 Python 编译、纯逻辑回归，以及真实 ROS Humble 节点初始化和消息序列化。
ROS 检查明确使用测试版 AGX 消息和网络替身，没有连接真实 CAN、相机或 GPU 模型。
真实驱动字段/频率、GPU 模型、现场外参和测速精度仍以这一轮现场验证为准。
