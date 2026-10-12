# Nero7 Gazebo 抓取示例

使用本机 ROS 2 Humble / Gazebo Classic 11，复用正式 `block_detector` 和
`grasp_controller(mode=conveyor_once)`。相机渲染 RGB-D，视觉分割与卡尔曼跟踪
估计速度，GR-ConvNet 给出抓取点、方向和宽度，控制器检查 TCP、逆解、预测
误差与启动时间窗口，然后执行一次夹取、抬升、放回、松爪。

## 启动

```bash
cd ~/Nero7_grasping
source /opt/ros/humble/setup.bash
source ~/agx_arm_ws/install/setup.bash
colcon build
source install/setup.bash
export ROS_DOMAIN_ID=197 ROS_LOCALHOST_ONLY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ros2 launch nero7_sim example.launch.py mode:=demo software_rendering:=true
```

`demo` 先在空带上用实际仿真反馈测量下降与空夹耗时。只有校验完成后才启动
抓取控制器并从上游放入一个物块。默认显示 Gazebo 窗口；已有正常硬件 OpenGL
时可省略 `software_rendering:=true`。无窗口运行加 `gui:=false`，RGB-D 传感器
仍需要可用的 X display，例如本机 `export DISPLAY=:1`。CUDA 不可用时加
示例默认在 CPU 上推理，CUDA 正常时可加 `inference_device:=cuda`。推理
太慢仍会按正式时序门限跳过目标。

示例 `real_time_factor=0.25`，给软件渲染与视觉处理留出 CPU 时间；仿真中
1 秒约对应现实 4 秒。运动预算按仿真时间测量，反馈断流仍按现实时间保护。

需要本机已有 `gazebo_ros`、`gazebo_plugins`、`agx_arm_msgs`、真实机械臂模型
`~/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf`
和控制器配置中指定的 GR-ConvNet 权重。没有复制模型或权重到本包；其他机器
可以通过 `urdf_path:=...` 指定模型，模型中的网格文件也必须存在。

| mode | 行为 |
|---|---|
| preview（默认） | 自动放块，只计算视觉候选和时间窗口，不发送抓取运动命令 |
| calibrate | 空带测量运动时序，保存后停在 `CONVEYOR_CALIB_DONE` |
| pick | 加载同一 `output_dir` 中的有效仿真时序，放块并单次抓取 |
| demo | 自动串联空带校验和单次运动抓取 |
| static | 停带、在抓取线放块，运行正式 `static_once` |

默认输出 `/tmp/nero7_gazebo`，可用 `output_dir:=...` 更改。
`verified_timings.json` 是校验记录，`trial.json` 是试抓事件和预测，
`plan.json` 是候选规划；`scene_files.json` 指向本轮生成的场景和配置。
仿真使用 `/sim/clock` 测量运动时间，记录标记 `time_source=ros_sim`，
实机控制器拒绝把这些仿真耗时当作实机耗时。
`camera_sync.py` 配对同一传感器时间戳的 RGB-D，等仿真时钟前进至少 30 ms
后再发布，避免 Gazebo 相机异步回调比 ROS 时钟提前；保留原图像和时间戳。
仿真图像链路使用 reliable QoS，避免大尺寸深度帧丢包；实机 QoS 配置保持原值。

## 与真机设置的关系

| 项目 | 来源 / 仿真处理 |
|---|---|
| 七轴关节原点、轴、位置限位和外观 | 原 `nero_description` URDF |
| 相机到基座外参、夹爪 TCP、默认抓取姿态 | `nero7_bringup/config/calibration.yaml` |
| 带面高度、方向、原点、抓取线 | `nero7_bringup/config/belt.yaml` |
| 感知与控制算法、状态机、预测和反馈保护 | 正式 `nero7_perception` / `nero7_control` |
| 相机内参及 ROI | 示例 `config/example.yaml`：1640×720、fx=775、ROI=[0,355,1480,475] |
| 带速 | 场景以 0.028 m/s 运动；控制器仅从渲染图像估计速度 |
| 执行器 | 理想位置伺服，速度由示例参数设置，空带实测耗时 |
| 夹持 | TCP 在物体足迹内、插入高度合格、开口闭合到物体投影宽度时创建固定约束；开爪释放 |

项目没有保存真机 CameraInfo 内参，因此仿真内参是示例值。宽视场与上游 ROI
提供足够的规划及接近距离。RGB 和深度使用同一个光学像素网格，生成的标定
副本把 `d2c_offset` 设为 0。`32FC1` 深度米转换为网络使用的毫米；原实机
`16UC1` 数据保持原处理方式。

示例带为恒速、无滑动，滤波过程噪声在示例配置中单独调低。真机感知配置
没有改动。理想伺服示例的时间标准差设为 0.03 s，实机默认仍为 0.15 s；
这是假设的重复性参数，需结合实际运行记录评估。臂体碰撞与电机/CAN 动力学
没有模拟，夹爪使用几何接触约束，
因此仿真完成只证明这条视觉和控制链路可以运行，不能替代实机的摩擦、碰撞、
时序及动态抓取验证。

示例物块朝向 `object_yaw` 和时序测量的夹爪方向 `closing_yaw` 分别配置。
抓取方向仍由网络计算，必须落在对应时序的允许范围内才会执行；更换物块
或方向后应重新执行 `demo` 或 `calibrate`，控制器不会强制套用该方向。

## 观察与验证

所有机械臂控制接口位于 `/sim/nero7/`，启动文件不加载实机驱动。
图像、跟踪、状态及 TF 使用 `/sim/` 话题；Gazebo master 默认端口 11355。
其他终端使用相同的 ROS_DOMAIN_ID 与 ROS_LOCALHOST_ONLY。

抓取任务默认同时打开物块测速图像窗口，每个 ID 单独标注视觉 XY 速度大小
（m/s）；`WAITING` / `LOST` 表示当前速度不可用。`gui:=false` 同时关闭该窗口。
如手动关闭后需要重新查看，可运行下面的 `rqt_image_view` 命令。

```bash
ros2 topic echo /sim/grasp/state
ros2 topic echo /sim/perception/tracks
ros2 run rqt_image_view rqt_image_view /sim/perception/debug_image
# 在 demo 开始前或空带校验期间启动独立观察器：
ros2 run nero7_sim sim_validate --timeout 360 --output /tmp/nero7_gazebo/validation
```

观察器不发送运动指令。它记录状态、视觉速度与 Gazebo 物体实际高度，只有
控制器完成、发生夹持且物体实际抬升超过 70 mm 才输出 `SIMULATION_PASS`。
保存 `validation.json`、抬升时的 `lift.png` 和末帧 `final.png`。
`/sim/scene/ground_truth` 只供观察器诊断，感知与控制器不订阅。

手动重新放块与停带接口如下；单次控制器完成后需重启才能执行下一轮：

```bash
ros2 service call /sim/belt/spawn std_srvs/srv/Trigger '{}'
ros2 service call /sim/belt/set_running std_srvs/srv/SetBool '{data: false}'
ros2 service call /sim/nero7/emergency_stop std_srvs/srv/Empty '{}'
```

程序组成：`scene.py` 生成世界和 URDF 副本；`plugins/scene.cpp` 驱动 Gazebo
关节、带面物块与夹持约束；`driver.py` 适配真机 ROS 命令/反馈接口；
`scenario.py` 管理放块；`launch/example.launch.py` 串联正式节点；
`validate.py` 收集独立验证证据。
