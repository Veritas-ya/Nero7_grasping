# 感知算法验证与只读使用方法

本页所有预览/采集入口都没有机械臂驱动或抓取控制器。测试期间先停止其他
抓取控制器，避免它们消费预览发布的 tracks。这里不提供执行真实抓取的命令。

## 构建与软件测试

```bash
cd ~/Nero7_grasping
source /opt/ros/humble/setup.bash
source ~/agx_arm_ws/install/setup.bash
colcon build --packages-select nero7_perception nero7_bringup nero7_sim
source install/setup.bash

MPLCONFIGDIR=/tmp/nero7_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/src/nero7_perception:$PWD/src/nero7_control:$PWD/src/nero7_sim:$PYTHONPATH" \
python3 -m pytest -q src/nero7_perception/test src/nero7_control/test src/nero7_sim/test
```

## 重现实测过的合成对照

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run nero7_perception perception_validate \
  --synthetic --config src/nero7_perception/config/perception.yaml \
  --output-dir /tmp/nero7_perception_horizontal

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run nero7_perception perception_validate \
  --synthetic --synthetic-tilt-rad 0.035 --plane-mode ransac3d \
  --config src/nero7_perception/config/perception.yaml \
  --output-dir /tmp/nero7_perception_tilted

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run nero7_perception perception_validate \
  --synthetic --synthetic-stress --config src/nero7_perception/config/perception.yaml \
  --output-dir /tmp/nero7_perception_stress
```

输出包含 metrics.json、带真值的 synthetic.npz、synthetic_annotations.json，
有效参数 effective_params.yaml，以及通过检查的 belt_plane.yaml。
合成数据为 320×240、60 帧（15 帧空带，45 帧
运动物块）、0.8 mm 光轴高斯噪声、1% 深度缺失、0.028 m/s 运动。
stress 额外包含灰色带面斑块、该斑块处 -6 mm 光轴偏差、物块区域 10% 深度离群点。
这些都是数学渲染数据；合成的 plane 文件不得拿到真实相机上使用。

## 采集真实数据（相机与已校核的光学系 TF 已运行）

确认正在使用的深度话题已注册到彩色光学系，且其 CameraInfo 对应注册后的网格。
只开启相机可使用 camera_only.launch.py；该入口不加载机械臂。

```bash
# 空载：确认所选 ROI 内没有物块/机械臂/支架，再标记 capture_empty=true。
ros2 run nero7_perception capture_rgbd --ros-args \
  -p capture_empty:=true -p capture_frames:=30 \
  -p output_file:=/tmp/nero7_empty.npz

# 先停带，放置量过高度/位置的物块，再采集静态样本。
ros2 run nero7_perception capture_rgbd --ros-args \
  -p capture_frames:=60 -p output_file:=/tmp/nero7_static.npz

# 最后仅运行传送带，采集运动样本；仍不启动抓取控制器。
ros2 run nero7_perception capture_rgbd --ros-args \
  -p capture_frames:=90 -p output_file:=/tmp/nero7_moving.npz
```

可设置 color_topic/depth_topic/color_camera_info/depth_camera_info、
camera_optical_frame/base_frame、qos_reliability、alignment_mode 和 use_sim_time。
默认等待最多 30 秒；配对不全时明确失败，不保存不完整的捕获文件。
无注册深度可使用 `-p alignment_mode:=points`，真实投影点云至彩色网格；
稀疏投影的有效深度比例也必须实测，不填补未知深度。

NPZ 使用 BGR uint8 的 color_bgr，单位米的 depth_m，每帧 transforms（光学系
到 base）、stamps（深度时间）、color_stamps、empty 标签，以及 K、D、
distortion_model 和光学/base frame 元数据。色彩、注册深度的尺寸/内参/坐标系不一致
会被拒绝。capture_empty 是操作者标注，不是软件证明该区域清空。

## 真实离线比较与平行假设检查

以下 ROI 示例来自项目已有带面观察范围，必须先按当前图像确认仅覆盖带面；
不同图像尺寸、安装或带面范围应重新选择。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run nero7_perception perception_validate \
  --input /tmp/nero7_empty.npz /tmp/nero7_static.npz /tmp/nero7_moving.npz \
  --config src/nero7_perception/config/perception.yaml \
  --plane-roi 180 355 1060 475 --roi 180 355 1060 475 \
  --plane-mode fixed --output-dir /tmp/nero7_real_validation
```

检查 plane_calibration 中的 normal、tilt_rad、slope_x/slope_y、xy_span_m、
height_by_xy_bins、residual_std_m、residual_p95_m、fixed_height_p95_m、
frame_height_spread_m。固定模型默认要求倾角 ≤0.01 rad、固定高度 P95 残差
≤6 mm、跨帧高度残差中位数范围 ≤4 mm、X/Y 支撑跨度分别 ≥50 mm。
三维模型默认也拒绝倾角超过 0.2 rad 的主平面，避免把墙/支架当带面。

固定模型失败时先检查 TF 和 ROI，确认是真实倾斜才选择 `--plane-mode ransac3d`。
两种模型都会被比较；只有所选模型通过检查才输出当前 belt_plane.yaml。
重新生成结果时旧 plane 文件移到带时间戳的 previous 文件，保留证据并避免误用旧结果。

没有真实目标真值时，定位误差、误检/漏检率、ID 切换等字段为 null；仍输出
空载误报、检测数量、速度资格样本数、拒绝帧及原因和处理耗时。不要把 null 当作 0。
要评估真值，在 JSON 中给每个按深度时间排序后的帧提供目标列表：

```json
{"frames": [[], [{"id": 1, "centroid_base": [-0.4, 0.2, -0.06],
                 "height": 0.04, "velocity_base": [0, -0.028, 0]}]]}
```

帧数必须与输入样本总帧数相等；上例只演示结构。用测量得到的表面中心/高度和
独立测速结果填写，不能用待验证检测器自己的输出当真值。
通过 `--annotations /path/to/annotations.json` 加载；位置用米，速度用 m/s。
评估采用一对一三维中心匹配（距离 ≤40 mm）。应同时检查原始图像并量化该匹配
门限对误报/漏报统计的影响，尤其是多个邻近物块。

## 不发送运动指令的在线预览

离线文件通过检查后，使用真实相机生成的文件：

```bash
ros2 launch nero7_bringup perception_preview.launch.py \
  detection_mode:=rgb_depth plane_mode:=fixed \
  plane_file:=/tmp/nero7_real_validation/belt_plane.yaml \
  roi_px:='[180,355,1060,475]'
```

相机未运行时可加 start_camera:=true。已有 TF 时保持 publish_camera_tf:=false；
若单独发布项目标定矩阵，可显式加 publish_camera_tf:=true，文件必须给出有效的
camera_color_optical_frame -> base_link 刚体变换及 camera_link_to_color_optical。
启动器组合后只发布 base_link -> camera_link，保留相机驱动自身的光学 TF；
不会用单位矩阵代替缺失标定，也不会给光学坐标系增加第二个父节点。
camera_only 默认启用 SDK 彩色去畸变，使 RGB 像素与已注册深度使用一致的内参。
外参更新后，旧平面文件的 TF 上下文不再有效，必须重新采集空带区域标定。

也可以显式执行空带在线标定（只生成感知平面文件，和机械臂时序标定不同）：

```bash
ros2 launch nero7_bringup perception_preview.launch.py \
  empty_belt_calibration:=true plane_mode:=fixed \
  plane_file:=/tmp/nero7_live_belt_plane.yaml \
  plane_roi_px:='[180,355,1060,475]' roi_px:='[180,355,1060,475]'
```

标定阶段不会发布 tracks；成功保存后也不会自动切换到抓取/检测。查看日志和
diagnostics 后结束该进程，再以检测模式重启。失败时检查原因并重新启动标定。

```bash
ros2 topic echo /perception/diagnostics
ros2 topic hz /perception/tracks
ros2 topic echo /perception/tracks
```

在线 processed_fps_window 是最近报告周期实际完成的检测帧率，processed_fps 是
启动以来平均帧率；process_ms 是单帧检测到发布的耗时，包含 TF 查找及跟踪。
offline_detection_fps 只由离线检测耗时计算，不能当作相机、ROS 或整条抓取链路帧率。

实机验收应记录：不同 X/Y 位置的空载高度、重复标定高度变化、静态 XYZ/高度误差、
不同光照与背景的误检/漏检、运动/遮挡时 ID 和速度、相机实发频率、处理/发布帧率、
丢帧和端到端延迟。保存原始数据、独立测量真值和参数后再决定是否切换生产模式。
新平面不回写 belt.yaml、TCP、夹爪偏移或机械臂时序文件；动态执行前另行校核这些几何和时序。
