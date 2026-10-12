# 视觉感知代码审查（2026-10-12，修改前）

真实路径是 `src/nero7_perception/nero7_perception/` 下的
`block_detector.py`、`alignment.py`、`conveyor_model.py`、
`block_tracker.py`、`kalman_motion.py`。消息在 `src/nero7_msgs/msg/`；
控制消费者在 `src/nero7_control/nero7_control/`。未发现 AGENTS.md。

## 已确认

1. `BeltPlane.fit()` 每轮无放回随机抽 3 点，最多 100 轮；用最小二乘拟合
   `Z=a*u+b*v+c`，统计 `abs(A@coeffs-Z)<0.006 m` 的内点。最多抽样
   30000 点，最低内点率 0.4。不检查三点秩，不对最终内点重拟合。
   拟合失败会沿用 last_good，未报告模型年龄。旧类和修改前快照均保留。
2. 上述残差是光轴 Z 的差，不是真实三维正交距离。透视下真实三维平面的
   深度通常满足逆深度关系，旧线性像素深度模型只是局部近似。
3. 对齐源按 uint16 将 mm 除以 1000；其他 dtype 直接当米，未核对编码。
   深度零/负值变 NaN；Inf 没有统一清理。Detector 深度尺寸不等时直接 resize。
4. 未校验 RGB、深度、CameraInfo 的 frame_id 和尺寸一致性，未处理/拒绝
   非零畸变。TF 查询用 depth stamp、配置的 optical frame 到 base frame。
   方向和时间用法成立的前提是该深度确实属于配置的彩色光学系。
5. RGB-D 只比较两个最新帧，相差容许 0.1 s；同一深度可被多个彩色帧使用。
   跟踪器会拒绝重复时间戳，但检测、平面拟合和发布仍可能重复。
6. PointsSource 根据 flat pixel 排序后选第一个，实际上没有取最近 Z；
   未使用四舍五入，丢弃亚像素坐标小于零但本可映射到边界的点。
   `pcl_utils` 的 PointField 类型映射错误（FLOAT32=7，FLOAT64=8），
   也未考虑组织点云 row_step 填充和大小端。
   分支还导入了本机 tf2_geometry_msgs 中不存在的 `do_transform_point3`，会在投影前报错。
7. `height` 是旧光轴深度差，`belt_z` 是另一条相机射线在拟合带面上得到的
   base Z。连续 FSM 和旧静态脚本会计算 `belt_z+height`，含义不一致。
   正式 static_once/conveyor_once 主要用网络表面 base Z 和 belt_z 做净空检查。
8. 三维 Detector 不声明、不使用 perception.yaml 中的 HSV 阈值。
   配置中 d2c_offset=[-4,2] 是额外像素平移；控制器网络深度没有同步应用此平移。
   像素平移不能替代深度到彩色的外参配准。
9. BlockTracker / KalmanMotion 用真实、独立时间戳位置更新 [x,y,vx,vy]，
   保留观测/状态时间区别、创新门限、ID、协方差、新鲜度和速度资格判定。
   本次不修改这些文件或消息定义。

## 潜在问题，尚无实测结论

- 带面是否平行于 base XY、TF 旋转/平移误差、深度噪声和现场灰色阈值。
- 当前实际运行驱动是否与磁盘驱动源码一致、所用深度话题是否真正注册到彩色。
  本机 Orbbec 驱动源码在 `~/桌面/OrbbecSDK_ROS2/orbbec_camera/`，代码会在
  depth_registration 开启时使用彩色光学 frame；这不是本次实时消息验证。
  该源码发布 PointCloud2 时将 XYZ 除以 1000，转换为米；仍需确认运行驱动版本。
- 工作区很宽且 ROI 全图，支架/机械臂可能混入主平面；空载 ROI 必须现场选定。
- 倾斜带面上的动态运动预测和抓取时序仍需独立验证；本次不改变 TCP/偏移。

没有找到可确认的真实 RGB-D 数据集。`build/nero7_sim/frame.npz` 是仿真快照，
不能作为实机精度证据。后续合成实验将明确标注；提供只读采集及离线验证工具。
修改前四个感知源码和配置保存在 `docs/perception_baseline/`，跟踪/消息哈希也已记录。
