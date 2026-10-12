# Nero7 视觉感知改进报告（2026-10-12）

完成了代码审查、可切换实现、合成离线实验及软件回归；提供真实 RGB-D 只读采集、
离线比较和在线静态预览入口。没有启动真实机械臂驱动/抓取控制器，也没有发送
机械臂、夹爪或 CAN 运动命令。实机精度、现场误检率和真实处理帧率尚未验证。

## 1. 审查结论及保留基准

完整审查见 [perception_review.md](perception_review.md)。已确认原实现是三点
随机采样 RANSAC，但拟合像素坐标中的线性深度模型；它的光轴残差不是三维
点到平面的距离。深度 resize、PointCloud2 类型/行步长/大小端、最近点投影、
重复深度帧和缺失坐标系检查均已找到具体实现依据。

原 `conveyor_model.py` 的 BeltPlane **没有改动**。修改前的 Detector、alignment、
pcl_utils、conveyor_model 和配置保存于 [perception_baseline](perception_baseline)。
`legacy_depth` 保留原分割、实时拟合/last_good 和光轴 height 语义作对照；
输入的编码、配准和时间校验统一加强，经验像素平移默认停用。
完整历史行为可查看快照，不能把旧 height 当作已修正的基座系垂直高度。

生产配置仍默认 legacy_depth；新模式需要显式选择并加载校核过的空带文件。
只读 perception_preview 默认使用 rgb_depth，不启动抓取控制器。

## 2. 文件修改与目的

| 文件 | 目的 |
|---|---|
| src/nero7_perception/nero7_perception/block_detector.py | 缓冲配对、独立时间戳处理、严格帧/内参检查、平面标定/加载、模式分发、诊断与实际完成帧率；保留发布接口和测速可视化 |
| src/nero7_perception/nero7_perception/alignment.py | 注册深度及 CameraInfo 校验；按消息编码转米；点云按采集时间 TF 投影、最近 Z 缓冲；移除无效导入 |
| src/nero7_perception/nero7_perception/pcl_utils.py | 正确读取 FLOAT32/FLOAT64、组织点云行填充及大小端 |
| src/nero7_perception/nero7_perception/geometry.py | 彩色光学系针孔/畸变模型、有效深度、ROI/尺寸/刚体变换检查、缓存像素射线 |
| src/nero7_perception/nero7_perception/base_plane.py | 固定 +Z 法向量高度估计、真实三维 RANSAC、跨帧/倾角/空间支撑检查、诊断、带上下文的持久化及旧文件归档 |
| src/nero7_perception/nero7_perception/detection.py | 旧分割对照、基座系深度分割、灰色 HSV 分割、RGB+高度约束、稳健内部深度定位 |
| src/nero7_perception/nero7_perception/validation.py | 同一数据/参数下比较七种组合，评估高度/XYZ/误检/漏检/ID/速度/耗时；无真值字段为 null |
| src/nero7_perception/nero7_perception/scripts/capture_rgbd.py | 只读同步 RGB-D/TF 采集，保存可重放的 NPZ |
| src/nero7_perception/nero7_perception/scripts/perception_validate.py | 离线 CLI、可复现合成真值、真实数据注释入口和报告输出 |
| src/nero7_perception/config/perception.yaml | 可切换配置、明确米/rad、HSV 和有效深度阈值 |
| src/nero7_perception/setup.py / package.xml | 注册两个验证入口及 scipy 依赖 |
| src/nero7_perception/test/test_base_perception.py | 新几何、标定、配准、同步、拒绝无效帧、定位、ID、速度及 ROS→控制器集成验证 |
| src/nero7_bringup/launch/perception_preview.launch.py / package.xml | 感知专用预览；可选相机和严格固定 TF；不加载机械臂驱动或控制器 |
| src/nero7_sim/launch/example.launch.py | 同位 Gazebo RGB-D 使用同一相机内参；适配严格校验和仿真诊断话题 |
| docs/README.md及本组报告/结果 | 审查、使用步骤、实际实验数据和验证边界 |

BlockTracker、KalmanMotion、BlockTrack.msg、BlockTrackArray.msg 均未修改，
已用 [原始哈希](perception_baseline/unchanged_contract.sha256)核对。
本次没有修改 TCP、夹爪下降偏移、belt.yaml 或抓取控制器算法。

## 3. 前后流程

原流程：最新 RGB+深度 → 尺寸不等时 resize → 拟合 Z=a*u+b*v+c →
光轴深度差阈值 → 连通域 → 区域深度中位数/二维中心反投影 → TF → 跟踪/发布。

新流程：注册深度与内参校验 → 缓冲配对（默认 ≤20 ms、每个深度时间戳只用一次）
→ 采集时间的 optical→base TF → 加载通过空带检查的固定几何 → 可切换分割
→ 侵蚀边缘、剔除无效/离群深度、稳健区域深度与稳定二维中心反投影 →
base 三维表面中心、同一 XY 下带面 Z 和垂直高度 → 原跟踪器/原消息。

平面标定使用明确的空载绿色带面 ROI、多帧独立深度、可选粗高度先验。
先用真实三维 RANSAC 检查空间支撑、倾角和残差，再决定是否采用 Z_b=h。
固定模型的 h 是三维内点 Z 的中位数；同时报告 X/Y 分箱 Z 分布、斜率和跨帧稳定性。
RANSAC 使用三点叉积法向量、单位法向量正交距离、内点 SVD 重拟合。
固定假设失败不会静默改用另一算法；可以检查 TF 后显式选择 ransac3d。

正常检测使用固定标定结果，不做实时三维 RANSAC。缓存光学射线后直接计算
点到平面对应的垂直高度图，与逐像素反投影、TF、三维平面计算严格等价，
避免正常检测构建完整 H×W×3 数组。完整三维点云用于空带标定。

| 检测模式 | 检测依据 | 深度/平面用途 |
|---|---|---|
| legacy_depth（模式 A 对照） | 原光轴深度差 | 原像素深度拟合，保留原 height 语义 |
| depth_plane（几何修正的模式 A） | base 垂直高度阈值 | fixed 或 ransac3d，真实几何 |
| rgb（模式 B） | ROI 内低饱和度灰色、排除绿色 | 内部有效深度定位，宽松已知带面高度检查，无实时 RANSAC |
| rgb_depth（模式 C） | 稳定 RGB 连通域 + 足够内部像素满足带面高度约束 | 抑制带面灰斑/坏深度，重点方案；fixed 或 ransac3d |

新模式均满足 `height = centroid_base.z - plane.z_at(x,y)`，
`belt_z = plane.z_at(x,y)`，因此 `belt_z + height = centroid_base.z`。
三维拟合残差是正交距离，发布 height 是沿 base +Z 的垂直高度，两者没有混用。

RGB 候选缺少足够内部深度、整帧深度覆盖不足、时间/内参/TF 不合格时，
拒绝整帧并报告原因，不发布“新鲜空带”。空带标定期间也不发布 tracks。
新平面文件记录光学系/base、K/D/尺寸、TF、像素偏移和诊断；运行时上下文变化
会要求重新标定。重新标定会归档旧文件，失败不会把旧文件当成本次结果。

## 4. 配置参数

距离均为米、角度为 rad；RGB 图像使用 BGR，HSV 使用 OpenCV H=0..179、S/V=0..255。
下表列出本次新增/启用参数及配置默认值。

| 参数 | 默认值 / 含义 |
|---|---|
| detection_mode / plane_mode | legacy_depth / fixed；上述四种检测、两种新平面可切换 |
| plane_file / plane_calibration_enabled | 空字符串 / false；新模式必须加载文件，标定必须显式开启 |
| plane_roi_px | [0,0,0,0]；标定禁止全零，必须指定空带 ROI |
| plane_calibration_frames / plane_calibration_max_points | 15 / 5000；独立帧数和每帧/汇总抽样上限 |
| base_plane_min_ratio | 0.7；新三维内点比例 |
| plane_max_tilt_rad / plane_max_3d_tilt_rad | 0.01 / 0.2；固定水平假设及三维带面倾角上限 |
| plane_max_flat_residual_m / plane_max_height_spread_m | 0.006 / 0.004；固定高度 P95 及跨帧残差中位数范围 |
| plane_min_xy_span_m | 0.05；X/Y 各自最小采样支撑范围 |
| plane_height_hint_enabled / plane_height_hint | true / -0.1028662282；来自既有 belt.yaml 的粗先验，不作为测量结果 |
| plane_height_hint_tolerance_m | 0.03；先验附近样本筛选带宽 |
| plane_empty_gray_max_ratio | 0.005；空载 ROI 中灰色比例门限 |
| sync_max_delta_s / image_max_age_s / sync_buffer_size | 0.02 / 0.2 / 8；配对误差、图像最大年龄、帧缓存 |
| color_rectified | false；raw 使用 K/D；rectified 使用 P，非单位 R/非零立体基线暂拒绝 |
| apply_d2c_offset | false；原经验像素平移默认停用，显式启用会警告 |
| depth_camera_info | /camera/depth/camera_info；必须对应已注册深度的彩色网格 |
| lower_green / upper_green | [35,40,40] / [85,255,255]；现在由三维 Detector 使用 |
| gray_saturation_max / gray_value_min / gray_value_max | 70 / 35 / 245；灰色候选范围 |
| depth_frame_min_valid_ratio | 0.1；检测 ROI 的最低有效深度覆盖 |
| depth_min_valid_pixels / depth_min_valid_ratio | 30 / 0.5；区域内部深度最低数量和比例 |
| depth_erode_px | 1；从候选边缘向内侵蚀像素数 |
| depth_region_mad_scale / depth_region_min_band_m | 3.0 / 0.003；深度 MAD 剔除和最小保留带宽 |
| rgb_height_min_m / rgb_height_max_m | 0 / 0.12；纯 RGB 模式的宽松几何范围 |

原 plane_ransac_distance=0.006 同时用于新三维正交残差；
plane_ransac_min_ratio=0.4、plane_refresh_every=10 仅用于旧基准。
block_height_min/max=0.008/0.060 在新模式中是基座系垂直高度。
roi_px、工作区、min_area_px 和所有跟踪/卡尔曼参数保留原接口。

## 5. 实际验证结果

完整原始指标：[水平场景](perception_validation_results/horizontal.json)、
[倾斜场景](perception_validation_results/tilted.json)、
[干扰场景](perception_validation_results/stress.json)、
[运行环境](perception_validation_results/environment.json)、
[软件回归输出](perception_validation_results/pytest.txt)。重现命令见
[验证方法](perception_validation.md)。

**所有下述数值都是本机合成离线实验，不是 Orbbec 或真实 Nero7 的实测精度。**
固定种子、320×240、60 帧、0.028 m/s、0.8 mm 光轴噪声、1% 缺失；
OpenBLAS/OMP/OpenCV 各一线程。误差以已知表面中心和垂直高度为真值。

水平带面估计 h=-0.102869045 m（生成真值 -0.102866228 m），倾角约
0.00005487 rad，正交残差 P95=1.456 mm，跨帧高度残差中位数范围=0.0438 mm。

| 水平场景 | XYZ RMSE (mm) | 高度 MAE (mm) | 带向速度 MAE (m/s) | 检测平均耗时 (ms) |
|---|---:|---:|---:|---:|
| legacy_depth | 0.662 | 2.565 | 0.000134 | 0.950 |
| depth_plane/fixed | 0.836 | 0.387 | 0.000133 | 0.781 |
| rgb/fixed | 0.606 | 0.134 | 0.000122 | 0.676 |
| rgb_depth/fixed | 0.606 | 0.134 | 0.000122 | 1.053 |

水平场景各模式误检、漏检和目标 ID 切换均为 0。几何深度模式的 XYZ RMSE
没有优于原基准；融合方案的检测耗时也略高于基准。收益主要是高度几何正确、
RGB 区域稳定及干扰抑制，不能声称每种新算法在每个指标都更好。

倾斜场景真值倾角 0.035 rad：fixed 明确拒绝；ransac3d 估计约 0.03460 rad。
rgb_depth/ransac3d 的 XYZ RMSE=1.442 mm，高度 MAE=0.164 mm，速度 MAE=
0.000193 m/s，误检/漏检/目标 ID 切换为 0。旧算法高度 MAE=3.662 mm。

干扰场景额外加入灰色带面斑块、斑块处 -6 mm 光轴偏差、物块内 10% 深度离群点：

| 干扰场景 | 误检次数 | 漏检次数 | XYZ RMSE (mm) | 高度 MAE (mm) | 目标 ID 切换 |
|---|---:|---:|---:|---:|---:|
| legacy_depth | 25 | 3 | 3.147 | 3.320 | 0 |
| depth_plane/fixed | 0 | 0 | 2.512 | 1.472 | 0 |
| rgb/fixed | 45 | 0 | 1.029 | 0.759 | 0 |
| rgb_depth/fixed | 0 | 0 | 0.726 | 0.406 | 0 |

纯 RGB 对灰色带面干扰会误报；融合平面高度约束在这组可控干扰中有效。
这里的次数是帧内检测事件，不是独立真实物体数。
耗时仅包括离线检测/相应上下文检查和旧模式重拟合，不含 ROS、相机注册、
GPU 抓取网络、跟踪或机械臂控制。离线倒数吞吐不能换算成真实处理帧率。
在线新增 /perception/diagnostics 的 process_ms、processed_fps_window 和
processed_fps 用于后续真实帧率测量。

软件回归覆盖：畸变投影/反投影、米单位、0/NaN/Inf、最近深度、PointCloud2
类型/填充/大小端、三维离群点、退化样本、水平/倾斜假设、跨帧漂移、ROI 内灰色污染、
标定上下文变化、区域深度缺失、深度帧去重、无 tracks 的标定、原跟踪 ID/速度、
ROS 序列化及控制器接触点预测、历史文件保留。最终测试数量与耗时见上方真实日志。

最终回归 **195 项通过**（7.30 秒，1 个既有 Matplotlib Axes3D 环境警告）；
nero7_perception、nero7_bringup、nero7_sim 三个包构建完成。
新入口安装、预览 launch 参数解析及跟踪/消息源码哈希核对通过。
合成测试有效参数另外保存为 [effective_params.yaml](perception_validation_results/effective_params.yaml)：
空带 ROI=[20,20,300,220]，检测全图，workspace_x=[-1,1]，其余使用项目配置。

## 6. 兼容性

新检测字典仍为 u/v、centroid_base、height、width_px、area_px、belt_z，
width_px 继续使用 sqrt(area)。轨迹与消息字段、ID 关联、常速度卡尔曼更新、
速度资格、观测/状态时间戳、协方差和 LOST 状态管理没有改动。
集成测试通过新检测→原跟踪器→ROS 序列化→TrackView→接触点预测。
原测速图像与抓取任务自动打开可视化行为保留。

兼容性结论针对软件接口和合成验证，尚不能宣称真实动态抓取已通过。
新模式 height 的几何语义纠正为基座系垂直高度；旧基准保留旧值供对照。
运行时应统一模式，避免多个检测器同时发布到同一 tracks 话题。

## 7. 尚未验证、误差与风险

- 未采集新的真实相机数据；没有现场平面精度、静态 XYZ、实景误检/漏检、遮挡 ID、
  独立测速或相机分辨率下真实帧率结果，也没有新的实机动态抓取结果。
- 固定模型依赖真实水平假设与正确 TF；倾角也可能来自外参旋转误差，不能自动调整
  TF 来“拉平”带面。空间支撑、颜色筛选、粗高度先验都可能使样本有偏。
- ROI 和 HSV 不能保证排除所有灰色机械臂、支架、阴影、反光或类似背景。现场
  光照/自动曝光会改变灰色范围；纯 RGB 的深度定位仍依赖可靠注册及表面深度。
- 像素区域中心加稳健深度是假定区域能代表可见表面中心；非平整顶面、透视、遮挡、
  孔洞和混合深度仍可能造成位置偏差。有效深度门限会使一些帧被拒绝。
- ≤20 ms 配对仍有剩余时差，0.028 m/s 时对应最大约 0.56 mm 运动量；这只是速度
  与时间差的计算，不证明驱动时间戳已同步或传感器曝光时间准确。
- 新检测器支持 raw K/D 畸变；现有 GR-ConvNet 包装仍使用 K 的针孔反投影。
  非零畸变原始图像下网络抓取点可能仍有偏差。应实测这部分，或提供正确的
  rectified RGB、注册深度和对应 CameraInfo；本次没有改变网络/控制器算法。
- 严格注册深度 CameraInfo 校验可能拒绝当前驱动不一致的元数据；需核查实际
  话题/内参，必要时使用真实点云投影，不能用 resize 或放宽坐标系检查掩盖问题。
- 平面文件不会更新 belt.yaml 的 belt_height、TCP、抓取偏移或已测时序。正式动态
  控制器还依赖这些参数；倾斜带面上的截取线高度/路径/时序没有自动适配。
  必须完成独立静态与运动验证后才决定实机执行设置。
- TF/内参变更会失效，但同一 TF 下真实带面移动无法仅靠文件上下文发现；
  传送带/相机重新安装、维护或位置变化后应重新采集空载数据并校核平面。
