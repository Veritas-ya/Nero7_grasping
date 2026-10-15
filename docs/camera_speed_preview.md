# 相机测速预览

该入口启动相机、相机 TF、物块检测、ID 跟踪以及带速度文字的画面。
它只启动视觉节点；机械臂驱动和抓取控制器需要各自单独启动。

## 构建和启动

在仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
source ~/agx_arm_ws/install/setup.bash
source ~/nero_ws/install/setup.bash
colcon build --base-paths src
bash scripts/camera_speed_preview.sh
```

脚本默认从 `~/agx_arm_ws` 和 `~/nero_ws` 加载已有工作空间；安装位置不同
时可设置 `NERO7_ARM_WS`、`NERO7_MODEL_WS`。ROS 域默认是 0，默认只在本机
通信；已有 `ROS_DOMAIN_ID`、`ROS_LOCALHOST_ONLY` 环境变量时保留其设置。

相机或相机 TF 已经运行时，避免重复启动对应节点：

```bash
bash scripts/camera_speed_preview.sh start_camera:=false publish_camera_tf:=false
```

只重新打开测速画面时，在已加载 ROS 环境的桌面终端执行：

```bash
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

画面通过原始 `sensor_msgs/Image` 发布，`debug_max_width=0` 保留相机实际
分辨率。现场相机为 1280×720；窗口仍可按显示大小缩放。脚本使用的 ROI
`[180,350,1060,450]` 针对当前固定相机和带面，移动相机后需要重新检查。

## 代码和参数

- `src/nero7_perception/nero7_perception/block_detector.py`：RGB-D 同步、检测调用、消息和图像发布。
- `detection.py`：深度分割及可选的 RGB/深度连通区域融合。
- `block_tracker.py`：全局一对一 ID 匹配及每个目标的卡尔曼状态更新。
- `speed_overlay.py`：ID、速度和失效状态文字。
- `src/nero7_perception/config/camera_speed_preview.yaml`：本次现场使用的测速参数。

这个预览显式启用 `legacy_rgb_components=true`，合并同一物块的多层可见
表面，并用颜色边缘支持过滤缺少轮廓的平滑反光。通用检测配置仍默认关闭
该选项，保留 legacy 对照行为。此模式沿用光轴方向的高度语义，不能替代
基座坐标系的三维带面标定。

ID 匹配使用全局一对一分配，同时保留物理距离和卡尔曼创新门限。预览把
ID 保留时间设置为 1 秒；丢失目标依然立即隐藏数值速度，重新观测后仍需
满足测速资格。卡尔曼状态为 `[x,y,vx,vy]`，画面显示 XY 速度模长，单位
为 m/s；消息中的 `speed` 是沿传送带方向的有符号投影。

预览需要至少 20 次有效观测、至少 0.8 秒时间跨度，以及不超过 0.003 m/s
的模型速度标准差。后者是不确定性门限，不能作为真实测速精度的保证。
`WAITING`、`LOST` 和检测异常画面均不显示旧的数值速度。检测失败时仍
发布当前彩色图及错误原因，避免窗口停留在上一帧的旧结果。

## 已有现场结果和范围

两件静止物块的同一段 90 帧实拍回放中，修改前产生 22 个 ID，修改后保持
两个 ID。一次 20 秒在线观察中，两个原有物块在 525 组输出中均保持各自
ID；反光区域仍可能产生短暂候选，不能据此声称反光误检已经完全消除。

独立测得带速为 0.0327 m/s 的一次 60 秒记录中，两件物块的合格速度均值
分别为 0.03145 和 0.03153 m/s，标准差约为 0.00039 和 0.00024 m/s，平均
偏低约 3.8% 和 3.6%。未通过比例补偿把读数强行对齐参考值。开始记录时
速度已经合格，这次记录不能验证启动收敛时间，也不能证明运动抓取成功。

相关回归覆盖多层物块、相邻物块、平滑反光、全局 ID 匹配、断帧后的测速
资格、原尺寸图像、标签避让及检测异常画面。实拍原始数据和构建产物保留
在现场机器，不作为源码提交。
