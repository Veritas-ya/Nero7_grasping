# 2026-10-12 相机外参更新

此报告记录 20:27:48 的版本，已被用户后来提供的 `base_to_camera_cv` 更新替换。
当前版本见 [反向矩阵更新记录](extrinsics_update_20261012_inverse.md)；本文件保留用于追溯。

使用用户提供的 `camera_to_world_cv`，单位米，光学坐标按 OpenCV 定义。
现场 ROS 查询与机器人 URDF 均确认 `world -> base_link` 为单位变换，
因此本次 `camera_to_base_cv = camera_to_world_cv`。

已同步更新 `src/nero7_bringup/config/calibration.yaml` 中的
`camera_to_world_cv`、`world_to_base_cv`、`camera_to_base_cv`、
`camera_in_base` 的平移及 wxyz 四元数、`depth_to_base_cv`。
深度光学外参通过现场相机内部 TF 组合得到，注册到彩色的深度图仍使用彩色光学系。
`gripper_tcp`、`tcp_offset`、`grasp_rpy`、`d2c_offset` 未修改。

旧配置备份为 `calibration.yaml.before_20261012_202748`，源目录和安装目录各有一份。
带面文件 `/tmp/nero7_live_belt_plane.yaml` 已归档为
`/tmp/nero7_live_belt_plane.previous.1791808068606753334.yaml`，不会作为新外参的有效平面使用。
外参平移变化约 120.55 mm，旋转变化约 7.587°。

更新前关闭抓取执行，并停止旧抓取控制器、检测器和相机外参发布进程。
随后仅发布新的 `base_link -> camera_link` 静态 TF，保留相机内部 TF 链。
节点名为 `nero7_camera_extrinsics_20261012`；启动后续整套系统前应先结束这个
独立 TF 发布进程，确保相机根坐标系只由一个外参发布者维护。
本次没有启动抓取，也没有发送位置、夹爪或使能指令；旧控制器退出时执行其原有停止流程。

验证：矩阵有限性、正交性和行列式通过；5 项相机 TF 单元测试通过；
重建 `nero7_bringup` 成功，源/安装配置相同；矩阵、四元数、两种启动入口以及
相机内部 TF 组合一致；归档的旧平面上下文会拒绝新外参。
输入矩阵包含约 5.8e-8 的旋转正交误差，保留原始输入数值；TF 四元数使用规范化旋转。
现场重新查询的彩色及深度光学系到 base_link 的 TF，与配置矩阵最大元素差均
小于 2.9e-8；world 到 base_link 仍为单位变换。反馈 motion_status=0、
err_status=0，旧抓取控制器和检测器已退出。

更新记录、用户原始矩阵及验证 JSON 位于 `build/extrinsics_update_20261012/`。
这些检查验证了配置和坐标链一致性，尚不能代表实际定位精度。
下次试抓前需用新外参重新标定空带平面，并检查独立已知点的静态三维定位误差。
现有 `belt.yaml` 带面高度和感知高度先验仍为旧测量值，需要现场核对后再使用；
本次没有直接修改带面高度、TCP 或抓取偏移。
