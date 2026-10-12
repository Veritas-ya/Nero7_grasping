# 2026-10-12 20:46:20 最新外参更新

用户提供的 `base_to_camera_cv` 表示 `base_link -> camera_color_optical_frame`，单位米。
项目使用的方向是 `camera_to_base_cv`，因此使用完整 4×4 矩阵的逆：

```text
camera_to_base_cv = inverse(base_to_camera_cv)

[-0.053309412, -0.825009225,  0.562599222, -0.859036975]
[-0.998265135,  0.029926473, -0.050706275, -0.177095236]
[ 0.024996534, -0.564326310, -0.825173308,  0.744145644]
[ 0.000000000,  0.000000000,  0.000000000,  1.000000000]
```

上述显示值已取舍；配置保存完整求逆精度，并保留用户原始 `base_to_camera_cv`。
当前 ROS 的 `world -> base_link` 仍为单位变换，`camera_to_world_cv` 已同步更新。
`camera_in_base` 的平移及 wxyz 四元数、`depth_to_base_cv` 也已同步。
深度光学矩阵根据相机现场内部 TF 组合，注册深度仍属于彩色光学系。

源文件为 `src/nero7_bringup/config/calibration.yaml`，上一版本备份为
`calibration.yaml.before_20261012_204620`，安装目录中也有对应备份。
重建 `nero7_bringup` 后，源文件与安装文件逐字节一致。
`gripper_tcp`、`tcp_offset`、`grasp_rpy`、`d2c_offset` 没有修改。

输入旋转行列式为 0.999999999999994，最大正交误差为 4.55e-15；
两向求逆回环最大元素误差为 1.11e-16。5 项相机 TF 测试通过；
矩阵/四元数、两个启动入口、彩色与深度光学变换组合一致。
这些数值是计算一致性，不能作为物理定位精度。

已停止上一版独立相机 TF，仅发布最新 `base_link -> camera_link`，
节点名仍为 `nero7_camera_extrinsics_20261012`；没有启动检测器或抓取控制器。
本次未发送机械臂运动命令。原带面文件仍处于归档状态，且旧平面上下文会拒绝最新外参。
试抓前需重新标定空带平面并校核实际静态定位，旧 `belt.yaml` 高度先验也需现场检查。

完整输入、更新信息和静态/现场验证结果位于
`build/extrinsics_update_20261012_inverse/`。旧版本记录保持独立以便回退。

现场正向、反向 TF 与配置及用户原矩阵的最大元素差均小于 1e-12；
反馈 motion_status=0、err_status=0，机械臂保持停止。
