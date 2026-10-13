# 传送带跟随抓取（conveyor_follow）

这是一个独立的单次抓取模式，与 conveyor_once 的定点拦截并存。默认
execution_enabled=false，只预览，不发使能、机械臂运动或夹爪命令。
一次成功后抬起并保持物体；不会自动放回传送带或松爪。

## 算法和可观测量

1. 在检测器第一次接受物体图像时，就绑定 ID、记录观测轨迹并请求 GR-ConvNet
   抓取点、方向、宽度。速度尚未通过资格检查时显示已观测轨迹；单帧不能确定速度。
2. 复用感知端 XY 常速度卡尔曼状态、图像时间和协方差，预测物体进入
   follow_region 的时间。计算准备、到位、同步、下降、闭合、抬升和减速所需的
   完整时间。物体当前可以在范围上游；所有实际跟随动作必须在范围内。
3. 后台线程验证整个跟随路径的 URDF 逆解、关节插值、奇异性、速度、法兰工作区
   和带面间隙。目标 ID、速度、抓取偏移、方向、宽度或机械臂起始姿态改变会使计划失效。
   准备/到位超过预算会终止本次抓取。
4. 到物体上方后开始连续跟随。控制律为对象速度前馈 + TCP 位置误差反馈；
   固定抓取方向，并用带阻尼的几何雅可比转换为有速度/加速度限制的关节目标。
   不是反复发送固定目标 move_l。
5. 使用真实反馈中的 TCP 位姿估计末端速度。至少 3 个不同反馈样本、持续 0.15 s
   满足相对速度、位置置信界和夹爪开口裕量，才开始下降。下降用五次曲线平滑 Z，
   同时继续跟随 XY；闭合时也持续跟随。
6. 闭合后要求多个新夹爪反馈确认宽度被物体阻挡。短暂手部遮挡仅允许在已发
   闭合命令后预测，最长默认 0.35 s，保持原始观测时间、传播协方差；速度失效/
   重新关联不能当作遮挡绕过。确认夹住后抬升，离开带面后平滑减速并保持。

速度的“相等”通过实际反馈在容差内验证，不能理解为数学上的严格相等。
位姿、关节和夹爪反馈均检查原始时间戳；重复发布旧消息不计为新样本。
默认相对 XY 速度容差 8 mm/s，沿带位置置信界 16 mm、横向 12 mm，
还要满足物体宽度对应的实际开口余量。默认试验速度为 8–40 mm/s。
这些是执行门限，不是已验证的实机性能指标。

## 驱动要求

当前执行后端为 ros_topic。官方 AGX ROS 驱动提供：
control/move_js (sensor_msgs/JointState) -> SDK move_js。
发送 joint1…joint7 的连续位置目标，驱动不使用 JointState.velocity。
move_js 是快速跟随接口，没有内置平滑/轨迹规划；arm_speed_percent
不能代替本模式的速度/加速度限制。控制器在默认 50 Hz 下主动生成平滑目标，
每次采样间隔超过 60 ms、实际关节落后命令超过 0.08 rad、接近奇异位姿等会请求停止。

接口依据（安装版本仍须核对）：
- [AGX ROS 驱动源码](https://github.com/agilexrobotics/agx_arm_ros/blob/e4ccec1999279d063c2962030b9437b78b0d0db6/src/agx_arm_ctrl/agx_arm_ctrl/agx_arm_ctrl_single_node.py)
- [Nero SDK API](https://github.com/agilexrobotics/pyAgxArm/blob/841a625f5f4920e776f20b934eb13048b747e6d0/docs/nero/nero_api.md)

执行要求该话题恰有一个订阅者、且只有本控制器一个发布者。
缺少接口或出现竞争发布者时拒绝执行，不自动退回到 move_l。
停止旧 grasp_controller 后再启动本模式；不能同时运行两种控制模式。
反馈 tcp_pose 必须与 URDF 法兰一致，TCP 使用现有 calibration.yaml 的 gripper_tcp。

## 配置和预览

新增 src/nero7_control/config/follow.yaml。follow_region 为
[x_min, x_max, y_min, y_max]，单位米，表示 base_link 中 TCP 的可用抓取范围。
默认只是现有工作区的粗矩形；现场应将它缩到量过的传送带有效范围，并保留
20 mm 默认边界余量。可达性由逐点 IK 进一步检查；IK 不提供整条机械臂与环境的碰撞检测。

先更新工作分支并重建 nero7_control / nero7_bringup，source 新 install/setup.bash。
使用已有 AGX 驱动（只保留一个 CAN 控制者），停止旧检测器/控制器后预览：

~~~bash
ros2 launch nero7_bringup follow_preview.launch.py start_camera:=true
~~~

相机已经运行时保留 start_camera:=false。已发布相机 TF 时加
publish_camera_tf:=false。已有验证过的感知配置时，传入原有 detection_mode、
plane_mode、plane_file、perception_params、roi_px / plane_roi_px。
ROI 必须覆盖上游进入段，不能只留抓取线附近；这直接影响提前规划能获得的时间。
未修改现场标定、TCP 或原模式默认配置。自定义模型路径等基础参数写入
follow.yaml 或通过 controller_params 指定现场参数文件。

查看：

~~~bash
ros2 topic echo /grasp/follow/diagnostics
ros2 topic echo /grasp/state
ros2 topic info /control/move_js --verbose
~~~

/grasp/follow/trajectory 是 PoseArray：速度收敛前为观测轨迹，收敛后为
按视觉估计预测的对象轨迹（不是仿真真实速度）。诊断默认 5 Hz，含实际/
参考末端速度、位置、相对速度误差和拒绝原因。完整计划的末端路径及时间写入
/tmp/nero7_follow_plan.json。不要把只看到了预览轨迹等同于已经可以执行。

现场验证过跟随响应、区域、反馈频率和延迟后，保持默认执行关闭启动，再用：

~~~bash
ros2 param set /grasp_controller execution_enabled true
~~~

关闭 execution_enabled 会在下一控制循环请求停止；原始物理急停独立保留。
失去反馈、驱动故障或抓取不明确时终止本次，保留夹爪状态，不发送自动松爪/
回位命令。单次模式完成/终止后需重启控制器重新规划。

## 验证边界

新增纯逻辑测试验证区域进入、完整时间预算、横向漂移、雅可比有限差分、
带执行滞后模型的下降/闭合速度跟随、闭合遮挡、夹持确认和停止条件。
状态机测试执行正式控制方法，驱动/运动学为明确测试替身。
ROS Humble 测试构建真实项目包，验证实际 JointState 序列化、话题端点、
新模式初始化、反馈时间戳/关节名称映射、配置安装及默认预览；AGX 消息/GPU 模型为 CI 替身。

Gazebo 驱动增加同名连续接口和断流停止，可用于后续接线；现有 Gazebo 示例
仍使用 conveyor_once，没有声称已经运行 follow 的 Gazebo 物理抓取实验。
本次 CI 没有实机相机、CAN、GPU 或夹爪物理接触。没有修改卡尔曼噪声
来强行让预测通过，也没有使用已知传送带速度替代视觉测速。

当前模型适用于平面传送带上的短期平移：预测 XY、抓取高度保持不变，
不预测物体翻滚/旋转或复杂三维轨迹。相机处理延迟和速度突变会使预测失效；
现场若频繁因置信界、循环超时或速度不匹配终止，应先解决反馈/感知频率和
控制响应，不能仅放宽门限。跟随模式尚需低速空夹跟随与单物体实机验证。
