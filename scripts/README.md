# 当前静止抓取入口

`bash scripts/static_once_bin.sh` 自动编译根目录 `src` 的五个所需包，核对实际加载的
`static_once.py` 与用户源码一致，并以 `execution_enabled=false` 启动。
相机、物体检测和单个机械臂驱动需要保持运行。脚本统一 ROS domain 0、本机发现，
显式指定当前标定路径和左侧拾取区，并自动启动仅规划模式的 MoveIt。

本机首次使用且没有系统 MoveIt 时，先执行一次：

```bash
bash /home/ubuntu22/.codex/worktrees/da2d/Nero7_grasping/scripts/setup_moveit_local.sh
```

该脚本只把缺少的 MoveIt 二进制包解压到项目的 `.moveit_local`，不使用 sudo，
不修改 `/opt/ros`。当前工作站已经完成这一步。

夹紧后，控制器先完成 100 mm 垂直抬升，随后立即请求 OMPL
`RRTConnectkConfigDefault` 从实测关节角规划到固定物料框上方。MoveIt 只规划，
不会启动模拟控制器占用真实机械臂；现有 AGX 驱动依次执行返回的碰撞检查路径点。
到达箱口上方后才垂直下降和释放，抬回箱口上方后再用 RRTConnect 回固定初始位。
规划前必须同时满足相机点云新鲜且 MoveIt OctoMap 非空，否则停止在当前安全位置。

看到 `STATIC_READY` 并具备现场执行条件后，另一终端运行：

```bash
bash /home/ubuntu22/.codex/worktrees/da2d/Nero7_grasping/scripts/static_once_bin.sh --execute
```

脚本读取新鲜的 READY 状态，确认只有一个静止抓取节点且当前未执行，先请求打开驱动
控制门，再设置执行参数。失败时请求关门；控制器退出时，启动脚本也会关闭本轮打开的门。
执行入口不直接发布机械臂或夹爪运动。

以前的 `validation/latest-branch/source/scripts/static_once_bin.sh` 默认转到这个入口，
因此原先的两行启动命令仍可使用。原验证版脚本保存在同目录
`static_once_bin_validation.sh`，显式示教/配置文件选项仍走原验证版。

脚本启动通过不代表夹爪开合或放置动作已经实机验证。

## 夹爪单独诊断

先停止完整抓取脚本，保留一个通信驱动。只读检查：

```bash
bash /home/ubuntu22/.codex/worktrees/da2d/Nero7_grasping/scripts/check_gripper.sh
```

保持七轴静止且手离开夹爪后，单独发送一次 50 mm 夹爪目标：

```bash
bash /home/ubuntu22/.codex/worktrees/da2d/Nero7_grasping/scripts/check_gripper.sh --execute --target-mm 50
```

诊断目标限制为 50～70 mm，执行前检查七轴连续静止和控制发布者，结束后关闭控制门。
它不标零、不修改量程、不发送七轴位置目标。报告保存在根目录 `validation/latest-branch`。
CAN 本机发送帧是否可抓到取决于驱动的 `local_loopback`，不能仅凭未抓到帧断定未发送。

`gripper_driver_trace.py` 是本地调试入口，启用回环和夹爪请求日志，并禁止反馈恢复线程
在 `auto_enable=false` 时自动重新使能七轴；需要先退出原通信驱动，不能同时运行两个驱动。

当前静态状态机要求开爪/释放有三张新反馈且跨度至少 0.1 秒，未确认开口不会继续路径。
这项检查不能代替夹爪反馈与实际开口的标定验证。
