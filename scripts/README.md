# 当前静止抓取入口

`bash scripts/static_once_bin.sh` 自动编译根目录 `src` 的四个所需包，核对实际加载的
`static_once.py` 与用户源码一致，并以 `execution_enabled=false` 启动。
相机、物体检测和单个机械臂驱动需要保持运行。脚本统一 ROS domain 0、本机发现，
显式指定当前标定路径和左侧拾取区，不修改用户的抓取状态机。

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
