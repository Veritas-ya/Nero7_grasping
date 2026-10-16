#!/usr/bin/env bash
# Operator entry for the user's current root-workspace StaticOnce implementation.
set -Ee -o pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(dirname -- "$script_dir")
color_root=/camera/color
execute=false
while (($#)); do
    case "$1" in
        --execute) execute=true; shift ;;
        --color-root)
            if (($#<2)); then echo '缺少 --color-root 参数。' >&2; exit 2; fi
            color_root=${2%/};shift 2 ;;
        --help|-h)
            echo '默认：编译根目录当前版本并启动静止抓取预览。'
            echo '用法：bash scripts/static_once_bin.sh [--color-root /camera/color]'
            echo '看到 STATIC_READY 后，另一终端运行同一脚本加 --execute。'
            exit 0 ;;
        *) echo "未知参数：$1" >&2;exit 2 ;;
    esac
done
cd -- "$workspace_dir"
source /opt/ros/humble/setup.bash
source "$script_dir/moveit_local_setup.bash"
if [[ -f /home/ubuntu22/ws_moveit/install/setup.bash ]]; then
    source /home/ubuntu22/ws_moveit/install/setup.bash
fi
source /home/ubuntu22/agx_arm_ws/install/setup.bash
source /home/ubuntu22/nero_ws/install/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR=/tmp/nero7_user_static_logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero7_mpl_cache
session_file="/tmp/nero7_user_static_${UID}.pid"
if $execute; then
    exec python3 "$script_dir/static_once_session.py" --execute --session-file "$session_file"
fi
exec 9>/tmp/nero7_visual_once_operator.lock
flock -n 9 || { echo '已有抓取或示教启动脚本运行，请先停止旧的抓取终端。';exit 1; }
echo "编译您当前修改的版本：$workspace_dir/src/nero7_control/nero7_control/static_once.py"
colcon build --base-paths src --packages-up-to nero7_bringup --symlink-install \
    --allow-overriding agx_arm_description
source install/setup.bash
python3 "$script_dir/static_once_session.py" --check-workspace "$workspace_dir"
echo "$$" > "$session_file"
cleanup() {
    python3 "$script_dir/static_once_session.py" --close-gate --session-file "$session_file" --owner-pid "$$" || true
    if [[ -f "$session_file" ]] && [[ $(cat -- "$session_file") == "$$" ]]; then rm -f -- "$session_file";fi
}
trap cleanup EXIT
moveit_pid=''
if ! ros2 action list --no-daemon 2>/dev/null | grep -qx '/move_action'; then
    echo '启动仅规划模式 MoveIt（RRTConnect + 相机 OctoMap）……'
    ros2 launch nero7_bringup moveit_plan_only.launch.py \
        arm_type:=nero effector_type:=agx_gripper follow:=true \
        point_cloud_topic:=/camera/depth/points use_sim_time:=false \
        > /tmp/nero7_moveit_plan_only.log 2>&1 &
    moveit_pid=$!
    for _ in $(seq 1 40); do
        ros2 action list --no-daemon 2>/dev/null | grep -qx '/move_action' && break
        sleep .25
    done
    if ! ros2 action list --no-daemon 2>/dev/null | grep -qx '/move_action'; then
        cat /tmp/nero7_moveit_plan_only.log >&2
        echo 'MoveIt /move_action 启动失败。' >&2
        exit 1
    fi
fi
cleanup_moveit() {
    if [[ -n "$moveit_pid" ]]; then kill "$moveit_pid" 2>/dev/null || true; fi
}
trap 'cleanup_moveit; cleanup' EXIT
echo '当前仅预览，不自动使能或移动机械臂。请保持相机、检测和一个机械臂驱动运行。'
echo "看到 STATIC_READY 后，在另一终端运行：bash $script_dir/static_once_bin.sh --execute"
ros2 run nero7_control grasp_controller --ros-args \
    -p mode:=static_once -p backend:=ros_topic \
    -p execution_enabled:=false -p exit_on_complete:=true \
    -p visualization_enabled:=false \
    -p "calibration_file:=$workspace_dir/src/nero7_bringup/config/calibration.yaml" \
    -p "urdf_path:=/home/ubuntu22/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf" \
    -p static_center_source:=network \
    -p static_max_network_track_xy_m:=0.04 \
    -p infer_target_radius_px:=25.0 \
    -p workspace.x_min:=-0.50 \
    -p 'static_roi_px:=[180,355,600,475]' \
    -p "color_topic:=$color_root/image_raw" \
    -p "color_camera_info:=$color_root/camera_info"
