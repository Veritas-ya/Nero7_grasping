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
colcon build --base-paths src --packages-up-to nero7_bringup --symlink-install
source install/setup.bash
python3 "$script_dir/static_once_session.py" --check-workspace "$workspace_dir"
echo "$$" > "$session_file"
cleanup() {
    python3 "$script_dir/static_once_session.py" --close-gate --session-file "$session_file" --owner-pid "$$" || true
    if [[ -f "$session_file" ]] && [[ $(cat -- "$session_file") == "$$" ]]; then rm -f -- "$session_file";fi
}
trap cleanup EXIT
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
