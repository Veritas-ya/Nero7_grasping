#!/usr/bin/env bash
set -Ee -o pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arg in "$@"; do
    if [[ "$arg" == --execute ]]; then
        exec 9>/tmp/nero7_visual_once_operator.lock
        flock -n 9 || { echo '请先停止完整抓取脚本，再进行夹爪单独测试。' >&2; exit 1; }
        break
    fi
done
source /opt/ros/humble/setup.bash
source /home/ubuntu22/agx_arm_ws/install/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=/tmp/nero7_gripper_probe
exec python3 -u "$script_dir/gripper_probe.py" "$@"
