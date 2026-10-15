#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "$script_dir/.." && pwd)"

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
for preview_underlay in "${NERO7_ARM_WS:-$HOME/agx_arm_ws}" "${NERO7_MODEL_WS:-$HOME/nero_ws}"; do
  if [[ -f "$preview_underlay/install/setup.bash" ]]; then
    source "$preview_underlay/install/setup.bash"
  fi
done
if [[ ! -f "$workspace_dir/install/setup.bash" ]]; then
  echo "Build this workspace first: colcon build --base-paths src" >&2
  exit 1
fi
source "$workspace_dir/install/setup.bash"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

preview_params="$(python3 -c 'from ament_index_python.packages import get_package_share_directory; print(get_package_share_directory("nero7_perception") + "/config/camera_speed_preview.yaml")')"

# This ROI was checked with the current fixed camera at 1280 x 720.
exec ros2 launch nero7_bringup perception_preview.launch.py \
  "perception_params:=$preview_params" \
  start_camera:=true publish_camera_tf:=true detection_mode:=legacy_depth \
  visualize:=true debug_max_width:=0 legacy_rgb_components:=true tracker_lost_timeout:=1.0 \
  'roi_px:=[180,350,1060,450]' "$@"
