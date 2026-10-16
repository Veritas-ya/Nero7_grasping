#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
debs="$root/.moveit_local/debs"
prefix="$root/.moveit_local/root"
mkdir -p "$debs" "$prefix"
cd "$debs"
packages=(
  ros-humble-moveit-msgs ros-humble-moveit-common ros-humble-moveit-core
  ros-humble-moveit-kinematics ros-humble-moveit-ros-occupancy-map-monitor
  ros-humble-moveit-ros-planning ros-humble-moveit-planners-ompl
  ros-humble-moveit-simple-controller-manager ros-humble-moveit-ros-move-group
  ros-humble-moveit-ros-perception ros-humble-moveit-ros-planning-interface
)
apt-get download "${packages[@]}"
for deb in ./*.deb; do dpkg-deb -x "$deb" "$prefix"; done
cat > "$prefix/opt/ros/humble/local_setup.bash" <<'EOF'
# Project-local binary extraction; environment is configured by scripts/moveit_local_setup.bash.
EOF
echo "MoveIt 已安装到项目本地目录：$prefix"
