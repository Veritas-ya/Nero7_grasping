#!/usr/bin/env bash
# Source the project-local MoveIt runtime when the system package is absent.
_nero_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
_moveit_prefix="$_nero_root/.moveit_local/root/opt/ros/humble"
if [[ ! -x "$_moveit_prefix/lib/moveit_ros_move_group/move_group" ]]; then
    echo '本机缺少 MoveIt 运行包。请先运行：bash scripts/setup_moveit_local.sh' >&2
    return 1 2>/dev/null || exit 1
fi
export AMENT_PREFIX_PATH="$_moveit_prefix:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$_moveit_prefix:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$_moveit_prefix/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$_moveit_prefix/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
unset _nero_root _moveit_prefix
