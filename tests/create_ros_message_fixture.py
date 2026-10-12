"""Create CI-only AGX message doubles outside the project workspace."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True)
target = Path(parser.parse_args().output_dir)/'agx_arm_msgs'
(target/'msg').mkdir(parents=True, exist_ok=True)
(target/'package.xml').write_text('''<?xml version="1.0"?>
<package format="3">
  <name>agx_arm_msgs</name><version>0.0.0</version>
  <description>CI message doubles; never install on a robot.</description>
  <maintainer email="test@example.invalid">CI</maintainer><license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
</package>
''')
(target/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.8)
project(agx_arm_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/AgxArmStatus.msg" "msg/GripperStatus.msg")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
''')
(target/'msg/AgxArmStatus.msg').write_text('uint8 arm_status\nuint8 motion_status\nuint32 err_status\n')
(target/'msg/GripperStatus.msg').write_text('float64 width\n')
print(target)
