"""Gazebo example: rendered vision -> the unchanged formal grasping algorithm."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nero7_bringup.launch_common import DEFAULT_URDF
from nero7_sim.scene import generate


def start(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)
    mode = arg('mode')
    if mode not in ('preview', 'calibrate', 'pick', 'demo', 'static'):
        raise ValueError('mode must be preview, calibrate, pick, demo or static')
    share = Path(get_package_share_directory('nero7_sim'))
    directory = Path(arg('output_dir')); directory.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='scene_', dir=directory))
    for name in ('plan.json', 'trial.json'):
        previous = directory/name
        if previous.exists():
            previous.rename(directory/f'{previous.stem}.previous.{run.name}.json')
    example = yaml.safe_load(Path(arg('example_file')).read_text())
    if mode == 'static':
        example['belt_speed'], example['spawn_distance_upstream'] = 0., 0.
    (run/'example.yaml').write_text(yaml.safe_dump(example))
    clock_parameters = run/'gazebo.yaml'
    clock_parameters.write_text(yaml.safe_dump({'gazebo': {'ros__parameters': {'publish_rate': 100.}}}))
    files = generate(run, arg('urdf_path'), arg('calibration_file'), arg('belt_file'), run/'example.yaml')
    timing = directory/'verified_timings.json'
    if mode in ('calibrate', 'demo') and timing.exists():
        timing.rename(directory/f'verified_timings.previous.{run.name}.json')
    tf_remap = [('/tf', '/sim/tf'), ('/tf_static', '/sim/tf_static'), ('/clock', '/sim/clock')]
    controller_remap = tf_remap+[(name, '/sim'+name) for name in
        ['/grasp/state', '/grasp/candidate', '/grasp/delta_s', '/grasp/reset']]
    common = dict(use_sim_time=True, mode='conveyor_once',
                  params_file=str(Path(get_package_share_directory('nero7_control'))/'config/controller.yaml'),
                  topic_prefix='/sim/nero7/', tracks_topic='/sim/perception/tracks',
                  color_topic='/sim/camera/color/image_raw', depth_topic='/sim/camera/depth/image_raw',
                  color_camera_info='/sim/camera/color/camera_info',
                  qos_reliability='reliable',
                  visualization_enabled=arg('gui') == 'true',
                  visualization_topic='/sim/perception/debug_image',
                  belt_file=files['belt'], calibration_file=files['calibration'], urdf_path=files['urdf'],
                  conveyor_driver_node='/sim/nero7/driver', inference_device=arg('inference_device'),
                  static_infer_timeout=120., result_max_age=.5,
                  static_roi_px=example['roi_px'],
                  conveyor_calibration_yaw=example['closing_yaw'],
                  conveyor_timing_std=example['conveyor_timing_std'],
                  conveyor_plan_file=str(directory/'plan.json'),
                  conveyor_timing_file=str(directory/'trial.json'),
                  **{'workspace.x_min': -.50})

    def controller(name, **parameters):
        return Node(package='nero7_control', executable='grasp_controller', name=name, namespace='/sim',
                    output='screen', parameters=[dict(common, **parameters)], remappings=controller_remap)

    def scenario():
        return Node(package='nero7_sim', executable='sim_scenario', namespace='/sim',
                    parameters=[dict(use_sim_time=True, delay=3.)], remappings=tf_remap, output='screen')

    gazebo_env = dict(GAZEBO_MASTER_URI=f'http://127.0.0.1:{arg("master_port")}',
                      GAZEBO_PLUGIN_PATH=os.pathsep.join([get_package_prefix('nero7_sim')+'/lib',
                                                        get_package_prefix('gazebo_ros')+'/lib',
                                                        os.environ.get('GAZEBO_PLUGIN_PATH', '')]))
    if arg('software_rendering') == 'true':
        gazebo_env.update(LIBGL_ALWAYS_SOFTWARE='1', __GLX_VENDOR_LIBRARY_NAME='mesa')
    actions = [ExecuteProcess(cmd=['gzserver', '--verbose', files['world'], '-s', 'libgazebo_ros_init.so',
                                  '--ros-args', '-r', '/clock:=/sim/clock',
                                  '--params-file', str(clock_parameters), '--'],
                              additional_env=gazebo_env, output='screen'),
               Node(package='nero7_sim', executable='sim_camera_sync', namespace='/sim',
                    parameters=[dict(use_sim_time=True)], remappings=tf_remap, output='screen'),
               Node(package='nero7_sim', executable='sim_driver', name='driver', namespace='/sim/nero7',
                    parameters=[dict(use_sim_time=True, urdf_path=files['urdf'], **{
                        key: example[key] for key in ('initial_joints', 'speed_percent',
                            'cartesian_speed_at_100_percent', 'joint_speed_at_100_percent', 'gripper_speed')})],
                    remappings=tf_remap, output='screen'),
               Node(package='robot_state_publisher', executable='robot_state_publisher', namespace='/sim',
                    parameters=[dict(use_sim_time=True, robot_description=Path(files['urdf']).read_text())],
                    remappings=tf_remap+[('/joint_states', '/sim/nero7/gazebo/joint_states')]),
               Node(package='nero7_perception', executable='block_detector', namespace='/sim',
                    parameters=[dict(use_sim_time=True,
                        params_file=str(Path(get_package_share_directory('nero7_perception'))/'config/perception.yaml'),
                        color_topic='/sim/camera/color/image_raw', depth_topic='/sim/camera/depth/image_raw',
                        # Gazebo RGB/depth share one optical sensor/grid and CameraInfo.
                        depth_camera_info='/sim/camera/color/camera_info',
                        qos_reliability='reliable',
                        points_topic='/sim/camera/depth/points', color_camera_info='/sim/camera/color/camera_info',
                        belt_file=files['belt'], calibration_file=files['calibration'], min_area_px=300,
                        roi_px=example['roi_px'], **{key: example[key] for key in
                            ('kalman_accel_noise_std', 'kalman_cross_accel_noise_std')})],
                    remappings=tf_remap+[('/perception/tracks', '/sim/perception/tracks'),
                                        ('/perception/diagnostics', '/sim/perception/diagnostics'),
                                        ('/perception/debug_image', '/sim/perception/debug_image')], output='screen')]
    optical = np.array(yaml.safe_load(Path(files['calibration']).read_text())['camera_to_base_cv'])
    quat = Rotation.from_matrix(optical[:3, :3]).as_quat()
    actions.append(Node(package='tf2_ros', executable='static_transform_publisher', namespace='/sim',
                        arguments=[str(v) for v in [*optical[:3, 3], *quat]]+
                                  ['base_link', 'camera_color_optical_frame'], remappings=tf_remap,
                        parameters=[dict(use_sim_time=True)]))
    if arg('gui') == 'true':
        actions.append(ExecuteProcess(cmd=['gzclient'], additional_env=gazebo_env, output='screen'))
    if mode in ('calibrate', 'demo'):
        calibration = controller('calibrator', execution_enabled=True, conveyor_calibration_only=True,
                                 conveyor_timing_file=str(timing), exit_on_complete=True)
        actions.append(calibration)
        if mode == 'demo':
            def completed(event, context):
                try: report = json.loads(timing.read_text())
                except (OSError, ValueError): report = {}
                if event.returncode != 0 or not report.get('timing_verified') or not any(
                        e['state']=='CONVEYOR_CALIB_DONE' for e in report.get('events', [])):
                    return [LogInfo(msg='Calibration failed; pick will not start. Inspect verified_timings.json.')]
                return [controller('picker', execution_enabled=True, conveyor_load_timings_file=str(timing)), scenario()]
            actions.append(RegisterEventHandler(OnProcessExit(target_action=calibration, on_exit=completed)))
    else:
        parameters = dict(execution_enabled=mode in ('pick', 'static'))
        if mode == 'pick': parameters['conveyor_load_timings_file'] = str(timing)
        if mode == 'static': parameters['mode'] = 'static_once'
        actions += [controller('picker', **parameters), scenario()]
    (directory/'scene_files.json').write_text(json.dumps(files, indent=2))
    return actions


def generate_launch_description():
    bringup = Path(get_package_share_directory('nero7_bringup'))
    sim = Path(get_package_share_directory('nero7_sim'))
    defaults = dict(mode='preview', gui='true', software_rendering='false', master_port='11355',
                    output_dir='/tmp/nero7_gazebo', urdf_path=DEFAULT_URDF,
                    calibration_file=str(bringup/'config/calibration.yaml'),
                    belt_file=str(bringup/'config/belt.yaml'), example_file=str(sim/'config/example.yaml'),
                    inference_device='cpu')
    return LaunchDescription([DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()]
                             +[OpaqueFunction(function=start)])
