"""Generate a Gazebo scene from the same URDF, TCP and optical extrinsics as hardware."""
import copy
import math
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def child(parent, tag, value=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if value is not None:
        element.text = str(value)
    return element


def numbers(values):
    return ' '.join(f'{float(v):.12g}' for v in values)


def geometry(link, name, size, color, pose=None):
    for kind in ('collision', 'visual'):
        item = child(link, kind, name=name)
        if pose is not None:
            child(item, 'pose', numbers(pose))
        box = child(child(item, 'geometry'), 'box')
        child(box, 'size', numbers(size))
        if kind == 'visual':
            material = child(item, 'material')
            child(material, 'ambient', color); child(material, 'diffuse', color)
        else:
            friction = child(child(child(item, 'surface'), 'friction'), 'ode')
            child(friction, 'mu', 0); child(friction, 'mu2', 0)


def robot_overlay(urdf, initial, tcp):
    root = ET.parse(urdf).getroot()
    root.set('name', 'nero7')
    mesh_root = Path(urdf).resolve().parent.parent
    for mesh in root.iter('mesh'):
        uri = mesh.get('filename')
        if uri.startswith('package://nero_description/'):
            filename = mesh_root/uri.removeprefix('package://nero_description/')
            if not filename.exists():
                raise FileNotFoundError(filename)
            mesh.set('filename', filename.as_uri())
    # Preserve all joint origins/axes/limits. Motor parameters in the hardware
    # export are zero; the example uses ideal position servos, not motor dynamics.
    for joint in root.findall('joint'):
        limit = joint.find('limit')
        if limit is not None:
            limit.set('effort', '50'); limit.set('velocity', '3')
    for link in root.findall('link'):
        if link.get('name') == 'world':
            continue
        gz = child(root, 'gazebo', reference=link.get('name'))
        child(gz, 'gravity', 'false')
        child(gz, 'selfCollide', 'false')
        # Arm collisions are simplified out; grasp geometry is checked by the
        # scene plugin. Original visual meshes and kinematics remain intact.
        for collision in link.findall('collision'):
            link.remove(collision)
    for side, sign in [('left', 1), ('right', -1)]:
        name = f'{side}_finger'
        link = child(root, 'link', name=name)
        inertial = child(link, 'inertial')
        ET.SubElement(inertial, 'mass', {'value': '.03'})
        child(inertial, 'inertia', ixx='.00001', iyy='.00001', izz='.00001', ixy='0', ixz='0', iyz='0')
        visual = child(link, 'visual')
        child(child(visual, 'geometry'), 'box', size='.06 .016 .006')
        material = child(visual, 'material', name='finger_grey')
        child(material, 'color', rgba='.15 .15 .17 1')
        joint = child(root, 'joint', name=f'{side}_finger_joint', type='prismatic')
        child(joint, 'parent', link='link7'); child(joint, 'child', link=name)
        child(joint, 'origin', xyz=numbers(np.asarray(tcp)-[.03, 0, 0]), rpy='0 0 0')
        child(joint, 'axis', xyz=f'0 0 {sign}')
        child(joint, 'limit', lower='.003', upper='.038', effort='20', velocity='.10')
        gz = child(root, 'gazebo', reference=name); child(gz, 'gravity', 'false')
    return root


def generate(directory, urdf_path, calibration_path, belt_path, example_path):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    calibration = yaml.safe_load(Path(calibration_path).read_text())
    belt = yaml.safe_load(Path(belt_path).read_text())
    example = yaml.safe_load(Path(example_path).read_text())
    # Rendered RGB/depth share one optical grid, so the physical sensor's pixel
    # registration correction is unnecessary in this copy of the calibration.
    sim_calibration = copy.deepcopy(calibration); sim_calibration['d2c_offset'] = [0, 0]
    (directory/'calibration.yaml').write_text(yaml.safe_dump(sim_calibration))
    (directory/'belt.yaml').write_text(yaml.safe_dump(belt))
    robot = robot_overlay(urdf_path, example['initial_joints'], calibration['gripper_tcp'])
    ET.ElementTree(robot).write(directory/'nero7.urdf', encoding='unicode')
    conversion = subprocess.run(['gz', 'sdf', '-p', str(directory/'nero7.urdf')],
                                text=True, capture_output=True)
    if conversion.returncode:
        raise RuntimeError('URDF -> SDF conversion failed: '+conversion.stderr)
    robot_sdf = ET.fromstring(conversion.stdout).find('model')
    if robot_sdf is None:
        raise ValueError('Gazebo failed to convert the hardware URDF')
    root = ET.Element('sdf', version='1.6'); world = child(root, 'world', name='nero7_example')
    child(world, 'gravity', '0 0 -9.81')
    physics = child(world, 'physics', name='ode', type='ode')
    child(physics, 'max_step_size', '.002')
    child(physics, 'real_time_update_rate', 500*example.get('real_time_factor', .25))
    light = child(world, 'light', name='sun', type='directional')
    child(light, 'pose', '0 0 3 0 0 0'); child(light, 'diffuse', '.85 .85 .85 1')
    child(light, 'direction', '-.3 .3 -1')
    scene = child(world, 'scene'); child(scene, 'ambient', '.55 .55 .55 1')
    child(scene, 'background', '.25 .28 .3 1')
    view = child(child(world, 'gui'), 'camera', name='overview')
    child(view, 'pose', '1.4 1.8 1.1 0 .46 -2.415')
    floor = child(world, 'model', name='floor'); child(floor, 'static', 'true')
    child(floor, 'pose', '0 0 -.4 0 0 0')
    geometry(child(floor, 'link', name='body'), 'floor', [3., 3., .04], '.35 .35 .38 1')
    direction = np.array(belt['belt_direction'], float); direction /= np.linalg.norm(direction)
    gate = np.array(belt['belt_origin'])+direction*belt['gate_s']
    conveyor = child(world, 'model', name='conveyor'); child(conveyor, 'static', 'true')
    child(conveyor, 'pose', numbers([*gate, belt['belt_height']-.025, 0, 0,
                                   math.atan2(direction[1], direction[0])]))
    geometry(child(conveyor, 'link', name='body'), 'belt', [1.4, .24, .05], '.06 .42 .15 1')
    world.append(robot_sdf)
    block = child(world, 'model', name='example_block')
    child(block, 'pose', '0 0 -2 0 0 0'); child(block, 'allow_auto_disable', 'false')
    link = child(block, 'link', name='body')
    inertial = child(link, 'inertial'); child(inertial, 'mass', '.08')
    inertia = child(inertial, 'inertia')
    for axis in ('ixx', 'iyy', 'izz'): child(inertia, axis, '.00004')
    geometry(link, 'block', example['object_size'], '.92 .88 .73 1')
    optical = np.array(calibration['camera_to_base_cv'], float)
    # Gazebo camera: X forward, Y left, Z up. ROS optical: Z forward, X right, Y down.
    camera_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    camera_rotation = Rotation.from_matrix(optical[:3, :3]@camera_from_optical.T)
    camera = child(world, 'model', name='camera'); child(camera, 'static', 'true')
    child(camera, 'pose', numbers([*optical[:3, 3], *camera_rotation.as_euler('xyz')]))
    sensor = child(child(camera, 'link', name='body'), 'sensor', name='rgbd', type='depth')
    child(sensor, 'always_on', 'true'); child(sensor, 'update_rate', example['camera_rate'])
    cam = child(sensor, 'camera')
    child(cam, 'horizontal_fov', 2*math.atan(example['camera_width']/(2*example['camera_fx'])))
    image = child(cam, 'image'); child(image, 'width', example['camera_width'])
    child(image, 'height', example['camera_height']); child(image, 'format', 'R8G8B8')
    clip = child(cam, 'clip'); child(clip, 'near', '.05'); child(clip, 'far', '3')
    plugin = child(sensor, 'plugin', name='rgbd_camera', filename='libgazebo_ros_camera.so')
    ros = child(plugin, 'ros'); child(ros, 'namespace', '/sim/camera')
    for original, target in [('image_raw', 'raw/color/image_raw'), ('camera_info', 'raw/color/camera_info'),
                             ('depth/image_raw', 'raw/depth/image_raw'), ('points', 'depth/points')]:
        child(ros, 'remapping', f'rgbd/{original}:=/sim/camera/{target}')
    child(plugin, 'camera_name', 'rgbd'); child(plugin, 'frame_name', 'camera_color_optical_frame')
    child(plugin, 'min_depth', '.05'); child(plugin, 'max_depth', '3')
    plugin = child(world, 'plugin', name='nero7_scene', filename='libnero7_scene.so')
    child(child(plugin, 'ros'), 'namespace', '/sim/scene')
    for name, values in [('initial_joints', example['initial_joints']),
                          ('tcp', calibration['gripper_tcp']), ('direction', direction), ('gate', gate),
                          ('object_size', example['object_size'])]: child(plugin, name, numbers(values))
    for name, value in [('belt_speed', example['belt_speed']), ('belt_height', belt['belt_height']),
                        ('spawn_distance', example['spawn_distance_upstream']),
                        ('object_yaw', example['object_yaw'])]: child(plugin, name, value)
    world_file = directory/'example.world'
    ET.ElementTree(root).write(world_file, encoding='unicode')
    return dict(world=str(world_file), urdf=str(directory/'nero7.urdf'),
                calibration=str(directory/'calibration.yaml'), belt=str(directory/'belt.yaml'))
