from setuptools import find_packages, setup

package_name = 'nero7_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/perception.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu22',
    maintainer_email='ubuntu22@todo.todo',
    description='Block detection, tracking and belt speed estimation for Nero7 conveyor grasping.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'block_detector = nero7_perception.block_detector:main',
            'depth_center_plan = nero7_perception.scripts.depth_center_plan:main',
            'handeye_collect = nero7_perception.scripts.handeye_collect:main',
            'handeye_solve = nero7_perception.scripts.handeye_solve:main',
            'touch_calib = nero7_perception.scripts.touch_calib:main',
            'drag_calib = nero7_perception.scripts.drag_calib:main',
            'verify_calib = nero7_perception.scripts.verify_calib:main',
            'belt_calib = nero7_perception.scripts.belt_calib:main',
            'record_session = nero7_perception.scripts.record_session:main',
            'capture_rgbd = nero7_perception.scripts.capture_rgbd:main',
            'perception_validate = nero7_perception.scripts.perception_validate:main',
            'block_detector_2d = nero7_perception.block_detector_2d:main',
        ],
    },
)
