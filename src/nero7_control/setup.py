from setuptools import find_packages, setup

package_name = 'nero7_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/controller.yaml', 'config/follow.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu22',
    maintainer_email='ubuntu22@todo.todo',
    description='Arm control, GR-ConvNet inference and grasp state machine for Nero7.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grasp_controller = nero7_control.grasp_controller:main',
            'arm_test = nero7_control.scripts.arm_test:main',
            'gripper_calib = nero7_control.scripts.gripper_calib:main',
            'teach_mode = nero7_control.scripts.teach_mode:main',
            'lead_calib = nero7_control.scripts.lead_calib:main',
            'benchmark_models = nero7_control.scripts.benchmark_models:main',
            'static_grasp = nero7_control.scripts.static_grasp:main',
        ],
    },
)
