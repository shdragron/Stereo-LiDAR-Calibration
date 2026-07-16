import os
from glob import glob
from setuptools import setup

package_name = 'fsg_sensors'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'extrinsics'),
         glob('config/extrinsics/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='race',
    maintainer_email='shchon724@gmail.com',
    description='FSG driverless sensor bringup (/sensors namespace) + preflight check.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'preflight = fsg_sensors.preflight:main',
            'zed_relay = fsg_sensors.zed_relay:main',
        ],
    },
)
