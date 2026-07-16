import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'charuco_lidar_calib'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='race',
    maintainer_email='shchon724@gmail.com',
    description='ZED2i <-> RS-16 extrinsic calibration with an 8x7 ChArUco board.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Offline calibration over captured png+pcd pairs
            'calibrate = charuco_lidar_calib.calibrate:main',
            # Reproject LiDAR onto the image using a solved extrinsic (visual check)
            'verify = charuco_lidar_calib.verify:main',
            # Paint the camera image onto the LiDAR cloud -> colored XYZRGB pcd
            'colorize = charuco_lidar_calib.colorize:main',
            # Publish the solved extrinsic as a static TF (live)
            'tf_publisher = charuco_lidar_calib.tf_publisher_node:main',
            # Identify the ArUco dictionary of a board from an image
            'dict_sniffer = charuco_lidar_calib.dict_sniffer:main',
            # Grab camera_info from a live topic into a yaml
            'grab_camera_info = charuco_lidar_calib.grab_camera_info:main',
        ],
    },
)
