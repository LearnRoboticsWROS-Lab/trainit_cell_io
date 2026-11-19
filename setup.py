import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'trainit_cell_io'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'examples', 'neutral_station'),
         glob('examples/neutral_station/*.yaml') + glob('examples/neutral_station/*.py')
         + glob('examples/neutral_station/*.md')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='fra',
    maintainer_email='ros.master.ai@gmail.com',
    description='TrainIt simulation control of a cell\'s machines (P0.b)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cell_io_node = trainit_cell_io.ros.cell_io_node:main',
            'cell_io_bridge = trainit_cell_io.ros.cell_io_bridge:main',
            'cell_io_cli = trainit_cell_io.ros.cli:main',
        ],
    },
)
