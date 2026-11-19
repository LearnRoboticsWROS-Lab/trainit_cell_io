"""The MOCK plant's cell process: `ros2 launch trainit_cell_io cell_io.launch.py config:=<cell_io.yaml>`.

A cell-level process (P0B_DESIGN.md §4), started apart from the robot: the robot bundle never launches it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', description='the cell\'s machine-simulation file (cell_io.yaml, plant: mock)'),
        Node(
            package='trainit_cell_io',
            executable='cell_io_node',
            name='cell_io',
            output='screen',
            emulate_tty=True,
            parameters=[{'config': LaunchConfiguration('config')}],
        ),
    ])
