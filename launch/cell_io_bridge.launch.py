"""The ROS side of the Isaac plant: `ros2 launch trainit_cell_io cell_io_bridge.launch.py config:=<cell_io.yaml>`.

A cell-level process (P0B_DESIGN.md §4), started apart from the robot, beside the Isaac stage runner (which runs the
station logic inside Isaac Sim and speaks std_msgs JSON): it translates to and from the canonical ChannelRequest /
ChannelStatus in the robot's slot. It never mints a heartbeat. The robot bundle never launches it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', description="the cell's machine-simulation file (cell_io.yaml, plant: isaac)"),
        Node(
            package='trainit_cell_io',
            executable='cell_io_bridge',
            name='cell_io_bridge',
            output='screen',
            emulate_tty=True,
            parameters=[{'config': LaunchConfiguration('config')}],
        ),
    ])
