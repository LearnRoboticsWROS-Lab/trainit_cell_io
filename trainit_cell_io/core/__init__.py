"""trainit_cell_io's core: simulation control without ROS (P0B_DESIGN.md §4).

Importable by any Python 3.10+ (the ROS nodes, a simulator's own interpreter, a unit test) -- it imports no rclpy and
no simulator module (case C1). Its only third-party dependency is PyYAML (the cell file).
"""

from .cell import Cell, ControlError, SequenceFactory
from .clock import ClockError, ScanClock, parse_dt
from .config import CellConfig, ConfigError, load, parse
from .contract import (FAULT_ACTUATOR_TIMEOUT, FAULT_NAMES, FAULT_NONE, FAULT_SENSOR_FAULT, FAULT_STATION_FAULT,
                       REJECT_BUSY, REJECT_FAULTED, REJECT_NAMES, REJECT_NONE, REJECT_TWO_REQUESTERS, STATE_BUSY,
                       STATE_DONE, STATE_FAULT, STATE_HOMING, STATE_NAMES, STATE_NONE, STATE_READY, RequestImage,
                       StatusImage)
from .logic import DONE, RUNNING, Fault, Sequence
from .mock_plant import MockPlantAdapter
from .plant import InjectionError, PlantAdapter
from .process_device import ProcessDevice, new_session

__all__ = [
    'Cell', 'ControlError', 'SequenceFactory', 'Sequence', 'Fault', 'DONE', 'RUNNING', 'ClockError', 'ScanClock',
    'parse_dt', 'CellConfig', 'ConfigError', 'load', 'parse',
    'FAULT_ACTUATOR_TIMEOUT', 'FAULT_NAMES', 'FAULT_NONE', 'FAULT_SENSOR_FAULT', 'FAULT_STATION_FAULT',
    'REJECT_BUSY', 'REJECT_FAULTED', 'REJECT_NAMES', 'REJECT_NONE', 'REJECT_TWO_REQUESTERS', 'STATE_BUSY',
    'STATE_DONE', 'STATE_FAULT', 'STATE_HOMING', 'STATE_NAMES', 'STATE_NONE', 'STATE_READY', 'RequestImage',
    'StatusImage', 'MockPlantAdapter', 'InjectionError', 'PlantAdapter', 'ProcessDevice', 'new_session',
]
