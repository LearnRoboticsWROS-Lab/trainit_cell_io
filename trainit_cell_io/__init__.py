"""trainit_cell_io -- simulation control of a cell's machines for TrainIt (P0.b; ADR-0013 Amendment P0).

The machine side of a machine service in SIMULATION: devices, station sequences (the `python_logic` backend), one
`ProcessDevice` per service implementing the source's rules of the machine-service contract, and a plant adapter
(the deterministic mock, or an Isaac Sim articulation). A cell-level process, started apart from the robot (the robot
bundle never launches it). Not a PLC, not a safety function (ADR-0033).
"""

__version__ = '0.1.0'
