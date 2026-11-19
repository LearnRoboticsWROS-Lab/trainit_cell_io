"""The QoS of the cell's topics (P0B_DESIGN.md §3.2 / §4)."""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# The canonical handshake images and the latched operator view: RELIABLE, TRANSIENT_LOCAL, depth 1 (level-based).
LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

# The source's READER of `<service>/request` (and the bridge's): the same contract QoS, a deeper queue on the reader
# side only (depth is not a matching policy). Each writer stays level-held at depth 1; the reader keeps one image per
# writer until it is taken, so two requesters' images that arrive within one executor spin both reach the cell, in
# order, instead of the later replacing the earlier unseen (implementation review fix C2).
REQUEST_READER = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)

# Operator commands: RELIABLE, VOLATILE -- a command is never replayed to a node that starts later.
COMMANDS = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)

CONTROL_TOPIC = '/cell_io/control'
STATE_TOPIC = '/cell_io/state'
