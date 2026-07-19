"""Keypoint-layout dispatcher.

Selects the skeleton implementation via the KP_LAYOUT environment variable:
    "coco133"     (default) — RTMW/DWPose 133 whole-body, signing subset (55 joints)
    "mediapipe75" — legacy MediaPipe 75-kp (67 joints after dropping the legs)

The rest of the codebase imports NUM_JOINTS, DFS_ORDER, COL_SWAP, ADJACENCY and
generate_tssi_75 from here, unaware of which layout is active.
"""

import os

LAYOUT = os.environ.get("KP_LAYOUT", "coco133")

if LAYOUT == "mediapipe75":
    from skeleton_mediapipe75 import (  # noqa: F401
        NUM_JOINTS, DFS_ORDER, JOINT2COL, COL_SWAP, PARTS, ADJACENCY,
        generate_tssi_75,
    )
else:
    from skeleton_coco133 import (  # noqa: F401
        NUM_JOINTS, DFS_ORDER, JOINT2COL, COL_SWAP, PARTS, ADJACENCY,
        generate_tssi_75,
    )
