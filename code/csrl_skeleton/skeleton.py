"""75-keypoint skeleton: DFS graph ordering and TSSI image generation."""

from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

NUM_JOINTS = 75
LH0, RH0 = 33, 54

# ── Skeleton graph edges ──
_POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(17,19),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32),
]
_BRIDGE = [(0,11),(0,12),(0,9),(0,10)]
_FINGER = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),
]
_LH = [(a + LH0, b + LH0) for a, b in _FINGER]
_RH = [(a + RH0, b + RH0) for a, b in _FINGER]
_WRIST_HAND = [(15, LH0), (16, RH0)]

_ADJ = defaultdict(list)
for u, v in _POSE_EDGES + _BRIDGE + _LH + _RH + _WRIST_HAND:
    _ADJ[u].append(v)
    _ADJ[v].append(u)


def _dfs_order(root: int = 0, n: int = NUM_JOINTS) -> list[int]:
    order, seen = [], set()
    def dfs(u):
        seen.add(u)
        order.append(u)
        for w in sorted(_ADJ[u]):
            if w not in seen:
                dfs(w)
    dfs(root)
    for j in range(n):
        if j not in seen:
            order.append(j)
    return order


DFS_ORDER = _dfs_order()
assert len(DFS_ORDER) == NUM_JOINTS
JOINT2COL = {j: c for c, j in enumerate(DFS_ORDER)}

# ── Left/right mirror map ──
_mirror = np.arange(NUM_JOINTS)
for a, b in [
    (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
    (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32),
]:
    _mirror[a], _mirror[b] = b, a
for i in range(21):
    _mirror[LH0 + i], _mirror[RH0 + i] = RH0 + i, LH0 + i
COL_SWAP = np.array([JOINT2COL[_mirror[DFS_ORDER[c]]] for c in range(NUM_JOINTS)])

# Per-part normalisation ranges (body, left hand, right hand)
PARTS = [(0, 33), (33, 54), (54, 75)]


def generate_tssi_75(kp: np.ndarray) -> np.ndarray:
    """Convert (T, J, C) keypoints to (3, J, T) TSSI image with per-part normalisation."""
    kp = np.asarray(kp, dtype=np.float32)
    T, J, C = kp.shape
    x = kp[:, :, 0].copy()
    y = kp[:, :, 1].copy()
    conf = kp[:, :, 2].copy() if C >= 3 else np.ones((T, J), np.float32)

    x = gaussian_filter1d(x, 1.5, axis=0)
    y = gaussian_filter1d(y, 1.5, axis=0)

    for s, e in PARTS:
        e = min(e, J)
        for arr in (x, y):
            amin = arr[:, s:e].min()
            amax = arr[:, s:e].max()
            arr[:, s:e] = (arr[:, s:e] - amin) / max(amax - amin, 1e-6)

    tssi = np.zeros((3, J, T), np.float32)
    for col, j in enumerate(DFS_ORDER):
        tssi[0, col] = np.clip(x[:, j], 0, 1)
        tssi[1, col] = np.clip(y[:, j], 0, 1)
        tssi[2, col] = np.clip(conf[:, j], 0, 1)
    return tssi
