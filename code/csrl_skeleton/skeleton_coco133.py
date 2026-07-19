"""Skeleton graph + TSSI generation for COCO-WholeBody 133 keypoints (RTMW/DWPose).

Input keypoints are the full 133-joint COCO-WholeBody layout:
    0-16   body (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
    17-22  feet
    23-90  face (68)
    91-111 left hand (21: wrist + 5 fingers x 4)
    112-132 right hand (21)

For continuous sign recognition we keep the "signing" subset only — the upper
body plus both hands (55 joints) — and drop legs, feet and the dense face. The
face could be added later (mouth/eyebrows) for non-manual markers.

Coordinates are body-centered (origin = shoulder midpoint, isotropic scale =
shoulder width) so hand shape is preserved and signer position/size are
normalised out. The third channel is a [0,1]-squashed keypoint confidence.

Public API matches skeleton_mediapipe75: NUM_JOINTS, DFS_ORDER, COL_SWAP,
ADJACENCY, generate_tssi_75(kp).
"""

from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

NUM_JOINTS_FULL = 133

# ── Signing subset (full-133 indices) ──
_BODY = list(range(0, 13))          # 0-12: head landmarks, shoulders, elbows, wrists, hips
_LH_FULL = list(range(91, 112))     # 91-111: left hand
_RH_FULL = list(range(112, 133))    # 112-132: right hand
KEEP = _BODY + _LH_FULL + _RH_FULL  # 55 joints
NUM_JOINTS = len(KEEP)
OLD2NEW = {old: new for new, old in enumerate(KEEP)}
LH0, RH0 = OLD2NEW[91], OLD2NEW[112]   # 13, 34

# ── Skeleton graph edges (full-133 indices) ──
_BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # nose-eyes-ears
    (0, 5), (0, 6),                          # nose-shoulders (connectivity bridge)
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12), (11, 12),              # torso to hips
]
_FINGER = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
    (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
    (18, 19), (19, 20),
]
_LH_EDGES = [(a + 91, b + 91) for a, b in _FINGER]
_RH_EDGES = [(a + 112, b + 112) for a, b in _FINGER]
_WRIST_HAND = [(9, 91), (10, 112)]           # body wrist -> hand root

_ADJ = defaultdict(list)
for u, v in _BODY_EDGES + _LH_EDGES + _RH_EDGES + _WRIST_HAND:
    if u in OLD2NEW and v in OLD2NEW:
        nu, nv = OLD2NEW[u], OLD2NEW[v]
        _ADJ[nu].append(nv)
        _ADJ[nv].append(nu)


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

# ── Left/right mirror map (full-133 pairs, remapped to kept indices) ──
_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
_mirror = np.arange(NUM_JOINTS)
for a, b in _MIRROR_PAIRS:
    if a in OLD2NEW and b in OLD2NEW:
        na, nb = OLD2NEW[a], OLD2NEW[b]
        _mirror[na], _mirror[nb] = nb, na
for i in range(21):                          # swap left/right hand joints
    la, ra = 91 + i, 112 + i
    if la in OLD2NEW and ra in OLD2NEW:
        nla, nra = OLD2NEW[la], OLD2NEW[ra]
        _mirror[nla], _mirror[nra] = nra, nla
COL_SWAP = np.array([JOINT2COL[_mirror[DFS_ORDER[c]]] for c in range(NUM_JOINTS)])

PARTS = [(0, LH0), (LH0, RH0), (RH0, NUM_JOINTS)]   # body, left hand, right hand

_FOV_SHOULDERS = 2.5     # half-width of the [0,1] box, in shoulder widths
_SCORE_NORM = 6.0        # RTMW score ~median; squashes the confidence channel to [0,1]
# full-133 shoulder indices, used for the normalisation anchor
_LSH, _RSH = 5, 6


def _normalized_adjacency() -> np.ndarray:
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for u, nbrs in _ADJ.items():
        cu = JOINT2COL[u]
        for v in nbrs:
            A[cu, JOINT2COL[v]] = 1.0
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-6))
    return (dinv[:, None] * A) * dinv[None, :]


ADJACENCY = _normalized_adjacency()


def _fill_missing(coord: np.ndarray, valid: np.ndarray) -> None:
    """Temporal linear interpolation over frames where a joint is low-confidence."""
    T, J = coord.shape
    t = np.arange(T)
    for j in range(J):
        v = valid[:, j]
        if v.all() or not v.any():
            continue
        coord[:, j] = np.interp(t, t[v], coord[v, j])


def generate_tssi_75(kp: np.ndarray) -> np.ndarray:
    """Full (T, 133, C>=2) COCO-WholeBody keypoints -> (3, NUM_JOINTS, T) TSSI image.

    Name kept for API compatibility with the MediaPipe module; emits NUM_JOINTS
    (= 55) rows.
    """
    kp = np.asarray(kp, dtype=np.float32)
    T, J, C = kp.shape
    x = kp[:, :, 0].copy()
    y = kp[:, :, 1].copy()
    score = kp[:, :, 2].copy() if C >= 3 else np.full((T, J), _SCORE_NORM, np.float32)

    # low-confidence points get temporally interpolated (RTMW misses <1% of the time)
    valid = score > (0.25 * _SCORE_NORM)
    _fill_missing(x, valid)
    _fill_missing(y, valid)

    x = gaussian_filter1d(x, 1.5, axis=0)
    y = gaussian_filter1d(y, 1.5, axis=0)

    # body-centered isotropic normalisation (full-index shoulders)
    cx = (x[:, _LSH] + x[:, _RSH]) * 0.5
    cy = (y[:, _LSH] + y[:, _RSH]) * 0.5
    sw = np.sqrt((x[:, _LSH] - x[:, _RSH]) ** 2 + (y[:, _LSH] - y[:, _RSH]) ** 2)
    denom = max(float(np.median(sw)), 1e-3) * 2.0 * _FOV_SHOULDERS
    x = (x - cx[:, None]) / denom + 0.5
    y = (y - cy[:, None]) / denom + 0.5

    conf = np.clip(score / _SCORE_NORM, 0.0, 1.0)

    tssi = np.zeros((3, NUM_JOINTS, T), np.float32)
    for col, jnew in enumerate(DFS_ORDER):
        jold = KEEP[jnew]
        tssi[0, col] = np.clip(x[:, jold], 0, 1)
        tssi[1, col] = np.clip(y[:, jold], 0, 1)
        tssi[2, col] = conf[:, jold]
    return tssi
