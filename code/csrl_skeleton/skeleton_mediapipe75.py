"""Skeleton graph ordering and TSSI image generation.

Legs are dropped (they carry no lexical information and only widen the vertical
range, compressing the arms and hands), missing joints are temporally
interpolated, and coordinates are body-centered with an isotropic scale so hand
shape is preserved and signer position/size are normalised out.

The stored keypoints are always the full 75-joint MediaPipe layout
(0-32 body, 33-53 left hand, 54-74 right hand); the leg drop happens here, at
image-build time, so the prepared pickles do not need to be regenerated.
"""

from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

NUM_JOINTS_FULL = 75
LH0_FULL, RH0_FULL = 33, 54

# MediaPipe Pose legs: knees (25,26), ankles (27,28), heels (29,30), foot index
# (31,32). Dropped; hips (23,24) are kept as a stable torso anchor.
DROP_JOINTS = {25, 26, 27, 28, 29, 30, 31, 32}
KEEP = [j for j in range(NUM_JOINTS_FULL) if j not in DROP_JOINTS]
NUM_JOINTS = len(KEEP)                                  # 67
OLD2NEW = {old: new for new, old in enumerate(KEEP)}
LH0, RH0 = OLD2NEW[LH0_FULL], OLD2NEW[RH0_FULL]         # 25, 46

# ── Skeleton graph edges (defined in FULL 75-joint indices) ──
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
_LH = [(a + LH0_FULL, b + LH0_FULL) for a, b in _FINGER]
_RH = [(a + RH0_FULL, b + RH0_FULL) for a, b in _FINGER]
_WRIST_HAND = [(15, LH0_FULL), (16, RH0_FULL)]

# Build adjacency over the KEPT joints only: drop any edge that touches a removed
# joint and remap the survivors to the compact 0..NUM_JOINTS-1 index space.
_ADJ = defaultdict(list)
for u, v in _POSE_EDGES + _BRIDGE + _LH + _RH + _WRIST_HAND:
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

# ── Left/right mirror map (FULL index pairs, remapped to kept indices) ──
_MIRROR_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
    (17,18),(19,20),(21,22),(23,24),
]
_mirror = np.arange(NUM_JOINTS)
for a, b in _MIRROR_PAIRS:
    if a in OLD2NEW and b in OLD2NEW:
        na, nb = OLD2NEW[a], OLD2NEW[b]
        _mirror[na], _mirror[nb] = nb, na
for i in range(21):
    la, ra = LH0_FULL + i, RH0_FULL + i
    if la in OLD2NEW and ra in OLD2NEW:
        nla, nra = OLD2NEW[la], OLD2NEW[ra]
        _mirror[nla], _mirror[nra] = nra, nla
COL_SWAP = np.array([JOINT2COL[_mirror[DFS_ORDER[c]]] for c in range(NUM_JOINTS)])

# Kept-index part boundaries (body, left hand, right hand), for reference.
PARTS = [(0, LH0), (LH0, RH0), (RH0, NUM_JOINTS)]

# Normalisation field of view: half-width of the [0,1] box, in shoulder widths.
_FOV_SHOULDERS = 2.5


def _normalized_adjacency() -> np.ndarray:
    """Symmetric-normalised adjacency D^-1/2 (A + I) D^-1/2 for a spatial
    graph-convolution front-end. Indexed in TSSI column order (the DFS layout
    used by generate_tssi_75), so it lines up with the J axis of the image."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for u, nbrs in _ADJ.items():
        cu = JOINT2COL[u]
        for v in nbrs:
            A[cu, JOINT2COL[v]] = 1.0
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-6))
    return (dinv[:, None] * A) * dinv[None, :]


ADJACENCY = _normalized_adjacency()   # (NUM_JOINTS, NUM_JOINTS), DFS column order


def _fill_missing(coord: np.ndarray, conf: np.ndarray) -> None:
    """Linear temporal interpolation over frames where a joint is missing
    (confidence == 0), in place. Joints missing in every frame are left as-is."""
    T, J = coord.shape
    t = np.arange(T)
    for j in range(J):
        valid = conf[:, j] > 0
        if valid.all() or not valid.any():
            continue
        coord[:, j] = np.interp(t, t[valid], coord[valid, j])


def generate_tssi_75(kp: np.ndarray) -> np.ndarray:
    """Convert full (T, 75, C) MediaPipe keypoints to a (3, NUM_JOINTS, T) TSSI image.

    Steps: fill missing joints by temporal interpolation, smooth, body-center the
    coordinates (origin at the shoulder midpoint, isotropic scale = shoulder width),
    then drop the legs and lay out the kept joints in DFS order. The function name is
    kept for API compatibility; it now emits NUM_JOINTS (= 67) rows, not 75.
    """
    kp = np.asarray(kp, dtype=np.float32)
    T, J, C = kp.shape
    x = kp[:, :, 0].copy()
    y = kp[:, :, 1].copy()
    conf = kp[:, :, 2].copy() if C >= 3 else np.ones((T, J), np.float32)

    # 1. fill missing keypoints so they do not corrupt the normalisation
    _fill_missing(x, conf)
    _fill_missing(y, conf)

    # 2. temporal smoothing
    x = gaussian_filter1d(x, 1.5, axis=0)
    y = gaussian_filter1d(y, 1.5, axis=0)

    # 3. body-centered, isotropic normalisation (uses full-index shoulders 11/12);
    #    a single scale on x and y preserves hand-shape aspect ratio.
    cx = (x[:, 11] + x[:, 12]) * 0.5
    cy = (y[:, 11] + y[:, 12]) * 0.5
    sw = np.sqrt((x[:, 11] - x[:, 12]) ** 2 + (y[:, 11] - y[:, 12]) ** 2)
    denom = max(float(np.median(sw)), 1e-3) * 2.0 * _FOV_SHOULDERS
    x = (x - cx[:, None]) / denom + 0.5
    y = (y - cy[:, None]) / denom + 0.5

    # 4. select kept joints in DFS order and build the image
    tssi = np.zeros((3, NUM_JOINTS, T), np.float32)
    for col, jnew in enumerate(DFS_ORDER):
        jold = KEEP[jnew]
        tssi[0, col] = np.clip(x[:, jold], 0, 1)
        tssi[1, col] = np.clip(y[:, jold], 0, 1)
        tssi[2, col] = np.clip(conf[:, jold], 0, 1)
    return tssi
