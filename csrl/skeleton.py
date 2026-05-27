"""
skeleton.py — Ordine DFS dello scheletro MediaPipe (48 keypoint) e
              generazione TSSI (Temporal Skeleton Sequence Image).

DFS_ORDER   : np.ndarray shape (135,) con indici di joint in ordine DFS
DFS_UNIQUE  : list dei joint unici nello stesso ordine
EXPECTED_JOINTS : numero atteso di joint (48)
"""

import numpy as np
import cv2
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


EXPECTED_JOINTS = 48


# ============================================================
# DFS ORDER
# ============================================================

def get_dfs_order_phoenix_correct() -> np.ndarray:
    """
    Costruisce l'ordine DFS corretto per lo scheletro MediaPipe a 48 keypoint.

    Layout keypoint:
        0-5   → corpo (6 joint)
        6-26  → mano sinistra (21 joint)
        27-47 → mano destra  (21 joint)

    Returns:
        np.ndarray shape (135,) dtype int32
    """
    adj = defaultdict(list)

    edges_body = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
    edges_wrist_to_hand = [(3, 6), (5, 27)]
    finger_edges = [
        (0, 1), (1, 2), (2, 3),  (3, 4),
        (0, 5), (5, 6), (6, 7),  (7, 8),
        (0, 9), (9, 10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    edges_lh = [(s + 6,  e + 6)  for s, e in finger_edges]
    edges_rh = [(s + 27, e + 27) for s, e in finger_edges]

    for u, v in edges_body + edges_wrist_to_hand + edges_lh + edges_rh:
        if u < 48 and v < 48:
            adj[u].append(v)
            adj[v].append(u)

    dfs_path, visited = [], set()

    def dfs(node):
        dfs_path.append(node)
        visited.add(node)
        for n in sorted([x for x in adj[node] if x not in visited]):
            dfs(n)
            if len(dfs_path) < 135:
                dfs_path.append(node)

    dfs(0)
    while len(dfs_path) < 135:
        dfs_path.extend(dfs_path[-10:])

    return np.array(dfs_path[:135], dtype=np.int32)


# Inizializzato a livello modulo: importalo dove serve
DFS_ORDER  = get_dfs_order_phoenix_correct()
DFS_UNIQUE = list(dict.fromkeys(DFS_ORDER))


# ============================================================
# TSSI GENERATION
# ============================================================

def generate_tssi_optimized(keypoints: np.ndarray, frame_h: int = None):
    """
    Converte una sequenza di keypoints in una TSSI di shape (3, n_joints, T).

    Pipeline:
        1. Separazione canali x, y, conf + gestione confidence assente
        2. Interpolazione keypoint mancanti (conf < 0.3)
        3. Temporal smoothing gaussiano (sigma=1.5)
        4. Per-part normalization (body / left hand / right hand separati)
        5. Resize temporale opzionale a frame_h
        6. Riordino joints in DFS order

    Args:
        keypoints : np.ndarray shape (T, J, 2|3)
                    Canali: x, y  oppure  x, y, confidence.
        frame_h   : int | None
                    Se fornito, resize bilineare a questa lunghezza temporale.
                    Se None, usa la lunghezza originale (consigliato per CTC
                    con padding variabile).

    Returns:
        tssi  : np.ndarray shape (3, n_unique_joints, T_out)
        T_out : int  lunghezza temporale effettiva
    """
    try:
        kp      = keypoints.astype(np.float32)
        T_raw, J, C = kp.shape

        # 0. Separa canali; gestisci assenza confidence
        x    = kp[:, :, 0].copy()
        y    = kp[:, :, 1].copy()
        conf = kp[:, :, 2].copy() if C >= 3 else np.ones((T_raw, J), dtype=np.float32)

        # 1. Interpolazione keypoint mancanti
        for j in range(J):
            valid_idx = np.where(conf[:, j] > 0.3)[0]
            if len(valid_idx) >= 2:
                t_all      = np.arange(T_raw)
                x[:, j]    = np.interp(t_all, valid_idx, x[valid_idx, j])
                y[:, j]    = np.interp(t_all, valid_idx, y[valid_idx, j])
                conf[:, j] = np.interp(t_all, valid_idx, conf[valid_idx, j])
            elif len(valid_idx) == 0:
                conf[:, j] = 0.0

        # 2. Temporal smoothing
        x    = gaussian_filter1d(x,    sigma=1.5, axis=0)
        y    = gaussian_filter1d(y,    sigma=1.5, axis=0)
        conf = gaussian_filter1d(conf, sigma=1.5, axis=0)

        # 3. Per-part normalization (body / left hand / right hand)
        for start, end in [(0, 6), (6, 27), (27, 48)]:
            end = min(end, J)
            if end <= start:
                continue
            for arr in [x, y]:
                a_min = arr[:, start:end].min()
                a_max = arr[:, start:end].max()
                rng   = max(a_max - a_min, 1e-6)
                arr[:, start:end] = (arr[:, start:end] - a_min) / rng

        # 4. Resize temporale opzionale
        if frame_h is not None:
            def _resize_temporal(arr_2d, target_h):
                if arr_2d.shape[0] == 1:
                    return np.repeat(arr_2d, target_h, axis=0)[:target_h]
                return cv2.resize(
                    arr_2d.astype(np.float32),
                    (J, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            x    = _resize_temporal(x,    frame_h)
            y    = _resize_temporal(y,    frame_h)
            conf = _resize_temporal(conf, frame_h)
            T_out = frame_h
        else:
            T_out = T_raw

        # 5. Riordina joints in DFS order
        dfs_unique = list(dict.fromkeys(DFS_ORDER))
        n_unique   = len(dfs_unique)

        # 6. Costruisci TSSI
        tssi = np.zeros((3, n_unique, T_out), dtype=np.float32)
        for col_idx, joint_id in enumerate(dfs_unique):
            if joint_id >= J:
                continue
            tssi[0, col_idx, :] = np.clip(x[:,    joint_id], 0.0, 1.0)
            tssi[1, col_idx, :] = np.clip(y[:,    joint_id], 0.0, 1.0)
            tssi[2, col_idx, :] = np.clip(conf[:, joint_id], 0.0, 1.0)

        return tssi, T_out

    except Exception as e:
        import traceback
        print(f"[generate_tssi_optimized] Errore: {e}")
        traceback.print_exc()
        fallback_h = frame_h if frame_h is not None else (T_raw if "T_raw" in dir() else 1)
        n_joints   = len(list(dict.fromkeys(DFS_ORDER)))
        return np.zeros((3, n_joints, fallback_h), dtype=np.float32), 1
