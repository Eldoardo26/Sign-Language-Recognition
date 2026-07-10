# -*- coding: utf-8 -*-
"""
extract_skeleton_feats.py — estrae le feature dell'encoder del modello SKELETON (teacher)
per la cross-modal distillation verso il Signformer (student).

Per ogni video salva la sequenza di feature pre-classificatore (B,T,512) del teacher pose.
Output: skeleton_feats/{train,dev,test}.pkl = {nome_video: np.float16 (T, 512)}

Uso (env: sign-language-dnn, dalla cartella MSKA-SLR):
    python extract_skeleton_feats.py
"""
import os, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIG
# ============================================================
# Repo layout: phoenix/code/distillation/extract_skeleton_feats.py
# Defaults are relative to the top-level "phoenix" folder; override any of them
# with the corresponding environment variable if your data lives elsewhere.
#   PHOENIX_ROOT       : top-level phoenix folder
#   MSKA_DATA_DIR      : folder with the MSKA-format pickles Phoenix-2014T.{split}
#   TEACHER_CKPT       : trained skeleton teacher checkpoint (tssi75_cslr_best.pt)
#   SKELETON_FEATS_DIR : output folder for the extracted teacher features
PHOENIX_ROOT = Path(os.environ.get('PHOENIX_ROOT', Path(__file__).resolve().parents[2]))
DATA_DIR   = os.environ.get('MSKA_DATA_DIR',
                            str(PHOENIX_ROOT / 'code' / 'csrl_skeleton' / 'data' / 'Phoenix-2014T'))
CKPT_PATH  = os.environ.get('TEACHER_CKPT',
                            str(PHOENIX_ROOT / 'dataset' / 'checkpoints' / 'tssi75_cslr_best.pt'))
OUT_DIR    = os.environ.get('SKELETON_FEATS_DIR',
                            str(PHOENIX_ROOT / 'dataset' / 'features' / 'skeleton_feats'))
MAX_FRAMES = 400
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SCHELETRO 75 KEYPOINT + TSSI  (identico al notebook tssi75_cslr)
# ============================================================
NUM_JOINTS = 75; LH0, RH0 = 33, 54
_pose_edges = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(17,19),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32)]
_bridge = [(0,11),(0,12),(0,9),(0,10)]
_finger = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
_adj = defaultdict(list)
for u, v in _pose_edges + _bridge + [(a+LH0,b+LH0) for a,b in _finger] + \
            [(a+RH0,b+RH0) for a,b in _finger] + [(15,LH0),(16,RH0)]:
    _adj[u].append(v); _adj[v].append(u)
def _dfs(root=0, n=NUM_JOINTS):
    order, seen = [], set()
    def go(u):
        seen.add(u); order.append(u)
        for w in sorted(_adj[u]):
            if w not in seen: go(w)
    go(root)
    for j in range(n):
        if j not in seen: order.append(j)
    return order
DFS_ORDER = _dfs()
PARTS = [(0,33),(33,54),(54,75)]

def generate_tssi_75(kp):
    kp = np.asarray(kp, dtype=np.float32); T, J, C = kp.shape
    x = kp[:,:,0].copy(); y = kp[:,:,1].copy()
    conf = kp[:,:,2].copy() if C >= 3 else np.ones((T,J), np.float32)
    x = gaussian_filter1d(x, 1.5, axis=0); y = gaussian_filter1d(y, 1.5, axis=0)
    for s, e in PARTS:
        e = min(e, J)
        for arr in (x, y):
            amin = arr[:,s:e].min(); amax = arr[:,s:e].max()
            arr[:,s:e] = (arr[:,s:e]-amin) / max(amax-amin, 1e-6)
    tssi = np.zeros((3, J, T), np.float32)
    for col, j in enumerate(DFS_ORDER):
        tssi[0,col] = np.clip(x[:,j],0,1); tssi[1,col] = np.clip(y[:,j],0,1); tssi[2,col] = np.clip(conf[:,j],0,1)
    return tssi

# ============================================================
# MODELLO (identico al notebook) — serve solo per caricare i pesi ed estrarre la feature
# ============================================================
class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x): return x
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        a, _ = self.attn(x, x, x); return self.norm(x + self.drop(a))
class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k=3, dilation=1, dropout=0.2, drop_path=0.0):
        super().__init__()
        pad = dilation*(k-1)//2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout),
            nn.ReLU(True), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout))
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))
                     if cin != cout else nn.Identity())
        self.dp = DropPath(drop_path)
    def forward(self, x): return F.relu(self.dp(self.net(x)) + self.skip(x))
class PoseNetworkCTC(nn.Module):
    def __init__(self, num_classes, in_channels=225, hidden_dim=256, tcn_blocks=3,
                 lstm_layers=3, dropout=0.3, drop_path_rate=0.1, attn_heads=4):
        super().__init__()
        mid = max(1, tcn_blocks // 2)
        dp = [drop_path_rate*i/max(tcn_blocks-1,1) for i in range(tcn_blocks)]
        first = [TCNBlock(in_channels, hidden_dim, 5, 1, dropout, dp[0])]
        for i in range(1, mid):
            first.append(TCNBlock(hidden_dim, hidden_dim, 3, min(2**(i-1),8), dropout, dp[i]))
        self.tcn_first = nn.Sequential(*first)
        second = [TCNBlock(hidden_dim, hidden_dim, 3, min(2**(i-1),8), dropout, dp[i]) for i in range(mid, tcn_blocks)]
        self.tcn_second = nn.Sequential(*second) if second else nn.Identity()
        self.temporal_attn = TemporalAttention(hidden_dim, attn_heads, dropout*0.5)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if lstm_layers>1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim*2); self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.aux_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                      nn.LayerNorm(hidden_dim*2), nn.Dropout(dropout*0.5))
    def forward_feat(self, x):
        """Ritorna la feature pre-classificatore (B, T, hidden*2) — quella da distillare."""
        B, C, J, T = x.shape
        xf = x.reshape(B, C*J, T)
        fm = self.tcn_first(xf).permute(0, 2, 1)
        feat = self.tcn_second(fm.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.temporal_attn(feat)
        feat, _ = self.bilstm(feat)
        feat = self.norm(feat)          # (B, T, hidden*2)  <-- feature teacher
        return feat

# ============================================================
# LOAD CHECKPOINT
# ============================================================
ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
sd = ck['model'] if 'model' in ck else ck
cfg = ck.get('cfg', {})
num_classes = sd['fc.weight'].shape[0]
hidden = sd['fc.weight'].shape[1] // 2
model = PoseNetworkCTC(num_classes=num_classes, in_channels=NUM_JOINTS*3,
                       hidden_dim=cfg.get('hidden_dim', hidden),
                       tcn_blocks=cfg.get('tcn_blocks', 3),
                       lstm_layers=cfg.get('num_layers', 3),
                       dropout=cfg.get('dropout', 0.3),
                       drop_path_rate=cfg.get('drop_path_rate', 0.1),
                       attn_heads=cfg.get('attn_heads', 4)).to(DEVICE)
model.load_state_dict(sd); model.eval()
print(f'teacher caricato | num_classes={num_classes} | feat_dim={hidden*2} | dev={DEVICE}')

# ============================================================
# ESTRAZIONE
# ============================================================
@torch.no_grad()
def extract_split(split):
    raw = pickle.load(open(os.path.join(DATA_DIR, f'Phoenix-2014T.{split}'), 'rb'))
    feats = {}
    for i, (name, s) in enumerate(raw.items()):
        kp = np.asarray(s['keypoint'], dtype=np.float32)       # (T,75,3)
        T = kp.shape[0]
        if T > MAX_FRAMES:
            sel = np.linspace(0, T-1, MAX_FRAMES).round().astype(int); kp = kp[sel]
        tssi = generate_tssi_75(kp)                            # (3,75,T)
        x = torch.from_numpy(tssi).unsqueeze(0).float().to(DEVICE)  # (1,3,75,T)
        f = model.forward_feat(x)[0].cpu().numpy().astype(np.float16)  # (T, 512)
        feats[name] = f
        if (i+1) % 500 == 0: print(f'  [{split}] {i+1}/{len(raw)}')
    out = os.path.join(OUT_DIR, f'{split}.pkl')
    with open(out, 'wb') as fp: pickle.dump(feats, fp)
    print(f'[{split}] {len(feats)} feature salvate -> {out}')

for split in ('train', 'dev', 'test'):
    extract_split(split)
print('FATTO. feature teacher pronte in', OUT_DIR)
