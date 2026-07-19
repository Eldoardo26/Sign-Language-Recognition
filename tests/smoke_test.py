# -*- coding: utf-8 -*-
"""Smoke test: verifies that every entry point of this repository is importable,
that the models build and run a forward pass, and that the notebook path
resolution works — without any training and, where possible, without data.

Run from the repository root:

    python tests/smoke_test.py

Statuses:
  PASS  the check succeeded
  ENV   a heavyweight dependency is missing/broken in this environment
        (e.g. TensorFlow for the Signformer framework) — not a repo defect
  DATA  the check needs dataset files that are not present locally
  FAIL  a genuine defect

Exit code is non-zero only on FAIL.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

TESTS = []


def t(name, code):
    TESTS.append((name, code))


# ---------- teacher stack, coco133 layout ----------
t("csrl coco133: imports, model build, dummy forward", r"""
import os, sys, torch
os.environ['KP_LAYOUT'] = 'coco133'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import config, vocab, losses
import skeleton_coco133 as sk
import model as M
assert sk.NUM_JOINTS == 55 and len(sk.DFS_ORDER) == 55
adj = sk.build_adjacency() if hasattr(sk, 'build_adjacency') else getattr(sk, 'ADJACENCY', None)
net = M.PoseNetworkCTC(num_classes=1087, in_channels=55*3, use_gcn=True,
                       gcn_channels=16, adjacency=adj)
n = sum(p.numel() for p in net.parameters())
with torch.no_grad():
    y = net(torch.randn(2, 3, 55, 40))
out = y[0] if isinstance(y, tuple) else y
print(f'params {n:,} | out {tuple(out.shape)}')
""")

# ---------- teacher stack, legacy mediapipe75 layout (reverse runs) ----------
t("csrl mediapipe75: legacy layout build + forward", r"""
import os, sys, torch
os.environ['KP_LAYOUT'] = 'mediapipe75'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import skeleton_mediapipe75 as sk
import model as M
net = M.PoseNetworkCTC(num_classes=1024, in_channels=75*3, use_gcn=False)
with torch.no_grad():
    y = net(torch.randn(2, 3, 75, 30))
print('ok', tuple((y[0] if isinstance(y, tuple) else y).shape))
""")

# ---------- decoding + WER ----------
t("csrl losses: greedy/beam decode + WER", r"""
import os, sys, torch
os.environ['KP_LAYOUT'] = 'coco133'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import losses
lp = torch.log_softmax(torch.randn(25, 2, 30), dim=-1)
prior = torch.log_softmax(torch.randn(30), dim=-1)
g = losses.greedy_decode(lp, beta=0.3, log_prior=prior)
b = losses.beam_decode(lp, beam_width=4, beta=0.3, log_prior=prior)
w, d = losses.compute_wer([[1, 2, 3]], [[1, 3]])
assert len(g) == 2 and len(b) == 2
print('greedy+beam ok | wer', round(w, 3), d)
""")

# ---------- script import-safety (all are __main__-guarded) ----------
for script, needs_signformer, env_extra in [
    ("code/csrl_skeleton/train_teacher.py", False, "os.environ['KP_LAYOUT']='coco133'"),
    ("code/csrl_skeleton/prepare_phoenix_133.py", False, "os.environ['KP_LAYOUT']='coco133'"),
    ("code/csrl_skeleton/extract_wholebody_133.py", False, ""),
    ("code/distillation/extract_skeleton_feats.py", True, "os.environ['KP_LAYOUT']='coco133'"),
    ("code/distillation/extract_transformer_feats.py", True, ""),
]:
    mod = os.path.basename(script)[:-3]
    d = os.path.dirname(script).replace("/", os.sep)
    sf = ("sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'signformer'))"
          if needs_signformer else "")
    t(f"import-safe: {mod}", f"""
import os, sys
{env_extra}
{sf}
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
sys.path.insert(0, os.path.join(os.getcwd(), r'{d}'))
import importlib
m = importlib.import_module('{mod}')
print('imported', m.__name__)
""")

# ---------- distillation libraries ----------
t("distillation: vocab_utils + fd_cmkd + distill", r"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'signformer'))
import vocab_utils, fd_cmkd, distill
print('modules ok')
""")

t("distillation: fd_cmkd_trainer (Signformer framework)", r"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'signformer'))
import fd_cmkd_trainer
print('imported fd_cmkd_trainer')
""")

t("distillation: reverse_distill", r"""
import os, sys
os.environ['KP_LAYOUT'] = 'mediapipe75'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'signformer'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import reverse_distill
print('imported reverse_distill')
""")

# ---------- experiments ----------
t("exp01: rkd_loss", r"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation', 'experiments', '01_relational_kd'))
import rkd_loss
print('rkd_loss ok')
""")

t("exp02: fusion stack", r"""
import os, sys
os.environ['KP_LAYOUT'] = 'coco133'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation', 'experiments', '02_fusion'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import fusion_loss, fusion_model, fusion_train, late_fusion, seq_routing, probe_appearance
print('fusion stack ok')
""")

t("exp03: two-stream build + dummy forward (expects 14.53M params)", r"""
import os, sys, torch
os.environ['KP_LAYOUT'] = 'coco133'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'distillation', 'experiments', '03_twostream_e2e'))
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
import twostream_data, twostream_model, twostream_train
import skeleton_coco133 as sk
adj = sk.build_adjacency() if hasattr(sk, 'build_adjacency') else getattr(sk, 'ADJACENCY', None)
m = twostream_model.TwoStreamFusion(num_classes=1087, num_joints=55, adjacency=adj)
n = sum(p.numel() for p in m.parameters())
pad = torch.zeros(2, 50, dtype=torch.bool); pad[1, 40:] = True
with torch.no_grad():
    out = m(torch.randn(2, 50, 1024), torch.randn(2, 3, 55, 60), pad,
            skel_lens=torch.tensor([60, 48]))
assert set(out) >= {'joint', 'app', 'skel'}
assert abs(n / 1e6 - 14.53) < 0.2, f'unexpected param count {n}'
print(f'params {n/1e6:.2f}M | heads {sorted(out)}')
""")

# ---------- notebook path resolution ----------
t("nb paths: run_twostream locates code/ and dataset/", r"""
import os
os.chdir(os.path.join('code', 'distillation', 'experiments', '03_twostream_e2e'))
_here = os.path.abspath('.')
CODE = None
for _k in range(6):
    _b = os.path.abspath(os.path.join(_here, *(['..'] * _k)))
    if os.path.exists(os.path.join(_b, 'csrl_skeleton', 'model.py')):
        CODE = _b; break
assert CODE, 'cannot locate code/csrl_skeleton'
print('resolved CODE =', os.path.relpath(CODE, os.path.dirname(CODE)))
""")

# ---------- optional: real data + real checkpoint (skipped cleanly if absent) ----------
t("e2e (optional): dev pickle -> TSSI -> run133 ckpt -> decode", r"""
import os, sys, torch, pickle, gzip
import numpy as np
os.environ['KP_LAYOUT'] = 'coco133'
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'csrl_skeleton'))
p = os.path.join('dataset', 'phoenix2014t_133kp', 'Phoenix-2014T.dev')
ck = os.path.join('runs', 'run133', 'tssi133_gcn_best.pt')
assert os.path.exists(p), f'dataset not present: {p}'
assert os.path.exists(ck), f'checkpoint not present: {ck}'
import skeleton_coco133 as sk, model as M, losses
try:
    data = pickle.load(gzip.open(p, 'rb'))
except Exception:
    data = pickle.load(open(p, 'rb'))
item = list(data.values())[0] if isinstance(data, dict) else data[0]
tssi = sk.generate_tssi_75(item['keypoint'])
sd = torch.load(ck, map_location='cpu', weights_only=False)
sd = sd.get('model', sd)
adj = sk.build_adjacency() if hasattr(sk, 'build_adjacency') else getattr(sk, 'ADJACENCY', None)
net = M.PoseNetworkCTC(num_classes=1087, in_channels=55*3, use_gcn=True,
                       gcn_channels=16, adjacency=adj)
net.load_state_dict(sd, strict=True)
net.eval()
x = torch.as_tensor(np.asarray(tssi), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    y = net(x)
logits = y[0] if isinstance(y, tuple) else y
lp = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
ids = losses.greedy_decode(lp)[0]
print('tssi', tuple(np.asarray(tssi).shape), '| ckpt strict-loaded | decoded', len(ids), 'glosses')
""")


def classify(r):
    err = r.stderr or ""
    last = err.strip().splitlines()[-1] if err.strip() else ""
    if r.returncode == 0:
        return "PASS", (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
    if ("_ARRAY_API not found" in err
            or ("numpy.core" in err and "failed to import" in err)
            or ("ModuleNotFoundError" in last and any(
                p in last for p in ("tensorflow", "mmpose", "mmcv", "mediapipe",
                                    "rtmlib", "onnxruntime", "sophia")))):
        return "ENV ", last
    if "AssertionError" in last and ("not present" in last or "cannot locate" in last):
        return "DATA", last
    return "FAIL", last


def main():
    results = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for name, code in TESTS:
        r = subprocess.run([PY, "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        status, detail = classify(r)
        results.append(status)
        print(f"[{status}] {name}")
        if detail:
            print(f"       {detail}")
    n = len(results)
    print(f"\nsummary: {results.count('PASS')}/{n} pass, "
          f"{results.count('ENV ')} env-skip, {results.count('DATA')} data-skip, "
          f"{results.count('FAIL')} FAIL")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    main()
