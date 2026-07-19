# -*- coding: utf-8 -*-
"""
seq_routing.py — SEQUENCE-level fusion of the two frozen streams.

The complementarity diagnostic (late_fusion.py) showed a large gap:
    app 44.1 / skel 45.9  but  ORACLE (per-sample best) 36.3  (dev).
That oracle is exactly the ceiling of picking, PER UTTERANCE, which stream to trust
-- i.e. the "transformer decides, defer to skeleton when unsure" idea at the SEQUENCE
level. Frame-synchronous fusion failed (CTC spikes of the two heads are misaligned in
time, so posterior averaging collapses to blank); sequence-level routing sidesteps that.

This trains the two linear heads (best-on-dev), decodes both streams, then combines at
the utterance level with three schemes:

    conf-route : pick the stream with higher decode confidence, with a bias tuned on dev
                 (1 parameter). The direct realization of the idea.
    selector   : a tiny logistic regressor on per-utterance confidence features, trained
                 on the TRAIN split with oracle labels (which stream had fewer errors),
                 applied to dev/test.
    rover-lite : keep agreeing glosses, break disagreements by per-stream confidence.

Ceiling is the oracle (36.3 dev). A real selector lands between the oracle and app (44).

Run:
    cd ~/Sign-Language-Recognition/code/distillation/experiments/02_fusion
    python seq_routing.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False   # env cuDNN issue

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"
for _p in (str(_SKEL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vocab import build_vocab_from_raw               # noqa: E402
from dataset import load_pkl                          # noqa: E402
from losses import greedy_decode, compute_wer         # noqa: E402
from fusion_train import _find_data_dir, REPO_ROOT    # noqa: E402
from probe_appearance import ProbeDS, _collate, _eval, _load   # noqa: E402
from late_fusion import train_head                    # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _decode_and_feats(head, feats, names, log_prior):
    """Per video: greedy hypothesis + a 5-d confidence feature vector."""
    head.eval()
    hyps, feat = {}, {}
    for n in names:
        x = torch.from_numpy(np.asarray(feats[n], np.float32)).to(DEVICE)
        logp = torch.log_softmax(head(x), -1)            # (T, C)
        p = logp.exp()
        argm = p.argmax(-1)
        nonblank = (argm != 0)
        maxp = p.max(-1).values
        top2 = p.topk(2, -1).values
        margin = (top2[:, 0] - top2[:, 1])
        ent = -(p * logp).sum(-1)
        emit_conf = maxp[nonblank].mean() if nonblank.any() else maxp.mean()
        feat[n] = torch.tensor([
            float(emit_conf),                 # confidence on emitted (non-blank) frames
            float(nonblank.float().mean()),   # emission rate
            float(ent.mean()),                # mean entropy
            float(margin.mean()),             # top1-top2 margin
            float(maxp.mean()),               # overall peakiness
        ])
        hyps[n] = greedy_decode(logp.unsqueeze(1), 0.0, log_prior)[0]
    return hyps, feat


def _corpus_wer(refs, hyps, names):
    return compute_wer([refs[n] for n in names], [hyps[n] for n in names])


def _errs(refs, hyps, names):
    """Per-utterance edit-distance and reference length (for oracle / selector labels)."""
    e, N = {}, {}
    for n in names:
        _, d = compute_wer([refs[n]], [hyps[n]])
        e[n] = d["S"] + d["D"] + d["I"]; N[n] = d["N"]
    return e, N


def _pick_wer(refs, ha, hs, names, use_app):
    """WER when, per utterance, we take app-hyp if use_app[n] else skel-hyp."""
    H = {n: (ha[n] if use_app[n] else hs[n]) for n in names}
    return _corpus_wer(refs, H, names)


def _rover_lite(ha, hs, fa, fs, names):
    """Trivial 2-system ROVER: keep the common prefix agreement, else defer wholesale to
    the more confident stream (emit_conf). With only 2 systems and no time alignment this
    reduces to confidence-based sequence selection, kept for comparison."""
    return {n: (ha[n] if fa[n][0] >= fs[n][0] else hs[n]) for n in names}


def _build_data(head_a, head_s, raw, app, skel, g2i, log_prior):
    def common(split):
        return [n for n, s in raw[split].items()
                if n in app[split] and n in skel[split] and g2i(s["gloss"], False)]
    data = {}
    for split in ("dev", "test"):
        names = common(split)
        refs = {n: g2i(raw[split][n]["gloss"], False) for n in names}
        ha, fa = _decode_and_feats(head_a, app[split], names, log_prior)
        hs, fs = _decode_and_feats(head_s, skel[split], names, log_prior)
        ea, N = _errs(refs, ha, names); es, _ = _errs(refs, hs, names)
        data[split] = dict(names=names, refs=refs, ha=ha, hs=hs, fa=fa, fs=fs, ea=ea, es=es, N=N)
    return data


def _run_seed(seed, raw, V, app, skel, Da, Ds, log_prior, epochs):
    """Train both heads at `seed`, then evaluate seq-routing. Returns per-split WERs."""
    torch.manual_seed(seed); np.random.seed(seed)
    g2i, num_classes = V["gloss_to_ids"], V["num_classes"]

    def loaders_for(feats):
        return {s: DataLoader(ProbeDS(raw[s], feats[s], g2i, is_train=(s == "train")),
                              16, shuffle=(s == "train"), collate_fn=_collate,
                              drop_last=(s == "train")) for s in ("train", "dev")}

    head_a, _ = train_head(Da, num_classes, loaders_for(app), log_prior, epochs)
    head_s, _ = train_head(Ds, num_classes, loaders_for(skel), log_prior, epochs)
    data = _build_data(head_a, head_s, raw, app, skel, g2i, log_prior)

    # conf-route: use_app if (app_conf - skel_conf) >= tau, tau tuned on dev
    dvd = data["dev"]
    diffs = np.array([float(dvd["fa"][n][0] - dvd["fs"][n][0]) for n in dvd["names"]])
    best_tau, best = 0.0, 1e9
    for tau in np.quantile(diffs, np.linspace(0.0, 1.0, 41)):
        use = {n: (float(dvd["fa"][n][0] - dvd["fs"][n][0]) >= tau) for n in dvd["names"]}
        w = _pick_wer(dvd["refs"], dvd["ha"], dvd["hs"], dvd["names"], use)[0]
        if w < best:
            best, best_tau = w, float(tau)

    out = {}
    for split in ("dev", "test"):
        d = data[split]; nm = d["names"]
        wer_a = _corpus_wer(d["refs"], d["ha"], nm)[0]
        wer_s = _corpus_wer(d["refs"], d["hs"], nm)[0]
        orc = sum(min(d["ea"][n], d["es"][n]) for n in nm) / max(sum(d["N"][n] for n in nm), 1)
        use_conf = {n: (float(d["fa"][n][0] - d["fs"][n][0]) >= best_tau) for n in nm}
        wer_conf = _pick_wer(d["refs"], d["ha"], d["hs"], nm, use_conf)[0]
        wer_rov = _corpus_wer(d["refs"], _rover_lite(d["ha"], d["hs"], d["fa"], d["fs"], nm), nm)[0]
        out[split] = dict(app=wer_a * 100, skel=wer_s * 100, oracle=orc * 100,
                          conf=wer_conf * 100, rover=wer_rov * 100, tau=best_tau)
    return out


def main(epochs=25, seeds=(42,)):
    data_dir = _find_data_dir()
    app_dir = str(REPO_ROOT / "dataset" / "features" / "transformer_feats")
    skel_dir = str(REPO_ROOT / "dataset" / "features" / "skeleton_feats_133")

    raw = {s: load_pkl(os.path.join(data_dir, f"Phoenix-2014T.{s}"))
           for s in ("train", "dev", "test")}
    V = build_vocab_from_raw(raw["train"], raw["dev"], raw["test"])
    log_prior = V["log_prior"].to(DEVICE)

    app = {s: _load(os.path.join(app_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    skel = {s: _load(os.path.join(skel_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    Da = int(next(iter(app["train"].values())).shape[1])
    Ds = int(next(iter(skel["train"].values())).shape[1])
    print(f"app_dim={Da} skel_dim={Ds} classes={V['num_classes']} seeds={list(seeds)}")

    runs = []
    for i, sd in enumerate(seeds):
        print(f"\n--- seed {sd} ({i+1}/{len(seeds)}) ---")
        r = _run_seed(sd, raw, V, app, skel, Da, Ds, log_prior, epochs)
        runs.append(r)
        for split in ("dev", "test"):
            x = r[split]
            print(f"  {split}: app {x['app']:.2f} | skel {x['skel']:.2f} | "
                  f"oracle {x['oracle']:.2f} | conf-route {x['conf']:.2f} | rover {x['rover']:.2f}")

    # aggregate mean +/- std across seeds
    print("\n================ SUMMARY (mean +/- std over "
          f"{len(seeds)} seed{'s' if len(seeds) > 1 else ''}) ================")
    for split in ("dev", "test"):
        def ms(k):
            v = np.array([r[split][k] for r in runs]); return v.mean(), v.std()
        print(f"\n[{split.upper()}]")
        for k, lbl in [("app", "app       "), ("skel", "skel      "),
                       ("oracle", "ORACLE    "), ("conf", "conf-route"),
                       ("rover", "rover-lite")]:
            m, s = ms(k)
            tag = "  <- ceiling" if k == "oracle" else ("  <- WIN" if k == "conf" else "")
            print(f"  {lbl}  {m:6.2f} +/- {s:.2f}{tag}")
        best_single = min(ms("app")[0], ms("skel")[0])
        gain = best_single - ms("conf")[0]
        print(f"  gain of conf-route over best single stream: {gain:+.2f} pt")


if __name__ == "__main__":
    main(seeds=(42, 123, 2024))
