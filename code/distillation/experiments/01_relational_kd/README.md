# 01 — Relational / similarity-preserving KD

Drop-in replacement for the feature term in FD-CMKD. **No new extraction** — uses the
`skeleton_feats_133/*.pkl` you already have.

## Why
FD-CMKD matches feature *values* (`fd_cmkd_loss`), forcing the student into the
teacher's geometry -> neutral. This instead matches the **frame-frame relation
structure**: "which frames the teacher thinks belong together". That is
modality-agnostic, so complementary modalities can agree on it without imitation.

## How to plug in (server, ~5 lines)
Copy `rkd_loss.py` next to `fd_cmkd.py` (or add its folder to `sys.path`), then in
`fd_cmkd_trainer.py::_train_batch`, replace the FD-CMKD feature term with the
relational one. Concretely, where the trainer builds the transfer loss, use:

```python
from rkd_loss import batch_rkd_loss           # add near the top imports
# ... inside _train_batch, in place of the feature part of self.fd(...):
fd_feat = batch_rkd_loss(
    encoder_output, batch.sgn_lengths, batch.sequence,
    self.teacher_feats, self.fd.proj,          # reuse the existing projection head
    mode="sp",                                  # "sp" (similarity) or "rkd" (distance)
    rkd_w=1.0,
)
total = norm_task + self.lambda_feat * fd_feat / self.batch_multiplier
```

Keep the align term if you want (`+ self.lambda_align * fd["align"]`), or drop it to
isolate the relational effect. Start with `mode="sp"`, `lambda_feat` as-is (0.161).

## What to compare
Same protocol as the warm distill run. If RKD lands **below** the matched control
(39.70/40.45) beyond the 0.30 margin, relational structure transfers where value
matching did not — a clean positive for the thesis. If still neutral, it strengthens
the "you need fusion, not distillation" conclusion.

## Files
- `rkd_loss.py` — `similarity_preserving`, `rkd_distance`, `batch_rkd_loss`
  (same signature as `distill.batch_distill_loss`).
