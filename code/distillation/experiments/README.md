# Experiments — beyond neutral distillation

Ordered attempts to beat the neutral cross-modal transfer (baseline 39.35/41.53;
FD-CMKD warm, new teacher 41.76: 39.51/40.81 ~ neutral). Each folder is
self-contained; see its README for how to plug it in.

| Folder | Idea | Uses | Effort | Expected |
|---|---|---|---|---|
| `01_relational_kd/` | match relational *structure* (frame-frame similarity), not feature values | teacher feats you already extracted | low (drop-in loss) | small, but tests the "orthogonal modalities" hypothesis |
| `02_fusion/` | two live streams + cross-attention + joint CTC (no more teacher/student) | both encoders + I3D + keypoints at test | medium | the real jump — mid/low-30s test |

**Diagnosis these attack:** distillation matches feature *values*, forcing the
appearance student into the skeleton geometry -> it loses its own information ->
neutral. Relational KD transfers structure (modality-agnostic); fusion preserves
both axes instead of collapsing one into the other.

Recommended order: run `01` for a quick datapoint, then invest in `02`.
