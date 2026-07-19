# 02 — Cross-Attention Two-Stream Fusion

Stop distilling; **fuse**. Two live encoders (appearance + skeleton) exchange
information via bidirectional cross-attention and decode from a joint CTC head. This
is the family that reaches ~20% test WER (TwoStream-SLR, NeurIPS 2022; MSKA). It
preserves both axes (appearance=handshape texture, skeleton=geometry/motion) instead
of collapsing one into the other.

**Trade-off vs distillation:** you need BOTH modalities at test time — I3D (have it)
and RTMW-133 keypoints (have them). The thesis avoided this (compact student, no
teacher at test); fusion embraces it.

## Files (ready + shape-tested)
- `fusion_model.py` — `CrossModalFusion(app_dim, skel_dim, num_classes, ...)`:
  takes the two encoders' output sequences, returns `{joint, app, skel}` CTC log-probs.
  Skeleton is resampled onto the appearance time axis. ~4.5M params on top of the encoders.
- `fusion_loss.py` — `FusionLoss`: `L_joint + alpha*(L_app+L_skel) + beta*L_crossKD`
  (bidirectional KL between the two stream posteriors = mutual learning).

## Integration (the work to do on the server)
The two modules are architecture-agnostic; wire them to your existing encoders:

1. **Encoders (reuse what you have):**
   - Appearance: `SignModel.encode(...)` -> `encoder_output` (B, Ta, 256).
   - Skeleton: `PoseNetworkCTC.forward_feat(tssi)` -> (B, Ts, 512) — the same method
     the extractor uses. Now trained **live** (not frozen).
2. **Wrap:** a `FusionNet(nn.Module)` holding both encoders + `CrossModalFusion`.
   `forward(sgn_i3d, sgn_mask, kp_tssi)` -> the 3-head dict.
3. **Data:** the loader must serve, per video, BOTH the I3D features and the
   133-kp TSSI, time-independent (they get aligned inside the fusion). This is the
   main new plumbing — extend `signformer/main/data.py`/`dataset.py` to also load the
   keypoint pickle (`dataset/phoenix2014t_133kp`) keyed by the same video name.
4. **Trainer:** compute `FusionLoss`, backprop; decode from `out["joint"]` with your
   usual beam+prior sweep.
5. **cuDNN:** keep `torch.backends.cudnn.enabled = False` (env issue).

## Suggested schedule
Warm-start each stream from its converged solo checkpoint (Signformer baseline +
skeleton 41.76), then fine-tune the fusion jointly at a small LR. Start
`alpha=0.3, beta=0.5, layers=2`.

## Expected
Realistically mid-30s -> low-30s test (from 40.8 distilled / 41.5 baseline). Not the
~20% SOTA (needs HRNet keypoints + pretraining + bigger nets), but a genuine jump and
the clean thesis punchline: *fusion beats distillation because it preserves the
complementarity distillation destroys.*
