# 03 — End-to-End Two-Stream Fusion (live encoders)

The frozen-feature study (`../02_fusion`) proved the two modalities are **complementary**
(oracle 36.7 test vs 44 for either stream) but that their independently-trained CTC
spikes are **misaligned**, so frame-level fusion collapsed and only sequence-level
routing (`conf-route`, 42.4) helped. The fix is to stop freezing: train both encoders
**jointly** so their representations and timing become compatible.

## Model (`twostream_model.py`)
- **appearance**: I3D (T,1024) → linear proj → 3-layer Transformer encoder (Signformer-style, d=256).
- **skeleton**: keypoint TSSI (3,J,T) → `PoseNetworkCTC.forward_feat` (GCN+TCN+attn+BiLSTM, 512-d) → proj.
- **fusion**: skeleton resampled to the appearance time axis → N bidirectional cross-attention
  blocks → **shared CTC head** (single, aligned emission timing) + per-stream aux CTC heads
  (deep supervision).

## Data (`twostream_data.py`)
Pairs, per video, the I3D gzip-pickle (`phoenix14t.pami0.{split}`) with the keypoint
pickle (`Phoenix-2014T.{split}`), reusing the skeleton TSSI transform verbatim.

## Training (`twostream_train.py`) — durable
- skeleton encoder **warm-started** from the trained skeleton checkpoint; appearance from scratch.
- loss = CTC(joint) + α·(CTC(app) + CTC(skel)).
- **Resumable checkpoints**: every epoch writes `last.pt` atomically (model + optimizer +
  scheduler + epoch + best-dev + early-stop counter + RNG states). `train(resume=True)`
  (default) continues a killed run exactly where it stopped. `best.pt` = best-on-dev.

## Run
Open `run_twostream.ipynb` → **Run All**. If the job is interrupted, just Run All again:
it resumes from `last.pt`. Baselines to beat: skeleton-alone ~42.6 test, conf-route 42.4,
Signformer 41.53; ceiling = oracle 36.7.
