# Temporal Adaptive Poison GAN

`temporal_adaptive_gan` is an independent K+1 poison-GAN variant. It is
kept separate from `poison_gan` so its model, losses, trainer, and checkpoints
can evolve without changing the legacy implementation.

## Design

- Running real-audio statistics calibrate generated audio without per-sample
  z-normalization or hard audio clipping.
- Per-frame noise, positional embeddings, temporal convolution, and
  class-specific scale/bias model video diversity and temporal structure.
- Real sequence lengths mask generated features and distribution losses.
- The preset uses a 1:3 D/G schedule, decaying instance noise, lazy R1,
  feature/statistical matching, audio mean/std/kurtosis matching, and a
  diversity warm-up.
- A server-broadcast prototype bank can initialize missing-class targets.

## Entry points

Use the dedicated scripts; the legacy poison-GAN launchers remain unchanged.

```bash
python fed_multimodal/Local/train_temporal_adaptive_gan.py --epochs 50

python fed_multimodal/Local/eval_temporal_adaptive_gan.py --checkpoint path/to/checkpoint.pt

python fed_multimodal/Local/generate_temporal_adaptive_features.py \
  --checkpoint path/to/checkpoint.pt --num_samples 1000
```
