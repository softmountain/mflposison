# DTM-GAN

`dtm_poison_gan` is an independent Distributional Temporal Matching GAN
variant. It deliberately lives beside `poison_gan` so checkpoints, entry
points, and future experiments cannot be confused with the legacy K+1 GAN.

## Combined design

- The K+1 discriminator remains an auxiliary realism signal with a low weight.
- The primary objective is class-conditional teacher-embedding alignment:
  local classes use multi-scale RBF MMD; missing classes fall back to a
  server-broadcast class mean and diagonal variance.
- A VICReg-style variance floor prevents generated class spread from falling
  below real class spread.
- Audio is generated without per-sample z-normalization or hard tail clipping.
  Running real-data mean and standard deviation provide a learnable affine
  calibration, while skew and kurtosis are matched separately.
- Video uses per-frame noise, a GRU temporal decoder, and
  `sigmoid(gate) * softplus(magnitude)` to model sparse MobileNet features.
- Generated and real sequence lengths are aligned before distribution losses.

## Train

```bash
bash fed_multimodal/Local/run_dtm_poison_gan_cloud.sh
```

Training does not run validation by default. To evaluate a small validation
slice during training, pass `--eval_batches N`.

For federated missing-class training, pass a server-exported embedding bank:

```bash
python fed_multimodal/Local/train_dtm_poison_gan.py \
  --prototype_path path/to/prototypes.pt
```

Every run exports `prototypes_<exp_name>.pt`, which can be aggregated or sent
to another client.

## Evaluate or generate

```bash
python fed_multimodal/Local/eval_dtm_poison_gan.py \
  --checkpoint path/to/final_dtm_cloud.pt

python fed_multimodal/Local/generate_dtm_poison_features.py \
  --checkpoint path/to/final_dtm_cloud.pt \
  --num_samples 1000 \
  --target_strategy balanced \
  --attack_mode clean_label
```
