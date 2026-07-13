#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python fed_multimodal/Local/train_dtm_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 4 \
  --save_interval 10 \
  --log_interval 20 \
  --target_strategy same_as_real \
  --freeze_d backbone \
  --exp_name dtm_cloud
