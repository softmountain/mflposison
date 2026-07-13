# Fed-PoisonGAN 云端运行手册

## 1. 上传后进入目录

假设你把整个 `FedPoisonGAN_cloud_bundle` 上传到了云服务器：

```bash
cd FedPoisonGAN_cloud_bundle
```

## 2. 安装环境

推荐 Python 3.9。先安装 PyTorch，再安装本项目。PyTorch 命令请按云服务器 CUDA 版本选择，之后执行：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

检查安装：

```bash
python - <<'PY'
import torch
import fed_multimodal
from fed_multimodal.poison_gan import PoisonGANConfig
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('poison_gan import ok')
PY
```

## 3. 检查数据文件

必须存在：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/feature.pkl
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/feature.pkl
fed_multimodal/datasets/ucf101/ucfTrainTestlist/trainlist01.txt
fed_multimodal/datasets/ucf101/ucfTrainTestlist/testlist01.txt
fed_multimodal/Local/results/local_training/best_model.pt
```

这些是已提取 feature 和 teacher 权重，不需要 UCF101 原始视频。

## 4. 先跑 smoke test

```bash
bash fed_multimodal/Local/run_poison_gan_smoke.sh
```

如果成功，会看到 D/G loss 日志，并生成：

```text
fed_multimodal/Local/results/poison_gan/ckpt_1_smoke.pt
fed_multimodal/Local/results/poison_gan/final_smoke.pt
fed_multimodal/Local/results/poison_gan/history_smoke.json
```

## 5. 正式训练

```bash
bash fed_multimodal/Local/run_poison_gan_cloud.sh
```

或手动执行：

```bash
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 4 \
  --save_interval 10 \
  --log_interval 20 \
  --target_strategy same_as_real \
  --freeze_d backbone \
  --exp_name cloud
```

如显存不足，优先改：

```bash
--batch_size 16 --num_workers 2
```

如只想快速确认长流程，增加：

```bash
--max_batches 10
```

## 6. 评估生成质量

```bash
python fed_multimodal/Local/eval_poison_gan.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_batches 20 \
  --batch_size 32 \
  --output_dir fed_multimodal/Local/results/poison_gan_eval/cloud
```

结果在：

```text
fed_multimodal/Local/results/poison_gan_eval/cloud/analysis_results.json
```

指标解释：

- `target_success_rate` 高：G 生成的 fake 能被 D 的前 K 类判为目标类。
- `fake_escape_rate` 高：fake 没有被 K+1 的第 K 类识别为 fake。
- `fake_class_prob` 低：逃过 fake 类的概率更高。
- `audio_diversity_ratio` / `video_diversity_ratio` 越接近 1：类内多样性越接近真实数据。
- `embedding_mean_l2_gap` / `embedding_var_l1_gap` 越低：fake 和 real 在 D embedding 空间越接近。

## 7. 生成 synthetic feature

Clean-label 版本：

```bash
python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_samples 1000 \
  --target_strategy balanced \
  --attack_mode clean_label \
  --output_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt
```

Label-flip 版本示例：当 synthetic-only 质量达标后，可生成 target 类外观，但训练标签设成 source 类，用于标签翻转中毒实验：

```bash
python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_samples 1000 \
  --target_strategy fixed \
  --target_class 0 \
  --attack_mode label_flip \
  --source_class 1 \
  --output_path fed_multimodal/Local/results/poison_features/cloud_label_flip.pt
```

## 8. 本地合成数据训练验证

```bash
python fed_multimodal/Local/train_with_poison_features.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --poison_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt \
  --mode clean_plus_poison \
  --poison_ratio 0.2 \
  --init_from_model \
  --num_epochs 3 \
  --batch_size 32 \
  --num_workers 4 \
  --output_dir fed_multimodal/Local/results/poison_classifier_eval/cloud
```

结果：

```text
fed_multimodal/Local/results/poison_classifier_eval/cloud/summary.json
fed_multimodal/Local/results/poison_classifier_eval/cloud/final_model.pt
```

## 9. 一键流程

```bash
bash fed_multimodal/Local/run_poison_pipeline.sh
```

这个脚本会依次执行生成器评估、生成 synthetic feature、真实数据 baseline、synthetic-only 训练验证和 real+synthetic 混合训练验证。正式实验前建议先读脚本并按 GPU/时间预算调整 batch size、样本数和分类器训练轮数。

## 10. 常见问题

### `ModuleNotFoundError: fed_multimodal`

在 bundle 根目录运行：

```bash
pip install -e .
```

### CUDA 显存不足

降低：

```bash
--batch_size 16
```

必要时用 CPU smoke test：

```bash
--device cpu --batch_size 2 --max_batches 1
```

### 找不到 feature.pkl

确认当前目录是 bundle 根目录，并且文件存在于：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/feature.pkl
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/feature.pkl
```

### teacher 权重维度不匹配

默认权重路径是：

```text
fed_multimodal/Local/results/local_training/best_model.pt
```

训练脚本会先加载 K 类 teacher，再自动扩展为 K+1 discriminator。不要直接把 K 类 checkpoint strict 加载到 K+1 模型。
