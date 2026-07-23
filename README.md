# MFL-Poison — 多模态联邦学习特征级 GAN 攻防

本仓库在开源基准 **[FedMultimodal](https://arxiv.org/pdf/2306.09486.pdf)**（USC-SAIL, 2023 KDD ADS）之上，扩展了两代**特征空间多模态 GAN**，用于研究联邦学习场景下的数据增强与投毒攻击。两代 GAN 在同一套框架内并存，共享分类器骨架与数据管线。

> 说明：本项目是防御性安全研究代码，用于评估多模态联邦学习对投毒攻击的鲁棒性。运行依赖预先提取的 UCF101 特征与 teacher 权重，不生成原始视频/音频。

## 项目由来

原始 FedMultimodal 是一个多模态联邦学习基准。本仓库在其上合并了两个衍生工作，它们本是同一条研究线上的两代：

| 代 | 模块 | 定位 | 核心思想 |
|:---:|:---|:---|:---|
| **一代** | `fed_multimodal/generator/` | 数据增强 / 知识蒸馏 | `MultimodalFeatureGAN`：双单模态判别器 + Joint Critic + 冻结 teacher（ACGAN 风格） |
| **二代** | `fed_multimodal/poison_gan/` | 投毒攻击（clean-label / label-flip） | `Fed-PoisonGAN-K+1`：K+1 判别器 + FiLM 生成器 + Memory Bank |

二代用于改善一代的类内多样性与 mode collapse 问题。当前生产架构、模块边界和兼容策略见 [`REFACTORING.md`](REFACTORING.md)。两代目前均针对 **UCF101**（音频 MFCC `[500,80]` + 视频 MobileNetV2 `[9,1280]`，51 类）。

## 目录结构

```
fed_multimodal/
├── constants/        特征维度、类别数常量
├── dataloader/       联邦场景数据管理
├── model/            共享分类器 MMActionClassifier 等
├── trainers/         上游联邦聚合算法（FedAvg / FedProx / FedOpt / SCAFFOLD / FedRS）
├── experiment/       上游各数据集联邦实验入口
├── features/         数据划分 / 缺失模态模拟 / 特征提取三步管线
├── generator/        【一代】增强/蒸馏 GAN
├── poison_gan/       【二代】K+1 投毒 GAN 核心
|-- temporal_adaptive_gan/  Independent temporal-adaptive poison GAN
└── Local/            两代的本地训练/评估/生成入口脚本
```

> 训练产物（`*.pt` / `*.pkl` / `results/` 内容）已通过 `.gitignore` 排除，不随代码上传。运行前需另行准备特征文件与 teacher 权重（见下文）。

## 支持的应用（继承自 FedMultimodal）

- 情感识别：CREMA-D、MELD
- 动作识别：UCF-101、MiT-51、MiT-10
- 人体活动识别：UCI-HAR、KU-HAR
- 社交媒体：Crisis-MMD、Hateful-Memes
- ECG 分类：PTB-XL

## 安装

```bash
git clone git@github.com:softmountain/mflposison.git
cd mflposison

conda create --name mflposison python=3.9
conda activate mflposison

pip install -r requirements.txt
pip install -e .
```

验证安装：

```bash
python - <<'PY'
import torch, fed_multimodal
from fed_multimodal.poison_gan import PoisonGANConfig
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('poison_gan import ok')
PY
```

## 运行前置数据

两代 GAN 都工作在**已提取的特征空间**，需先准备以下文件（不在版本控制内）：

```text
fed_multimodal/results/feature/audio/mfcc/ucf101/feature.pkl        # UCF101 MFCC 音频特征
fed_multimodal/results/feature/video/mobilenet_v2/ucf101/feature.pkl # UCF101 MobileNetV2 视频特征
fed_multimodal/datasets/ucf101/ucfTrainTestlist/{trainlist01,testlist01}.txt
fed_multimodal/Local/results/local_training/best_model.pt           # K 类 teacher 权重
```

特征可用 `fed_multimodal/Local/extract_features_local.py` 提取；teacher 可用 `fed_multimodal/Local/train_local.py` 集中训练得到。

---

## 统一联邦攻防入口

生产主线使用严格八段场景配置运行 clean pretrain、M* 选择、客户端独立生成器、攻击、服务器检测与防御：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
```

默认结果写入 `artifacts/ucf101_generative_poison_defense/`。`summary.json` 保存最终指标，`round_records/` 保存逐轮检测与聚合审计，`snapshots/` 和 `generator_checkpoints/` 保存模型产物。可通过 `--artifact-root` 覆盖输出目录。

## 二代旧版：Fed-PoisonGAN（K+1 投毒）

以下脚本保留用于加载和评估旧 checkpoint；新的联邦攻击训练应使用上面的统一入口。

### 组成结构

- **生成器 `PoisonFeatureGenerator`**：`z(256) + label_emb(128)` → 共享 trunk → 两个 **FiLM 块**（label 控类别、z 控类内变化，解耦以缓解 mode collapse）→ 音频支（ConvTranspose 上采样 + per-sample z-norm）/ 视频支（MLP + ReLU 非负 + clamp）。
- **`legacy`** keeps the original `PoisonFeatureGenerator` and remains compatible with existing checkpoints.
- **`temporal_adaptive` - `fed_multimodal/temporal_adaptive_gan/`** is now a physically separate package with its own config, model, losses, and trainer. It uses running real-audio calibration, per-frame noise, temporal convolution, class-specific video scale/bias, a 1:3 D/G schedule, decaying instance noise, lazy R1, audio moment matching, and diversity warm-up.
- **判别器（K+1）**：把 K 类分类器扩成 K+1 类，第 K 类代表 fake/poison 类；骨干与前 K 行分类头从 teacher 迁移，fake 类行用 teacher 权重均值初始化。
- **损失**：判别器 `CE(real) + λ·CE(fake→K)`；生成器 = target CE + avoid（压低 fake 类）+ feature matching + variance matching + mode-seeking diversity + 统计对齐（多样性项带 warmup）。**Memory Bank** 用动量维护类原型，解决 batch 内缺类问题。

### Independent `temporal_adaptive` variant

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
python fed_multimodal/Local/eval_temporal_adaptive_gan.py --checkpoint path/to/checkpoint.pt
python fed_multimodal/Local/generate_temporal_adaptive_features.py --checkpoint path/to/checkpoint.pt
```

Set `generator.variant: temporal_adaptive` in the scenario config. The old
training script is a temporary wrapper around the unified runner so generator
training cannot bypass a malicious client's FedMM partition. Existing
checkpoints remain supported by the evaluation and generation backends.

### 1. Smoke test

```bash
bash fed_multimodal/Local/run_poison_gan_smoke.sh
```

### 2. 正式训练

```bash
bash fed_multimodal/Local/run_poison_gan_cloud.sh
# 或手动：
python fed_multimodal/Local/train_poison_gan.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --epochs 50 --batch_size 32 --num_workers 4 \
  --save_interval 10 --target_strategy same_as_real \
  --freeze_d backbone --exp_name cloud
```

### 3. 评估生成质量

```bash
python fed_multimodal/Local/eval_poison_gan.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_batches 20 \
  --output_dir fed_multimodal/Local/results/poison_gan_eval/cloud
```

关键指标：`target_success_rate`（被判为目标类）、`fake_escape_rate`（逃过 fake 类检测）、`fake_class_prob`（越低越好）、`audio/video_diversity_ratio`（越接近 1 越好）、`embedding_mean_l2_gap` / `embedding_var_l1_gap`（越低越好）。

### 4. 生成合成/毒特征

```bash
# clean-label（偷渡式，标签正确）
python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_samples 1000 --target_strategy balanced \
  --attack_mode clean_label \
  --output_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt

# label-flip（生成目标类外观，标记为源类）
python fed_multimodal/Local/generate_poison_features.py \
  --checkpoint fed_multimodal/Local/results/poison_gan/ckpt_50_cloud.pt \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --num_samples 1000 --target_strategy fixed --target_class 0 \
  --attack_mode label_flip --source_class 1 \
  --output_path fed_multimodal/Local/results/poison_features/cloud_label_flip.pt
```

### 5. 下游验证（旧 checkpoint 兼容）

```bash
python fed_multimodal/Local/train_with_poison_features.py \
  --model_path fed_multimodal/Local/results/local_training/best_model.pt \
  --poison_path fed_multimodal/Local/results/poison_features/cloud_clean_label.pt \
  --mode clean_plus_poison --poison_ratio 0.2 --init_from_model \
  --num_epochs 3 --batch_size 32 --num_workers 4 \
  --output_dir fed_multimodal/Local/results/poison_classifier_eval/cloud
```

以上命令仅用于已有 checkpoint 的独立评估。新的联邦中毒与服务器防御
流程统一使用：

```bash
python -m mflpoison.runner \
  --config configs/scenarios/ucf101_generative_poison_defense.yaml
```

兼容别名：`bash fed_multimodal/Local/run_poison_pipeline.sh`。

---

## 一代：MultimodalFeatureGAN（增强 / 蒸馏）

### 组成结构

- **生成器**：音频、视频**各一个独立生成器**，标签嵌入与噪声拼接输入；音频用 ConvTranspose 上采样、视频用 MLP，末端均做尺度约束 + clamp。
- **判别器**：双单模态判别器（带 Spectral Norm + ACGAN 辅助分类头）+ Joint Critic（联合真假判别）+ 冻结的 teacher（提供语义蒸馏损失）。
- **损失**：对抗 + ACGAN 辅助 + teacher 蒸馏 + Joint + Feature Matching + Moment Matching 的加权和，配课程学习 warmup。

### 运行

```bash
# 训练
python fed_multimodal/Local/train_local_gan.py   # 参数见脚本 / GAN_EXPERIMENT_README.md
# 评估
python fed_multimodal/Local/eval_local_gan_quality.py
```

评估维度：t-SNE 可视化、类中心距离、TSTR（ML Efficacy）、分布统计。详见 `fed_multimodal/Local/GAN_EXPERIMENT_README.md`。

---

## 上游 FedMultimodal 快速上手（UCI-HAR 示例）

```bash
cd fed_multimodal/data && bash download_uci_har.sh && cd ..
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
cd experiment/uci-har && bash run_base.sh
```

数据处理三步（划分 / 缺失模态模拟 / 特征提取）与更多数据集，参见 `fed_multimodal/features/`。

## 引用

本项目基于 FedMultimodal，若使用请引用原始工作：

```bibtex
@inproceedings{feng2023fedmultimodal,
  title={Fedmultimodal: A benchmark for multimodal federated learning},
  author={Feng, Tiantian and Bose, Digbalay and Zhang, Tuo and Hebbar, Rajat and Ramakrishna, Anil and Gupta, Rahul and Zhang, Mi and Avestimehr, Salman and Narayanan, Shrikanth},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4035--4045},
  year={2023}
}
```

原框架致谢：USC-SAIL 团队（Corresponding Author: Tiantian Feng, tiantiaf@usc.edu）。框架配图来自 [openmoji.org](https://openmoji.org/)。
