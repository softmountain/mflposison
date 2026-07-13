#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from fed_multimodal.Local.dataloader import UCF101LocalDataManager
from fed_multimodal.temporal_adaptive_gan import (
    PoisonDiscriminator,
    TemporalAdaptiveGANConfig,
    TemporalAdaptiveGANTrainer,
    TemporalAdaptivePoisonGenerator,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a temporal-adaptive GAN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="fed_multimodal/Local/results/local_training/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="fed_multimodal/results")
    parser.add_argument("--dataset_dir", type=str, default="fed_multimodal/datasets/ucf101")
    parser.add_argument("--output_dir", type=str, default="fed_multimodal/Local/results/temporal_adaptive_gan_eval/default")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = TemporalAdaptiveGANConfig.from_dict(checkpoint["config"])

    dm = UCF101LocalDataManager(args.data_dir, args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    loaders = dm.get_dataloaders()
    disc_model, _ = build_kplus1_discriminator(
        args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        freeze=config.freeze_d,
        device=args.device,
    )
    discriminator = PoisonDiscriminator(disc_model)
    generator = TemporalAdaptivePoisonGenerator(
        num_classes=config.num_classes,
        audio_seq_len=config.audio_seq_len,
        audio_feat_dim=config.audio_feat_dim,
        video_seq_len=config.video_seq_len,
        video_feat_dim=config.video_feat_dim,
        z_dim=config.z_dim,
        label_emb_dim=config.label_emb_dim,
        hidden_dim=config.hidden_dim,
        video_out_max=config.video_out_max,
        video_scale_max=config.video_scale_max,
        frame_noise_dim=config.frame_noise_dim,
        temporal_groups_max=config.temporal_groups_max,
        audio_stats_momentum=config.audio_stats_momentum,
    )
    trainer = TemporalAdaptiveGANTrainer(generator, discriminator, config, device=args.device)
    trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
    loader = loaders["full_train"] if args.use_train else loaders["test"]
    metrics = trainer.evaluate(loader, num_batches=args.num_batches)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "metrics": metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_dir / 'analysis_results.json'}")


if __name__ == "__main__":
    main()
