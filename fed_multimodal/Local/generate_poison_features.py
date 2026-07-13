#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from fed_multimodal.poison_gan import (
    FedPoisonGANTrainer,
    PoisonDiscriminator,
    PoisonGANConfig,
    build_poison_generator,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic UCF101 feature tensors for downstream training")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="fed_multimodal/Local/results/local_training/best_model.pt")
    parser.add_argument("--output_path", type=str, default="fed_multimodal/Local/results/poison_features/poison.pt")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_strategy", type=str, default="balanced", choices=["balanced", "fixed", "mixed"])
    parser.add_argument("--attack_mode", type=str, default="clean_label", choices=["clean_label", "label_flip"])
    parser.add_argument("--source_class", type=int, default=-1)
    parser.add_argument("--target_class", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_labels(args, config):
    if args.target_strategy == "fixed":
        if args.target_class < 0:
            raise ValueError("--target_class is required for fixed strategy")
        condition = torch.full((args.num_samples,), args.target_class, dtype=torch.long)
    elif args.target_strategy == "mixed":
        if args.target_class < 0:
            raise ValueError("--target_class is required for mixed strategy")
        condition = torch.randint(0, config.num_classes, (args.num_samples,), dtype=torch.long)
        mask = torch.rand(args.num_samples) < config.mixed_target_prob
        condition[mask] = args.target_class
    else:
        condition = torch.arange(args.num_samples, dtype=torch.long) % config.num_classes
    if args.attack_mode == "clean_label":
        train_label = condition.clone()
    else:
        if args.source_class < 0:
            raise ValueError("--source_class is required for label_flip mode")
        train_label = torch.full_like(condition, args.source_class)
    return condition, train_label


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = PoisonGANConfig.from_dict(checkpoint["config"])
    disc_model, _ = build_kplus1_discriminator(
        args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        freeze=config.freeze_d,
        device=args.device,
    )
    generator = build_poison_generator(config)
    trainer = FedPoisonGANTrainer(generator, PoisonDiscriminator(disc_model), config, device=args.device)
    trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
    condition, train_label = build_labels(args, config)
    data = trainer.generate(args.num_samples, condition, batch_size=args.batch_size, train_labels=train_label)
    data["meta"] = {
        "checkpoint": args.checkpoint,
        "attack_mode": args.attack_mode,
        "target_strategy": args.target_strategy,
        "source_class": args.source_class,
        "target_class": args.target_class,
        "config": config.to_dict(),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved {args.num_samples} synthetic samples to {output_path}")


if __name__ == "__main__":
    main()
