#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from fed_multimodal.Local.dataloader import UCF101LocalDataManager
from fed_multimodal.poison_gan import (
    FedPoisonGANTrainer,
    PoisonDiscriminator,
    PoisonGANConfig,
    build_poison_generator,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fed-PoisonGAN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="fed_multimodal/Local/results/local_training/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="fed_multimodal/results")
    parser.add_argument("--dataset_dir", type=str, default="fed_multimodal/datasets/ucf101")
    parser.add_argument("--output_dir", type=str, default="fed_multimodal/Local/results/poison_gan_eval/default")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = PoisonGANConfig.from_dict(checkpoint["config"])

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
    generator = build_poison_generator(config)
    trainer = FedPoisonGANTrainer(generator, PoisonDiscriminator(disc_model), config, device=args.device)
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
