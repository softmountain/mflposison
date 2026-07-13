#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from fed_multimodal.Local.dataloader import UCF101LocalDataManager
from fed_multimodal.dtm_poison_gan import (
    DTMDiscriminator,
    DTMGANConfig,
    DTMGANTrainer,
    DTMGenerator,
)
from fed_multimodal.poison_gan.kplus1 import build_kplus1_discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DTM-GAN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_path",
        default="fed_multimodal/Local/results/local_training/best_model.pt",
    )
    parser.add_argument("--data_dir", default="fed_multimodal/results")
    parser.add_argument(
        "--dataset_dir",
        default="fed_multimodal/datasets/ucf101",
    )
    parser.add_argument(
        "--output_dir",
        default="fed_multimodal/Local/results/dtm_poison_gan_eval/default",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get("gan_type") != "dtm_gan":
        raise ValueError("checkpoint is not a DTM-GAN checkpoint")
    config = DTMGANConfig.from_dict(checkpoint["config"])
    data_manager = UCF101LocalDataManager(
        args.data_dir,
        args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = data_manager.get_dataloaders()
    discriminator_model, _ = build_kplus1_discriminator(
        model_path=args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        freeze=config.freeze_d,
        device=args.device,
    )
    trainer = DTMGANTrainer(
        DTMGenerator(config),
        DTMDiscriminator(discriminator_model),
        config,
        device=args.device,
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
    loader = loaders["full_train"] if args.use_train else loaders["test"]
    metrics = trainer.evaluate(loader, args.num_batches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analysis_results.json"
    with open(output_path, "w") as handle:
        json.dump(
            {
                "gan_type": "dtm_gan",
                "checkpoint": args.checkpoint,
                "metrics": metrics,
            },
            handle,
            indent=2,
        )
    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
