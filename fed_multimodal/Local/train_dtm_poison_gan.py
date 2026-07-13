#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(
        description="Train Distributional Temporal Matching GAN on UCF101 features"
    )
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
        default="fed_multimodal/Local/results/dtm_poison_gan",
    )
    parser.add_argument("--exp_name", default="dtm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--hid_size", type=int, default=128)
    parser.add_argument("--att", action="store_true")
    parser.add_argument("--att_name", default="")
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--frame_noise_dim", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=3e-4)
    parser.add_argument("--lr_d", type=float, default=5e-5)
    parser.add_argument("--lambda_distribution", type=float, default=1.0)
    parser.add_argument("--lambda_var_floor", type=float, default=0.25)
    parser.add_argument("--lambda_diversity", type=float, default=0.2)
    parser.add_argument(
        "--target_strategy",
        default="same_as_real",
        choices=["same_as_real", "balanced", "fixed", "mixed"],
    )
    parser.add_argument("--fixed_target", type=int, default=-1)
    parser.add_argument(
        "--freeze_d",
        default="backbone",
        choices=["none", "backbone", "head_only"],
    )
    parser.add_argument(
        "--prototype_path",
        default=None,
        help="Optional server-broadcast ClassEmbeddingBank checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_manager = UCF101LocalDataManager(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = data_manager.get_dataloaders()
    config = DTMGANConfig(
        num_classes=data_manager.num_classes,
        fake_class=data_manager.num_classes,
        audio_seq_len=data_manager.audio_seq_len,
        audio_feat_dim=data_manager.audio_feat_dim,
        video_seq_len=data_manager.video_seq_len,
        video_feat_dim=data_manager.video_feat_dim,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        frame_noise_dim=args.frame_noise_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_distribution=args.lambda_distribution,
        lambda_var_floor=args.lambda_var_floor,
        lambda_diversity=args.lambda_diversity,
        target_strategy=args.target_strategy,
        fixed_target=args.fixed_target,
        freeze_d=args.freeze_d,
        seed=args.seed,
    )
    discriminator_model, _ = build_kplus1_discriminator(
        model_path=args.model_path,
        num_classes=config.num_classes,
        audio_input_dim=config.audio_feat_dim,
        video_input_dim=config.video_feat_dim,
        hid_size=args.hid_size,
        att=args.att,
        att_name=args.att_name,
        freeze=args.freeze_d,
        device=args.device,
    )
    trainer = DTMGANTrainer(
        DTMGenerator(config),
        DTMDiscriminator(discriminator_model),
        config,
        loaders["full_train"],
        args.device,
    )
    if args.prototype_path:
        trainer.load_prototypes(args.prototype_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(
            epoch,
            max_batches=args.max_batches,
            log_interval=args.log_interval,
        )
        eval_metrics = {}
        if args.eval_batches > 0:
            eval_metrics = trainer.evaluate(
                loaders["val"],
                num_batches=args.eval_batches,
            )
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            trainer.save_checkpoint(
                output_dir / f"ckpt_{epoch}_{args.exp_name}.pt",
                epoch,
                eval_metrics or train_metrics,
            )

    final_metrics = history[-1]["eval"] or history[-1]["train"] if history else {}
    final_path = output_dir / f"final_{args.exp_name}.pt"
    trainer.save_checkpoint(final_path, args.epochs, final_metrics)
    torch.save(
        trainer.export_prototypes(),
        output_dir / f"prototypes_{args.exp_name}.pt",
    )
    with open(output_dir / f"history_{args.exp_name}.json", "w") as handle:
        json.dump(
            {"gan_type": "dtm_gan", "config": config.to_dict(), "history": history},
            handle,
            indent=2,
        )
    print(f"Saved final DTM-GAN checkpoint to {final_path}")


if __name__ == "__main__":
    main()
