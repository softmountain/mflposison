#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Train Fed-PoisonGAN K+1 discriminator on UCF101 features")
    parser.add_argument("--model_path", type=str, default="fed_multimodal/Local/results/local_training/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="fed_multimodal/results")
    parser.add_argument("--dataset_dir", type=str, default="fed_multimodal/datasets/ucf101")
    parser.add_argument("--output_dir", type=str, default="fed_multimodal/Local/results/poison_gan")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--gan_variant", type=str, default="temporal_adaptive", choices=PoisonGANConfig.VARIANTS)
    parser.add_argument("--prototype_bank", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--hid_size", type=int, default=128)
    parser.add_argument("--att", action="store_true")
    parser.add_argument("--att_name", type=str, default="")
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    parser.add_argument("--lambda_d_fake", type=float, default=None)
    parser.add_argument("--lambda_avoid", type=float, default=None)
    parser.add_argument("--lambda_div", type=float, default=None)
    parser.add_argument("--target_strategy", type=str, default="same_as_real", choices=["same_as_real", "balanced", "fixed", "mixed"])
    parser.add_argument("--fixed_target", type=int, default=-1)
    parser.add_argument("--freeze_d", type=str, default="none", choices=["none", "backbone", "head_only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dm = UCF101LocalDataManager(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = dm.get_dataloaders()

    config = PoisonGANConfig.for_variant(
        args.gan_variant,
        num_classes=dm.num_classes,
        fake_class=dm.num_classes,
        audio_seq_len=dm.audio_seq_len,
        audio_feat_dim=dm.audio_feat_dim,
        video_seq_len=dm.video_seq_len,
        video_feat_dim=dm.video_feat_dim,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        target_strategy=args.target_strategy,
        fixed_target=args.fixed_target,
        freeze_d=args.freeze_d,
        seed=args.seed,
    )
    if args.lambda_d_fake is not None:
        config.lambda_d_fake = args.lambda_d_fake
    if args.lambda_avoid is not None:
        config.lambda_avoid = args.lambda_avoid
    if args.lambda_div is not None:
        config.lambda_div = args.lambda_div

    discriminator_model, teacher_checkpoint = build_kplus1_discriminator(
        model_path=args.model_path,
        num_classes=dm.num_classes,
        audio_input_dim=dm.audio_feat_dim,
        video_input_dim=dm.video_feat_dim,
        hid_size=args.hid_size,
        att=args.att,
        att_name=args.att_name,
        freeze=args.freeze_d,
        device=args.device,
    )
    discriminator = PoisonDiscriminator(discriminator_model)
    generator = build_poison_generator(config)
    trainer = FedPoisonGANTrainer(generator, discriminator, config, loaders["full_train"], args.device)
    if args.prototype_bank:
        trainer.load_prototypes(args.prototype_bank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(epoch=epoch, max_batches=args.max_batches, log_interval=args.log_interval)
        eval_metrics = trainer.evaluate(loaders["val"], num_batches=min(args.max_batches or 20, 20))
        row = {"epoch": epoch, "train": train_metrics, "eval": eval_metrics}
        history.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            trainer.save_checkpoint(output_dir / f"ckpt_{epoch}_{args.exp_name}.pt", epoch, eval_metrics)

    final_path = output_dir / f"final_{args.exp_name}.pt"
    trainer.save_checkpoint(final_path, args.epochs, history[-1]["eval"] if history else {})
    if history:
        trainer.save_checkpoint(output_dir / f"ckpt_{args.epochs}_{args.exp_name}.pt", args.epochs, history[-1]["eval"])
    with open(output_dir / f"history_{args.exp_name}.json", "w") as f:
        json.dump({"config": config.to_dict(), "history": history}, f, indent=2)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
