from pathlib import Path

import torch

from fed_multimodal.poison_gan.kplus1 import trainable_parameters
from .losses import discriminator_loss, generator_loss, r1_gradient_penalty
from fed_multimodal.poison_gan.memory_bank import ClassEmbeddingBank
from fed_multimodal.poison_gan.metrics import (
    classification_metrics,
    diversity_ratio,
    embedding_gaps,
    finalize_metric_sums,
    merge_metric_sums,
    tensor_stats,
)


class TemporalAdaptiveGANTrainer:
    def __init__(self, generator, discriminator, config, dataloader=None, device="cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(trainable_parameters(self.discriminator), lr=config.lr_d, betas=(0.5, 0.999))
        self.bank = ClassEmbeddingBank(config.num_classes, momentum=0.9, device=device)
        self._d_updates = 0

    def sample_targets(self, y_real):
        strategy = self.config.target_strategy
        if strategy == "same_as_real":
            return y_real.clone()
        if strategy == "balanced":
            return torch.randint(0, self.config.num_classes, y_real.shape, device=y_real.device)
        if strategy == "fixed":
            if self.config.fixed_target < 0:
                raise ValueError("fixed_target must be set for fixed target strategy")
            return torch.full_like(y_real, int(self.config.fixed_target))
        if strategy == "mixed":
            if self.config.fixed_target < 0:
                raise ValueError("fixed_target must be set for mixed target strategy")
            fixed = torch.full_like(y_real, int(self.config.fixed_target))
            random_targets = torch.randint(0, self.config.num_classes, y_real.shape, device=y_real.device)
            mask = torch.rand(y_real.shape, device=y_real.device) < self.config.mixed_target_prob
            return torch.where(mask, fixed, random_targets)
        raise ValueError(f"Unknown target strategy: {strategy}")

    def _move_batch(self, batch):
        real_audio, real_video, len_a, len_v, y_real = batch
        return (
            real_audio.float().to(self.device),
            real_video.float().to(self.device),
            len_a.long().to(self.device),
            len_v.long().to(self.device),
            y_real.long().to(self.device),
        )

    def _fake_lengths(self, len_a, len_v):
        return (
            len_a.clamp(min=0, max=self.config.audio_seq_len),
            len_v.clamp(min=0, max=self.config.video_seq_len),
        )

    def _instance_noise_std(self, epoch):
        start = float(self.config.instance_noise_std)
        decay_epochs = max(1, int(self.config.instance_noise_decay_epochs))
        return start * max(0.0, 1.0 - float(epoch) / float(decay_epochs))

    @staticmethod
    def _add_instance_noise(x, lengths, std):
        if std <= 0:
            return x
        noisy = x + torch.randn_like(x) * std
        mask = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0)
            < lengths.to(x.device).unsqueeze(1)
        )
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        return noisy * mask.to(dtype=noisy.dtype)

    def load_prototypes(self, path_or_state):
        """Seed the local class bank from a server-broadcast prototype state."""
        state = path_or_state
        if isinstance(path_or_state, (str, Path)):
            state = torch.load(path_or_state, map_location=self.device)
        if "embedding_bank" in state:
            state = state["embedding_bank"]
        self.bank.load_state_dict(state)

    def export_prototypes(self):
        state = self.bank.state_dict()
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in state.items()
        }

    def train_d_step(self, batch, epoch=0):
        real_audio, real_video, len_a, len_v, y_real = self._move_batch(batch)
        y_target = self.sample_targets(y_real)
        z = torch.randn(y_real.size(0), self.config.z_dim, device=self.device)
        fake_len_a, fake_len_v = self._fake_lengths(len_a, len_v)
        if hasattr(self.generator, "update_real_stats"):
            self.generator.update_real_stats(real_audio, len_a)
        with torch.no_grad():
            fake_audio, fake_video = self.generator(z, y_target, fake_len_a, fake_len_v)

        noise_std = self._instance_noise_std(epoch)
        real_audio_d = self._add_instance_noise(real_audio, len_a, noise_std)
        real_video_d = self._add_instance_noise(real_video, len_v, noise_std)
        fake_audio_d = self._add_instance_noise(fake_audio.detach(), fake_len_a, noise_std)
        fake_video_d = self._add_instance_noise(fake_video.detach(), fake_len_v, noise_std)
        use_r1 = (
            self.config.r1_gamma > 0
            and self._d_updates % self.config.r1_interval == 0
        )
        if use_r1:
            real_audio_d = real_audio_d.detach().requires_grad_(True)
            real_video_d = real_video_d.detach().requires_grad_(True)

        self.opt_d.zero_grad(set_to_none=True)
        logits_real, emb_real = self.discriminator(
            real_audio_d,
            real_video_d,
            len_a,
            len_v,
            return_embed=True,
        )
        logits_fake = self.discriminator(
            fake_audio_d,
            fake_video_d,
            fake_len_a,
            fake_len_v,
        )
        loss, metrics = discriminator_loss(
            logits_real,
            y_real,
            logits_fake,
            self.config.fake_class,
            self.config.lambda_d_fake,
        )
        loss_r1 = loss.new_tensor(0.0)
        if use_r1:
            loss_r1 = r1_gradient_penalty(
                logits_real,
                y_real,
                [real_audio_d, real_video_d],
                [len_a, len_v],
                gamma=self.config.r1_gamma,
            )
            loss = loss + loss_r1
        metrics["d_r1"] = float(loss_r1.detach().cpu())
        metrics["instance_noise_std"] = float(noise_std)
        loss.backward()
        self.opt_d.step()
        self._d_updates += 1
        self.bank.update(emb_real.detach(), y_real.detach())
        return metrics

    def train_g_step(self, batch, epoch=0):
        real_audio, real_video, len_a, len_v, y_real = self._move_batch(batch)
        y_target = self.sample_targets(y_real)
        z = torch.randn(y_real.size(0), self.config.z_dim, device=self.device)
        fake_len_a, fake_len_v = self._fake_lengths(len_a, len_v)

        self.opt_g.zero_grad(set_to_none=True)
        fake_audio, fake_video = self.generator(z, y_target, fake_len_a, fake_len_v)
        with torch.no_grad():
            _, emb_real = self.discriminator(real_audio, real_video, len_a, len_v, return_embed=True)
            self.bank.update(emb_real.detach(), y_real.detach())
        logits_fake, emb_fake = self.discriminator(fake_audio, fake_video, fake_len_a, fake_len_v, return_embed=True)
        loss, metrics = generator_loss(
            logits_fake=logits_fake,
            emb_fake=emb_fake,
            y_target=y_target,
            fake_class=self.config.fake_class,
            config=self.config,
            emb_real=emb_real,
            y_real=y_real,
            bank=self.bank,
            generator=self.generator,
            z=z,
            len_a=fake_len_a,
            len_v=fake_len_v,
            fake_audio=fake_audio,
            fake_video=fake_video,
            real_audio=real_audio,
            real_video=real_video,
            epoch=epoch,
        )
        loss.backward()
        self.opt_g.step()
        return metrics

    def train_epoch(self, epoch=0, max_batches=None, log_interval=20):
        self.generator.train()
        self.discriminator.train()
        accum = {}
        for batch_idx, batch in enumerate(self.dataloader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            for _ in range(self.config.d_steps):
                d_metrics = self.train_d_step(batch, epoch=epoch)
            for _ in range(self.config.g_steps):
                g_metrics = self.train_g_step(batch, epoch=epoch)
            metrics = {**d_metrics, **g_metrics}
            merge_metric_sums(accum, metrics)
            if log_interval and batch_idx % log_interval == 0:
                print(f"epoch={epoch} batch={batch_idx} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return finalize_metric_sums(accum)

    @torch.no_grad()
    def evaluate(self, dataloader=None, num_batches=None):
        dataloader = dataloader or self.dataloader
        self.generator.eval()
        self.discriminator.eval()
        accum = {}
        for batch_idx, batch in enumerate(dataloader, start=1):
            if num_batches is not None and batch_idx > num_batches:
                break
            real_audio, real_video, len_a, len_v, y_real = self._move_batch(batch)
            y_target = self.sample_targets(y_real)
            z = torch.randn(y_real.size(0), self.config.z_dim, device=self.device)
            fake_len_a, fake_len_v = self._fake_lengths(len_a, len_v)
            fake_audio, fake_video = self.generator(z, y_target, fake_len_a, fake_len_v)
            logits_fake, emb_fake = self.discriminator(fake_audio, fake_video, fake_len_a, fake_len_v, return_embed=True)
            _, emb_real = self.discriminator(real_audio, real_video, len_a, len_v, return_embed=True)
            metrics = classification_metrics(logits_fake, y_target, self.config.num_classes, self.config.fake_class)
            metrics.update(tensor_stats(real_audio, "real_audio"))
            metrics.update(tensor_stats(fake_audio, "fake_audio"))
            metrics.update(tensor_stats(real_video, "real_video"))
            metrics.update(tensor_stats(fake_video, "fake_video"))
            metrics["audio_diversity_ratio"] = diversity_ratio(fake_audio, real_audio, y_target)
            metrics["video_diversity_ratio"] = diversity_ratio(fake_video, real_video, y_target)
            metrics.update(embedding_gaps(emb_fake, emb_real))
            merge_metric_sums(accum, metrics, n=y_real.size(0))
        return finalize_metric_sums(accum)

    @torch.no_grad()
    def generate(self, num_samples, target_labels=None, batch_size=64, train_labels=None):
        self.generator.eval()
        audio_parts, video_parts, len_a_parts, len_v_parts, cond_parts, label_parts = [], [], [], [], [], []
        remaining = num_samples
        while remaining > 0:
            bsz = min(batch_size, remaining)
            if target_labels is None:
                y_target = torch.randint(0, self.config.num_classes, (bsz,), device=self.device)
            else:
                start = num_samples - remaining
                y_target = target_labels[start:start + bsz].to(self.device)
            len_a = torch.full((bsz,), self.config.audio_seq_len, dtype=torch.long, device=self.device)
            len_v = torch.full((bsz,), self.config.video_seq_len, dtype=torch.long, device=self.device)
            z = torch.randn(bsz, self.config.z_dim, device=self.device)
            audio, video = self.generator(z, y_target, len_a, len_v)
            if train_labels is None:
                labels = y_target
            else:
                start = num_samples - remaining
                labels = train_labels[start:start + bsz].to(self.device)
            audio_parts.append(audio.cpu())
            video_parts.append(video.cpu())
            len_a_parts.append(len_a.cpu())
            len_v_parts.append(len_v.cpu())
            cond_parts.append(y_target.cpu())
            label_parts.append(labels.cpu())
            remaining -= bsz
        return {
            "audio": torch.cat(audio_parts, dim=0),
            "video": torch.cat(video_parts, dim=0),
            "len_a": torch.cat(len_a_parts, dim=0),
            "len_v": torch.cat(len_v_parts, dim=0),
            "condition_label": torch.cat(cond_parts, dim=0),
            "train_label": torch.cat(label_parts, dim=0),
        }

    def save_checkpoint(self, path, epoch, metrics=None, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.opt_g.state_dict(),
                "optimizer_d_state_dict": self.opt_d.state_dict(),
                "embedding_bank": self.bank.state_dict(),
                "metrics": metrics or {},
                "extra": extra or {},
            },
            path,
        )

    def load_checkpoint(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)
        if load_optimizers:
            if "optimizer_g_state_dict" in checkpoint:
                self.opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            if "optimizer_d_state_dict" in checkpoint:
                self.opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if "embedding_bank" in checkpoint:
            self.bank.load_state_dict(checkpoint["embedding_bank"])
        return checkpoint
