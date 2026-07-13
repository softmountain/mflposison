from pathlib import Path

import torch

from fed_multimodal.poison_gan.kplus1 import trainable_parameters
from fed_multimodal.poison_gan.memory_bank import ClassEmbeddingBank
from fed_multimodal.poison_gan.metrics import (
    classification_metrics,
    diversity_ratio,
    embedding_gaps,
    finalize_metric_sums,
    merge_metric_sums,
    tensor_stats,
)

from .losses import discriminator_loss, generator_loss


class DTMGANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        config,
        dataloader=None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.config = config
        self.dataloader = dataloader
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(0.5, 0.999),
        )
        self.opt_d = torch.optim.Adam(
            trainable_parameters(self.discriminator),
            lr=config.lr_d,
            betas=(0.5, 0.999),
        )
        self.bank = ClassEmbeddingBank(
            config.num_classes,
            momentum=0.9,
            device=self.device,
        )

    def sample_targets(self, real_labels):
        strategy = self.config.target_strategy
        if strategy == "same_as_real":
            return real_labels.clone()
        if strategy == "balanced":
            return torch.randint(
                0,
                self.config.num_classes,
                real_labels.shape,
                device=real_labels.device,
            )
        if strategy == "fixed":
            if self.config.fixed_target < 0:
                raise ValueError("fixed_target must be set for fixed strategy")
            return torch.full_like(real_labels, int(self.config.fixed_target))
        if strategy == "mixed":
            if self.config.fixed_target < 0:
                raise ValueError("fixed_target must be set for mixed strategy")
            fixed = torch.full_like(real_labels, int(self.config.fixed_target))
            random_targets = torch.randint(
                0,
                self.config.num_classes,
                real_labels.shape,
                device=real_labels.device,
            )
            fixed_mask = (
                torch.rand(real_labels.shape, device=real_labels.device)
                < self.config.mixed_target_prob
            )
            return torch.where(fixed_mask, fixed, random_targets)
        raise ValueError(f"Unknown target strategy: {strategy}")

    def _move_batch(self, batch):
        audio, video, len_audio, len_video, labels = batch
        return (
            audio.float().to(self.device),
            video.float().to(self.device),
            len_audio.long().to(self.device),
            len_video.long().to(self.device),
            labels.long().to(self.device),
        )

    def _aligned_lengths(self, len_audio, len_video):
        return (
            len_audio.clamp(0, self.config.audio_seq_len),
            len_video.clamp(0, self.config.video_seq_len),
        )

    def load_prototypes(self, path_or_state):
        """Load server-broadcast class embedding mean/variance statistics."""
        state = path_or_state
        if isinstance(path_or_state, (str, Path)):
            state = torch.load(path_or_state, map_location=self.device)
        if "embedding_bank" in state:
            state = state["embedding_bank"]
        self.bank.load_state_dict(state)

    def export_prototypes(self):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in self.bank.state_dict().items()
        }

    def train_d_step(self, batch):
        real_audio, real_video, len_audio, len_video, real_labels = (
            self._move_batch(batch)
        )
        fake_len_audio, fake_len_video = self._aligned_lengths(
            len_audio,
            len_video,
        )
        target_labels = self.sample_targets(real_labels)
        z = torch.randn(
            real_labels.size(0),
            self.config.z_dim,
            device=self.device,
        )
        self.generator.update_real_stats(real_audio, fake_len_audio)
        with torch.no_grad():
            fake_audio, fake_video = self.generator(
                z,
                target_labels,
                fake_len_audio,
                fake_len_video,
            )

        self.opt_d.zero_grad(set_to_none=True)
        logits_real, real_embeddings = self.discriminator(
            real_audio,
            real_video,
            len_audio,
            len_video,
            return_embed=True,
        )
        logits_fake = self.discriminator(
            fake_audio,
            fake_video,
            fake_len_audio,
            fake_len_video,
        )
        loss, metrics = discriminator_loss(
            logits_real,
            real_labels,
            logits_fake,
            self.config.fake_class,
            self.config.lambda_d_fake,
        )
        loss.backward()
        self.opt_d.step()
        self.bank.update(real_embeddings, real_labels)
        return metrics

    def train_g_step(self, batch, epoch=0):
        real_audio, real_video, len_audio, len_video, real_labels = (
            self._move_batch(batch)
        )
        fake_len_audio, fake_len_video = self._aligned_lengths(
            len_audio,
            len_video,
        )
        target_labels = self.sample_targets(real_labels)
        z = torch.randn(
            real_labels.size(0),
            self.config.z_dim,
            device=self.device,
        )
        with torch.no_grad():
            _, real_embeddings = self.discriminator(
                real_audio,
                real_video,
                len_audio,
                len_video,
                return_embed=True,
            )
            self.bank.update(real_embeddings, real_labels)

        parameter_grad_states = [
            parameter.requires_grad for parameter in self.discriminator.parameters()
        ]
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            self.opt_g.zero_grad(set_to_none=True)
            fake_audio, fake_video = self.generator(
                z,
                target_labels,
                fake_len_audio,
                fake_len_video,
            )
            logits_fake, fake_embeddings = self.discriminator(
                fake_audio,
                fake_video,
                fake_len_audio,
                fake_len_video,
                return_embed=True,
            )
            loss, metrics = generator_loss(
                logits_fake=logits_fake,
                fake_embeddings=fake_embeddings,
                target_labels=target_labels,
                config=self.config,
                real_embeddings=real_embeddings,
                real_labels=real_labels,
                bank=self.bank,
                generator=self.generator,
                z=z,
                fake_audio=fake_audio,
                fake_video=fake_video,
                real_audio=real_audio,
                real_video=real_video,
                len_audio=fake_len_audio,
                len_video=fake_len_video,
                epoch=epoch,
            )
            loss.backward()
            self.opt_g.step()
        finally:
            for parameter, requires_grad in zip(
                self.discriminator.parameters(),
                parameter_grad_states,
            ):
                parameter.requires_grad_(requires_grad)
        return metrics

    def train_epoch(self, epoch=0, max_batches=None, log_interval=20):
        self.generator.train()
        self.discriminator.train()
        accumulated = {}
        for batch_index, batch in enumerate(self.dataloader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            for _ in range(self.config.d_steps):
                d_metrics = self.train_d_step(batch)
            for _ in range(self.config.g_steps):
                g_metrics = self.train_g_step(batch, epoch)
            metrics = {**d_metrics, **g_metrics}
            merge_metric_sums(accumulated, metrics)
            if log_interval and batch_index % log_interval == 0:
                summary = " ".join(
                    f"{key}={value:.4f}" for key, value in metrics.items()
                )
                print(f"epoch={epoch} batch={batch_index} {summary}")
        return finalize_metric_sums(accumulated)

    @torch.no_grad()
    def evaluate(self, dataloader=None, num_batches=None):
        dataloader = dataloader or self.dataloader
        self.generator.eval()
        self.discriminator.eval()
        accumulated = {}
        for batch_index, batch in enumerate(dataloader, start=1):
            if num_batches is not None and batch_index > num_batches:
                break
            real_audio, real_video, len_audio, len_video, real_labels = (
                self._move_batch(batch)
            )
            fake_len_audio, fake_len_video = self._aligned_lengths(
                len_audio,
                len_video,
            )
            target_labels = self.sample_targets(real_labels)
            z = torch.randn(
                real_labels.size(0),
                self.config.z_dim,
                device=self.device,
            )
            fake_audio, fake_video = self.generator(
                z,
                target_labels,
                fake_len_audio,
                fake_len_video,
            )
            logits_fake, fake_embeddings = self.discriminator(
                fake_audio,
                fake_video,
                fake_len_audio,
                fake_len_video,
                return_embed=True,
            )
            _, real_embeddings = self.discriminator(
                real_audio,
                real_video,
                len_audio,
                len_video,
                return_embed=True,
            )
            metrics = classification_metrics(
                logits_fake,
                target_labels,
                self.config.num_classes,
                self.config.fake_class,
            )
            metrics.update(tensor_stats(real_audio, "real_audio"))
            metrics.update(tensor_stats(fake_audio, "fake_audio"))
            metrics.update(tensor_stats(real_video, "real_video"))
            metrics.update(tensor_stats(fake_video, "fake_video"))
            metrics["audio_diversity_ratio"] = diversity_ratio(
                fake_audio,
                real_audio,
                target_labels,
            )
            metrics["video_diversity_ratio"] = diversity_ratio(
                fake_video,
                real_video,
                target_labels,
            )
            metrics.update(embedding_gaps(fake_embeddings, real_embeddings))
            merge_metric_sums(accumulated, metrics, n=real_labels.size(0))
        return finalize_metric_sums(accumulated)

    @torch.no_grad()
    def generate(
        self,
        num_samples,
        target_labels=None,
        batch_size=64,
        train_labels=None,
    ):
        self.generator.eval()
        parts = {
            key: []
            for key in (
                "audio",
                "video",
                "len_a",
                "len_v",
                "condition_label",
                "train_label",
            )
        }
        for start in range(0, num_samples, batch_size):
            size = min(batch_size, num_samples - start)
            if target_labels is None:
                labels = torch.randint(
                    0,
                    self.config.num_classes,
                    (size,),
                    device=self.device,
                )
            else:
                labels = target_labels[start : start + size].to(self.device)
            len_audio = torch.full(
                (size,),
                self.config.audio_seq_len,
                dtype=torch.long,
                device=self.device,
            )
            len_video = torch.full(
                (size,),
                self.config.video_seq_len,
                dtype=torch.long,
                device=self.device,
            )
            z = torch.randn(size, self.config.z_dim, device=self.device)
            audio, video = self.generator(
                z,
                labels,
                len_audio,
                len_video,
            )
            output_labels = (
                labels
                if train_labels is None
                else train_labels[start : start + size].to(self.device)
            )
            parts["audio"].append(audio.cpu())
            parts["video"].append(video.cpu())
            parts["len_a"].append(len_audio.cpu())
            parts["len_v"].append(len_video.cpu())
            parts["condition_label"].append(labels.cpu())
            parts["train_label"].append(output_labels.cpu())
        return {key: torch.cat(value, dim=0) for key, value in parts.items()}

    def save_checkpoint(self, path, epoch, metrics=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gan_type": "dtm_gan",
                "epoch": epoch,
                "config": self.config.to_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.opt_g.state_dict(),
                "optimizer_d_state_dict": self.opt_d.state_dict(),
                "embedding_bank": self.bank.state_dict(),
                "metrics": metrics or {},
            },
            path,
        )

    def load_checkpoint(self, path, load_optimizers=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"],
            strict=False,
        )
        if load_optimizers:
            if "optimizer_g_state_dict" in checkpoint:
                self.opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            if "optimizer_d_state_dict" in checkpoint:
                self.opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if "embedding_bank" in checkpoint:
            self.bank.load_state_dict(checkpoint["embedding_bank"])
        return checkpoint
