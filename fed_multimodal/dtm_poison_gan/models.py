import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_by_length(x, lengths):
    if lengths is None:
        return x
    mask = (
        torch.arange(x.size(1), device=x.device).unsqueeze(0)
        < lengths.to(x.device).unsqueeze(1)
    )
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x * mask.to(dtype=x.dtype)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(condition_dim, hidden_dim * 2)

    def forward(self, x, condition):
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        x = self.proj(self.norm(x))
        return F.leaky_relu(x * (1.0 + gamma) + beta, 0.2)


class RunningFeatureCalibration(nn.Module):
    """Restore real per-dimension scale without per-sample z-normalization."""

    def __init__(self, feat_dim, momentum=0.95):
        super().__init__()
        self.momentum = float(momentum)
        self.log_scale = nn.Parameter(torch.zeros(feat_dim))
        self.shift = nn.Parameter(torch.zeros(feat_dim))
        self.register_buffer("target_mean", torch.zeros(feat_dim))
        self.register_buffer("target_std", torch.ones(feat_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, real_features, lengths=None):
        if lengths is None:
            values = real_features.reshape(-1, real_features.size(-1))
        else:
            mask = (
                torch.arange(real_features.size(1), device=real_features.device)
                .unsqueeze(0)
                < lengths.to(real_features.device).unsqueeze(1)
            )
            values = real_features[mask]
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_std = values.std(dim=0, unbiased=False).clamp_min(1e-3)
        if int(self.num_updates.item()) == 0:
            self.target_mean.copy_(batch_mean)
            self.target_std.copy_(batch_std)
        else:
            update_weight = 1.0 - self.momentum
            self.target_mean.lerp_(batch_mean, update_weight)
            self.target_std.lerp_(batch_std, update_weight)
        self.num_updates.add_(1)

    def forward(self, x, lengths=None):
        learned_scale = self.log_scale.clamp(-4.0, 4.0).exp()
        out = x * (self.target_std * learned_scale).view(1, 1, -1)
        out = out + (self.target_mean + self.shift).view(1, 1, -1)
        return mask_by_length(out, lengths)


class DTMGenerator(nn.Module):
    """Calibrated audio plus sparse, noise-injected temporal video generator."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        self.label_embedding = nn.Embedding(
            config.num_classes,
            config.label_emb_dim,
        )
        self.trunk = nn.Sequential(
            nn.Linear(config.z_dim + config.label_emb_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.film1 = FiLMBlock(config.hidden_dim, config.label_emb_dim)
        self.film2 = FiLMBlock(config.hidden_dim, config.label_emb_dim)

        self.audio_seed_len = 64
        self.audio_projection = nn.Linear(
            config.hidden_dim,
            256 * self.audio_seed_len,
        )
        self.audio_decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 192, 4, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(192, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.LeakyReLU(0.2),
            nn.Conv1d(96, config.audio_feat_dim, 3, padding=1),
        )
        self.audio_calibration = RunningFeatureCalibration(
            config.audio_feat_dim,
            config.audio_stats_momentum,
        )

        self.video_positions = nn.Parameter(
            torch.randn(1, config.video_seq_len, config.hidden_dim) * 0.02
        )
        self.temporal_decoder = nn.GRU(
            input_size=config.hidden_dim + config.frame_noise_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.video_gate = nn.Linear(config.hidden_dim, config.video_feat_dim)
        self.video_magnitude = nn.Linear(config.hidden_dim, config.video_feat_dim)
        self.video_scale = nn.Embedding(config.num_classes, config.video_feat_dim)
        self.video_shift = nn.Embedding(config.num_classes, config.video_feat_dim)
        nn.init.ones_(self.video_scale.weight)
        nn.init.zeros_(self.video_shift.weight)

    @torch.no_grad()
    def update_real_stats(self, real_audio, lengths=None):
        self.audio_calibration.update(real_audio, lengths)

    def forward(self, z, labels, len_audio=None, len_video=None):
        label_embedding = self.label_embedding(labels)
        hidden = self.trunk(torch.cat([z, label_embedding], dim=-1))
        hidden = self.film1(hidden, label_embedding)
        hidden = self.film2(hidden, label_embedding)

        audio = self.audio_projection(hidden).view(
            z.size(0),
            256,
            self.audio_seed_len,
        )
        audio = self.audio_decoder(audio)
        audio = F.interpolate(
            audio,
            size=self.config.audio_seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        audio = self.audio_calibration(audio, len_audio)

        frame_noise = torch.randn(
            z.size(0),
            self.config.video_seq_len,
            self.config.frame_noise_dim,
            device=z.device,
            dtype=z.dtype,
        )
        context = hidden.unsqueeze(1) + self.video_positions
        temporal, _ = self.temporal_decoder(
            torch.cat([context, frame_noise], dim=-1)
        )
        gate = torch.sigmoid(self.video_gate(temporal))
        magnitude = F.softplus(self.video_magnitude(temporal))
        scale = self.video_scale(labels).clamp(
            0.1,
            self.config.video_scale_max,
        ).unsqueeze(1)
        shift = self.video_shift(labels).unsqueeze(1)
        video = F.relu(gate * magnitude * scale + shift)
        if self.config.video_out_max > 0:
            video = video.clamp_max(self.config.video_out_max)
        video = mask_by_length(video, len_video)
        return audio, video


class DTMDiscriminator(nn.Module):
    """Adapter around the project's K+1 teacher-derived classifier."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        audio,
        video,
        len_audio=None,
        len_video=None,
        return_embed=False,
    ):
        logits, embedding = self.model(
            audio,
            video,
            len_audio,
            len_video,
        )
        if return_embed:
            return logits, embedding
        return logits
