import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, label_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(label_emb_dim, hidden_dim * 2)

    def forward(self, x, label_emb):
        gamma, beta = self.film(label_emb).chunk(2, dim=-1)
        x = self.linear(self.norm(x))
        x = x * (1.0 + gamma) + beta
        return F.leaky_relu(x, 0.2)


def mask_by_len(x, lengths):
    if lengths is None:
        return x
    max_len = x.size(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x * mask.to(dtype=x.dtype)




class RunningAudioCalibration(nn.Module):
    """Map raw audio features with running real-data statistics, without batch coupling."""

    def __init__(self, feat_dim, momentum=0.95):
        super().__init__()
        self.momentum = float(momentum)
        self.log_scale = nn.Parameter(torch.zeros(feat_dim))
        self.shift = nn.Parameter(torch.zeros(feat_dim))
        self.register_buffer("target_mean", torch.zeros(feat_dim))
        self.register_buffer("target_std", torch.ones(feat_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_target_stats(self, real_audio, lengths=None):
        if lengths is None:
            values = real_audio.reshape(-1, real_audio.size(-1))
        else:
            mask = (
                torch.arange(real_audio.size(1), device=real_audio.device).unsqueeze(0)
                < lengths.to(real_audio.device).unsqueeze(1)
            )
            values = real_audio[mask]
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_std = values.std(dim=0, unbiased=False).clamp_min(1e-3)
        if int(self.num_updates.item()) == 0:
            self.target_mean.copy_(batch_mean)
            self.target_std.copy_(batch_std)
        else:
            self.target_mean.lerp_(batch_mean, 1.0 - self.momentum)
            self.target_std.lerp_(batch_std, 1.0 - self.momentum)
        self.num_updates.add_(1)

    def forward(self, x, lengths=None):
        scale = self.target_std * self.log_scale.clamp(-4.0, 4.0).exp()
        out = x * scale.view(1, 1, -1)
        out = out + self.target_mean.view(1, 1, -1) + self.shift.view(1, 1, -1)
        return mask_by_len(out, lengths)


def _largest_group_divisor(channels, maximum):
    maximum = min(int(maximum), int(channels))
    for groups in range(maximum, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalAdaptivePoisonGenerator(nn.Module):
    """Poison GAN variant with calibrated audio and class-aware temporal video."""

    def __init__(
        self,
        num_classes,
        audio_seq_len=500,
        audio_feat_dim=80,
        video_seq_len=9,
        video_feat_dim=1280,
        z_dim=256,
        label_emb_dim=128,
        hidden_dim=512,
        video_out_max=20.0,
        video_scale_max=8.0,
        frame_noise_dim=64,
        temporal_groups_max=64,
        audio_stats_momentum=0.95,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.audio_seq_len = audio_seq_len
        self.audio_feat_dim = audio_feat_dim
        self.video_seq_len = video_seq_len
        self.video_feat_dim = video_feat_dim
        self.z_dim = z_dim
        self.video_out_max = video_out_max
        self.video_scale_max = video_scale_max
        self.frame_noise_dim = frame_noise_dim

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(z_dim + label_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.film1 = FiLMBlock(hidden_dim, label_emb_dim)
        self.film2 = FiLMBlock(hidden_dim, label_emb_dim)

        self.audio_seed_len = 64
        self.audio_proj = nn.Linear(hidden_dim, 256 * self.audio_seed_len)
        self.audio_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 192, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(192, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.LeakyReLU(0.2),
            nn.Conv1d(96, audio_feat_dim, kernel_size=3, padding=1),
        )
        self.audio_calibration = RunningAudioCalibration(
            audio_feat_dim,
            momentum=audio_stats_momentum,
        )

        decoder_dim = max(hidden_dim * 2, min(video_feat_dim, 1024))
        self.video_pos_emb = nn.Parameter(
            torch.randn(1, video_seq_len, hidden_dim) * 0.02
        )
        self.frame_decoder = nn.Sequential(
            nn.Linear(hidden_dim + frame_noise_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_dim, video_feat_dim),
        )
        temporal_groups = _largest_group_divisor(video_feat_dim, temporal_groups_max)
        self.temporal_conv = nn.Conv1d(
            video_feat_dim,
            video_feat_dim,
            kernel_size=3,
            padding=1,
            groups=temporal_groups,
        )
        self.class_scale = nn.Embedding(num_classes, video_feat_dim)
        self.class_bias = nn.Embedding(num_classes, video_feat_dim)
        nn.init.ones_(self.class_scale.weight)
        nn.init.zeros_(self.class_bias.weight)

    @torch.no_grad()
    def update_real_stats(self, real_audio, lengths=None):
        self.audio_calibration.update_target_stats(real_audio, lengths)

    def forward(self, z, y_target, len_a=None, len_v=None):
        label_emb = self.label_emb(y_target)
        h = self.trunk(torch.cat([z, label_emb], dim=1))
        h = self.film1(h, label_emb)
        h = self.film2(h, label_emb)

        audio = self.audio_proj(h).view(z.size(0), 256, self.audio_seed_len)
        audio = self.audio_conv(audio)
        audio = F.adaptive_avg_pool1d(audio, self.audio_seq_len).transpose(1, 2)
        audio = self.audio_calibration(audio, len_a)

        video_context = h.unsqueeze(1) + self.video_pos_emb
        frame_noise = torch.randn(
            z.size(0),
            self.video_seq_len,
            self.frame_noise_dim,
            device=z.device,
            dtype=z.dtype,
        )
        frames = self.frame_decoder(torch.cat([video_context, frame_noise], dim=-1))
        temporal = self.temporal_conv(frames.transpose(1, 2)).transpose(1, 2)
        frames = frames + temporal
        scale = self.class_scale(y_target).unsqueeze(1)
        scale = scale.clamp(0.1, self.video_scale_max)
        bias = self.class_bias(y_target).unsqueeze(1)
        video = F.relu(frames * scale + bias)
        if self.video_out_max and self.video_out_max > 0:
            video = video.clamp_max(self.video_out_max)
        video = mask_by_len(video, len_v)
        return audio, video


