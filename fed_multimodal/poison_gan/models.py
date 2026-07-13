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


def masked_mean_torch(x, lengths=None, dim=1, eps=1e-6):
    if lengths is None:
        return x.mean(dim=dim)
    max_len = x.size(dim)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
    mask = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def apply_per_sample_znorm(x, lengths=None, eps=1e-5, clamp_value=3.0):
    if lengths is None:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return ((x - mean) / std).clamp(-clamp_value, clamp_value)
    mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
    mask = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / denom
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
    out = (x - mean) / var.sqrt().clamp_min(eps)
    return (out.clamp(-clamp_value, clamp_value) * mask)


class PoisonFeatureGenerator(nn.Module):
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
        audio_out_max=3.0,
        video_out_max=20.0,
        video_scale_max=8.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.audio_seq_len = audio_seq_len
        self.audio_feat_dim = audio_feat_dim
        self.video_seq_len = video_seq_len
        self.video_feat_dim = video_feat_dim
        self.z_dim = z_dim
        self.audio_out_max = audio_out_max
        self.video_out_max = video_out_max
        self.video_scale_max = video_scale_max

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
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.LeakyReLU(0.2),
            nn.Conv1d(96, audio_feat_dim, kernel_size=3, padding=1),
        )

        self.video_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, video_seq_len * video_feat_dim),
        )
        self.video_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, y_target, len_a=None, len_v=None):
        label_emb = self.label_emb(y_target)
        h = self.trunk(torch.cat([z, label_emb], dim=1))
        h = self.film1(h, label_emb)
        h = self.film2(h, label_emb)

        audio = self.audio_proj(h).view(z.size(0), 256, self.audio_seed_len)
        audio = self.audio_conv(audio)
        audio = F.adaptive_avg_pool1d(audio, self.audio_seq_len).transpose(1, 2)
        audio = apply_per_sample_znorm(audio, len_a, clamp_value=self.audio_out_max)

        video = self.video_decoder(h).view(z.size(0), self.video_seq_len, self.video_feat_dim)
        scale = self.video_scale.clamp(0.1, self.video_scale_max)
        video = F.relu(video) * scale
        video = video.clamp(0.0, self.video_out_max)
        video = mask_by_len(video, len_v)
        return audio, video


class PoisonDiscriminator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, video, len_audio=None, len_video=None, return_embed=False):
        logits, emb = self.model(audio, video, len_audio, len_video)
        if return_embed:
            return logits, emb
        return logits
