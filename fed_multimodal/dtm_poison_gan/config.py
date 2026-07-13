from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DTMGANConfig:
    """Configuration for Distributional Temporal Matching GAN."""

    num_classes: int
    fake_class: int
    audio_seq_len: int = 500
    audio_feat_dim: int = 80
    video_seq_len: int = 9
    video_feat_dim: int = 1280
    z_dim: int = 256
    label_emb_dim: int = 128
    hidden_dim: int = 512
    frame_noise_dim: int = 64
    lr_g: float = 3e-4
    lr_d: float = 5e-5
    d_steps: int = 1
    g_steps: int = 3
    lambda_d_fake: float = 0.5
    lambda_adv: float = 0.2
    lambda_avoid: float = 0.1
    lambda_distribution: float = 1.0
    lambda_var_floor: float = 0.25
    lambda_raw_stat: float = 0.1
    lambda_audio_tail: float = 0.1
    lambda_diversity: float = 0.2
    diversity_start_epoch: int = 3
    diversity_warmup_epochs: int = 5
    audio_stats_momentum: float = 0.95
    video_out_max: float = 20.0
    video_scale_max: float = 8.0
    target_strategy: str = "same_as_real"
    fixed_target: int = -1
    mixed_target_prob: float = 0.7
    freeze_d: str = "backbone"
    seed: int = 42

    def __post_init__(self):
        if self.diversity_warmup_epochs < 1:
            raise ValueError("diversity_warmup_epochs must be at least 1")
        if self.audio_seq_len < 1 or self.video_seq_len < 1:
            raise ValueError("sequence lengths must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DTMGANConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})
