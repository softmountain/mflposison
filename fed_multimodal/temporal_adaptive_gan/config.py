from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict


@dataclass
class TemporalAdaptiveGANConfig:
    VARIANTS: ClassVar[tuple] = ("temporal_adaptive",)

    num_classes: int
    fake_class: int
    gan_variant: str = "temporal_adaptive"
    audio_seq_len: int = 500
    audio_feat_dim: int = 80
    video_seq_len: int = 9
    video_feat_dim: int = 1280
    z_dim: int = 256
    label_emb_dim: int = 128
    hidden_dim: int = 512
    lr_g: float = 3e-4
    lr_d: float = 5e-5
    d_steps: int = 1
    g_steps: int = 3
    lambda_d_fake: float = 1.0
    lambda_adv: float = 0.5
    lambda_avoid: float = 0.3
    lambda_fm: float = 0.5
    lambda_var: float = 0.3
    lambda_div: float = 0.2
    lambda_stat: float = 0.2
    lambda_audio_dist: float = 0.1
    audio_kurtosis_weight: float = 0.1
    lambda_mod: float = 0.0
    audio_out_max: float = 3.0
    video_out_max: float = 20.0
    video_scale_max: float = 8.0
    warmup_epochs: int = 5
    diversity_start_epoch: int = 3
    diversity_warmup_epochs: int = 5
    r1_gamma: float = 10.0
    r1_interval: int = 16
    instance_noise_std: float = 0.1
    instance_noise_decay_epochs: int = 50
    audio_stats_momentum: float = 0.95
    frame_noise_dim: int = 64
    temporal_groups_max: int = 64
    target_strategy: str = "same_as_real"
    fixed_target: int = -1
    mixed_target_prob: float = 0.7
    freeze_d: str = "none"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalAdaptiveGANConfig":
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)

    def __post_init__(self):
        self.gan_variant = str(self.gan_variant).lower()
        if self.gan_variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown GAN variant: {self.gan_variant}. "
                f"Expected one of {', '.join(self.VARIANTS)}"
            )
        if self.r1_interval < 1:
            raise ValueError("r1_interval must be at least 1")
        if self.diversity_warmup_epochs < 1:
            raise ValueError("diversity_warmup_epochs must be at least 1")

    @classmethod
    def for_variant(cls, gan_variant: str, **kwargs) -> "TemporalAdaptiveGANConfig":
        """Build a config with tuned defaults while preserving legacy behavior."""
        gan_variant = str(gan_variant).lower()
        if gan_variant not in cls.VARIANTS:
            raise ValueError(
                f"Unknown GAN variant: {gan_variant}. "
                f"Expected one of {', '.join(cls.VARIANTS)}"
            )
        defaults = {}
        if gan_variant == "temporal_adaptive":
            defaults = {
                "lr_g": 3e-4,
                "lr_d": 5e-5,
                "d_steps": 1,
                "g_steps": 3,
                "lambda_adv": 0.5,
                "lambda_avoid": 0.3,
                "lambda_fm": 0.5,
                "lambda_var": 0.3,
                "lambda_div": 0.2,
                "lambda_stat": 0.2,
                "lambda_audio_dist": 0.1,
                "diversity_start_epoch": 3,
                "diversity_warmup_epochs": 5,
                "r1_gamma": 10.0,
                "r1_interval": 16,
                "instance_noise_std": 0.1,
                "instance_noise_decay_epochs": 50,
            }
        defaults.update({key: value for key, value in kwargs.items() if value is not None})
        return cls(gan_variant=gan_variant, **defaults)

