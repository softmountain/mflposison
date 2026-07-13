from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class PoisonGANConfig:
    num_classes: int
    fake_class: int
    audio_seq_len: int = 500
    audio_feat_dim: int = 80
    video_seq_len: int = 9
    video_feat_dim: int = 1280
    z_dim: int = 256
    label_emb_dim: int = 128
    hidden_dim: int = 512
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    d_steps: int = 1
    g_steps: int = 2
    lambda_d_fake: float = 1.0
    lambda_adv: float = 1.0
    lambda_avoid: float = 0.2
    lambda_fm: float = 0.2
    lambda_var: float = 0.1
    lambda_div: float = 0.05
    lambda_stat: float = 0.05
    lambda_mod: float = 0.0
    audio_out_max: float = 3.0
    video_out_max: float = 20.0
    video_scale_max: float = 8.0
    warmup_epochs: int = 5
    diversity_start_epoch: int = 10
    target_strategy: str = "same_as_real"
    fixed_target: int = -1
    mixed_target_prob: float = 0.7
    freeze_d: str = "none"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoisonGANConfig":
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)
