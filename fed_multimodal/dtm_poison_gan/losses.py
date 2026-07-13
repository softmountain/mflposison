import torch
import torch.nn.functional as F


def discriminator_loss(
    logits_real,
    real_labels,
    logits_fake,
    fake_class,
    fake_weight=0.5,
):
    fake_labels = torch.full_like(real_labels, int(fake_class))
    real_loss = F.cross_entropy(logits_real, real_labels)
    fake_loss = F.cross_entropy(logits_fake, fake_labels)
    total = real_loss + float(fake_weight) * fake_loss
    return total, {
        "d_loss": float(total.detach().cpu()),
        "d_real": float(real_loss.detach().cpu()),
        "d_fake": float(fake_loss.detach().cpu()),
    }


def _squared_distances(x, y):
    return torch.cdist(x.float(), y.float(), p=2).pow(2)


def multi_scale_mmd(x, y, scales=(0.5, 1.0, 2.0, 4.0), eps=1e-6):
    """Biased multi-scale RBF MMD, stable for unequal and small batches."""
    combined = torch.cat([x.detach(), y.detach()], dim=0).float()
    distances = _squared_distances(combined, combined)
    positive = distances[distances > 0]
    bandwidth = (
        positive.median() if positive.numel() else distances.new_tensor(1.0)
    ).clamp_min(eps)

    xx = _squared_distances(x, x)
    yy = _squared_distances(y, y)
    xy = _squared_distances(x, y)
    mmd = x.new_tensor(0.0)
    for scale in scales:
        denominator = 2.0 * bandwidth * float(scale)
        mmd = mmd + (
            torch.exp(-xx / denominator).mean()
            + torch.exp(-yy / denominator).mean()
            - 2.0 * torch.exp(-xy / denominator).mean()
        )
    return (mmd / len(scales)).clamp_min(0.0)


def _prototype_for_class(bank, cls):
    if bank is None:
        return None, None
    mean, var, valid = bank.lookup(cls.view(1))
    if valid is None or not bool(valid[0].item()):
        return None, None
    return mean[0].detach(), var[0].detach()


def classwise_distribution_loss(
    fake_embeddings,
    target_labels,
    real_embeddings=None,
    real_labels=None,
    bank=None,
):
    """Use local classwise MMD, with server/local prototypes as missing-class fallback."""
    losses = []
    for cls in target_labels.unique():
        fake = fake_embeddings[target_labels == cls]
        if fake.numel() == 0:
            continue
        real = None
        if real_embeddings is not None and real_labels is not None:
            real_mask = real_labels == cls
            if real_mask.any():
                real = real_embeddings[real_mask].detach()
        if real is not None:
            losses.append(multi_scale_mmd(fake, real))
            continue
        target_mean, target_var = _prototype_for_class(bank, cls)
        if target_mean is None:
            continue
        loss = F.mse_loss(fake.mean(dim=0), target_mean)
        if fake.size(0) > 1:
            fake_std = fake.var(dim=0, unbiased=False).clamp_min(0).sqrt()
            target_std = target_var.clamp_min(0).sqrt()
            loss = loss + F.l1_loss(fake_std, target_std)
        losses.append(loss)
    if not losses:
        return fake_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def variance_floor_loss(
    fake_embeddings,
    target_labels,
    real_embeddings=None,
    real_labels=None,
    bank=None,
):
    """VICReg-style floor: fake class spread should not fall below real spread."""
    losses = []
    for cls in target_labels.unique():
        fake = fake_embeddings[target_labels == cls]
        if fake.size(0) < 2:
            continue
        target_var = None
        if real_embeddings is not None and real_labels is not None:
            real = real_embeddings[real_labels == cls]
            if real.size(0) > 1:
                target_var = real.detach().var(dim=0, unbiased=False)
        if target_var is None:
            _, target_var = _prototype_for_class(bank, cls)
        if target_var is None:
            continue
        fake_std = fake.var(dim=0, unbiased=False).clamp_min(0).sqrt()
        target_std = target_var.clamp_min(0).sqrt()
        losses.append(F.relu(target_std - fake_std).mean())
    if not losses:
        return fake_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def _masked_values(x, lengths=None):
    if lengths is None:
        return x.reshape(-1, x.size(-1))
    mask = (
        torch.arange(x.size(1), device=x.device).unsqueeze(0)
        < lengths.to(x.device).unsqueeze(1)
    )
    return x[mask]


def raw_stat_loss(
    fake_audio,
    fake_video,
    real_audio,
    real_video,
    len_audio=None,
    len_video=None,
):
    loss = fake_audio.new_tensor(0.0)
    pairs = (
        (_masked_values(fake_audio, len_audio), _masked_values(real_audio, len_audio)),
        (_masked_values(fake_video, len_video), _masked_values(real_video, len_video)),
    )
    for fake, real in pairs:
        if fake.numel() == 0 or real.numel() == 0:
            continue
        real = real.detach()
        loss = loss + F.l1_loss(fake.mean(dim=0), real.mean(dim=0))
        loss = loss + F.l1_loss(
            fake.std(dim=0, unbiased=False),
            real.std(dim=0, unbiased=False),
        )
    return loss


def audio_tail_loss(fake_audio, real_audio, lengths=None, eps=1e-5):
    """Match audio skew/kurtosis without clipping away the real long tail."""
    fake = _masked_values(fake_audio, lengths)
    real = _masked_values(real_audio, lengths).detach()
    if fake.size(0) < 4 or real.size(0) < 4:
        return fake_audio.new_tensor(0.0)

    def standardized_moments(values):
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(eps)
        normalized = (values - mean) / std
        return normalized.pow(3).mean(dim=0), normalized.pow(4).mean(dim=0) - 3.0

    fake_skew, fake_kurtosis = standardized_moments(fake)
    real_skew, real_kurtosis = standardized_moments(real)
    return F.l1_loss(
        fake_skew.clamp(-10.0, 10.0),
        real_skew.clamp(-10.0, 10.0),
    ) + F.l1_loss(
        fake_kurtosis.clamp(-10.0, 10.0),
        real_kurtosis.clamp(-10.0, 10.0),
    )


def mode_seeking_loss(
    generator,
    z,
    labels,
    len_audio=None,
    len_video=None,
    eps=1e-6,
):
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z_other = torch.randn_like(z)
    audio_a, video_a = generator(z, labels, len_audio, len_video)
    audio_b, video_b = generator(z_other, labels, len_audio, len_video)
    output_distance = (
        (audio_a - audio_b).flatten(1).abs().mean(dim=1)
        + (video_a - video_b).flatten(1).abs().mean(dim=1)
    )
    latent_distance = (z - z_other).abs().mean(dim=1)
    return -(output_distance / (latent_distance + eps)).mean()


def generator_loss(
    logits_fake,
    fake_embeddings,
    target_labels,
    config,
    real_embeddings=None,
    real_labels=None,
    bank=None,
    generator=None,
    z=None,
    fake_audio=None,
    fake_video=None,
    real_audio=None,
    real_video=None,
    len_audio=None,
    len_video=None,
    epoch=0,
):
    target_loss = F.cross_entropy(logits_fake, target_labels)
    probabilities = F.softmax(logits_fake, dim=1)
    avoid_loss = probabilities[:, int(config.fake_class)].mean()
    distribution_loss = classwise_distribution_loss(
        fake_embeddings,
        target_labels,
        real_embeddings,
        real_labels,
        bank,
    )
    floor_loss = variance_floor_loss(
        fake_embeddings,
        target_labels,
        real_embeddings,
        real_labels,
        bank,
    )
    stat_loss = raw_stat_loss(
        fake_audio,
        fake_video,
        real_audio,
        real_video,
        len_audio,
        len_video,
    )
    tail_loss = audio_tail_loss(fake_audio, real_audio, len_audio)

    diversity_ramp = 0.0
    diversity_loss = logits_fake.new_tensor(0.0)
    if epoch >= config.diversity_start_epoch:
        diversity_ramp = min(
            1.0,
            float(epoch - config.diversity_start_epoch + 1)
            / float(config.diversity_warmup_epochs),
        )
        diversity_loss = mode_seeking_loss(
            generator,
            z,
            target_labels,
            len_audio,
            len_video,
        )

    total = (
        config.lambda_adv * target_loss
        + config.lambda_avoid * avoid_loss
        + config.lambda_distribution * distribution_loss
        + config.lambda_var_floor * diversity_ramp * floor_loss
        + config.lambda_raw_stat * stat_loss
        + config.lambda_audio_tail * tail_loss
        + config.lambda_diversity * diversity_ramp * diversity_loss
    )
    return total, {
        "g_loss": float(total.detach().cpu()),
        "g_target": float(target_loss.detach().cpu()),
        "g_avoid": float(avoid_loss.detach().cpu()),
        "g_distribution": float(distribution_loss.detach().cpu()),
        "g_var_floor": float(floor_loss.detach().cpu()),
        "g_raw_stat": float(stat_loss.detach().cpu()),
        "g_audio_tail": float(tail_loss.detach().cpu()),
        "g_diversity": float(diversity_loss.detach().cpu()),
        "g_diversity_ramp": float(diversity_ramp),
        "fake_class_prob": float(avoid_loss.detach().cpu()),
    }
