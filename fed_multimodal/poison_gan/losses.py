from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def discriminator_loss(logits_real, y_real, logits_fake, fake_class, lambda_d_fake=1.0):
    fake_label = torch.full_like(y_real, int(fake_class))
    loss_real = F.cross_entropy(logits_real, y_real)
    loss_fake = F.cross_entropy(logits_fake, fake_label)
    total = loss_real + lambda_d_fake * loss_fake
    metrics = {
        "d_loss_real": float(loss_real.detach().cpu()),
        "d_loss_fake": float(loss_fake.detach().cpu()),
        "d_loss": float(total.detach().cpu()),
    }
    return total, metrics


def feature_matching_loss(emb_fake, y_target, emb_real=None, y_real=None, bank=None):
    losses = []
    for cls in y_target.unique():
        cls_fake = emb_fake[y_target == cls]
        if cls_fake.numel() == 0:
            continue
        target_mean = None
        if emb_real is not None and y_real is not None and (y_real == cls).any():
            target_mean = emb_real[y_real == cls].detach().mean(dim=0)
        elif bank is not None:
            bank_mean, _, valid = bank.lookup(cls.view(1))
            if valid is not None and bool(valid[0].item()):
                target_mean = bank_mean[0].detach()
        if target_mean is not None:
            losses.append(F.mse_loss(cls_fake.mean(dim=0), target_mean))
    if not losses:
        return emb_fake.new_tensor(0.0)
    return torch.stack(losses).mean()


def variance_matching_loss(emb_fake, y_target, emb_real=None, y_real=None, bank=None):
    losses = []
    for cls in y_target.unique():
        cls_fake = emb_fake[y_target == cls]
        if cls_fake.size(0) < 2:
            continue
        target_var = None
        if emb_real is not None and y_real is not None and (y_real == cls).sum() > 1:
            target_var = emb_real[y_real == cls].detach().var(dim=0, unbiased=False)
        elif bank is not None:
            _, bank_var, valid = bank.lookup(cls.view(1))
            if valid is not None and bool(valid[0].item()):
                target_var = bank_var[0].detach()
        if target_var is not None:
            losses.append(F.l1_loss(cls_fake.var(dim=0, unbiased=False), target_var))
    if not losses:
        return emb_fake.new_tensor(0.0)
    return torch.stack(losses).mean()


def mode_seeking_diversity_loss(generator, z, y_target, len_a=None, len_v=None, eps=1e-6):
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z2 = torch.randn_like(z)
    fake_a1, fake_v1 = generator(z, y_target, len_a, len_v)
    fake_a2, fake_v2 = generator(z2, y_target, len_a, len_v)
    dist_a = (fake_a1 - fake_a2).flatten(1).norm(p=1, dim=1) / fake_a1[0].numel()
    dist_v = (fake_v1 - fake_v2).flatten(1).norm(p=1, dim=1) / fake_v1[0].numel()
    dist_z = (z - z2).flatten(1).norm(p=1, dim=1) / z.size(1)
    return -((dist_a + dist_v) / (dist_z + eps)).mean()


def stat_matching_loss(fake_audio, fake_video, real_audio, real_video, len_a=None, len_v=None):
    def masked_values(x, lengths):
        if lengths is None:
            return x.reshape(-1, x.size(-1))
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
        return x[mask]

    fa = masked_values(fake_audio, len_a)
    ra = masked_values(real_audio, len_a)
    fv = masked_values(fake_video, len_v)
    rv = masked_values(real_video, len_v)
    loss = fake_audio.new_tensor(0.0)
    if fa.numel() > 0 and ra.numel() > 0:
        loss = loss + F.l1_loss(fa.mean(dim=0), ra.detach().mean(dim=0))
        loss = loss + F.l1_loss(fa.std(dim=0, unbiased=False), ra.detach().std(dim=0, unbiased=False))
    if fv.numel() > 0 and rv.numel() > 0:
        loss = loss + F.l1_loss(fv.mean(dim=0), rv.detach().mean(dim=0))
        loss = loss + F.l1_loss(fv.std(dim=0, unbiased=False), rv.detach().std(dim=0, unbiased=False))
    return loss


def generator_loss(
    logits_fake,
    emb_fake,
    y_target,
    fake_class,
    config,
    emb_real=None,
    y_real=None,
    bank=None,
    generator=None,
    z=None,
    len_a=None,
    len_v=None,
    fake_audio=None,
    fake_video=None,
    real_audio=None,
    real_video=None,
    epoch=0,
):
    loss_target = F.cross_entropy(logits_fake, y_target)
    prob = F.softmax(logits_fake, dim=1)
    loss_avoid = prob[:, int(fake_class)].mean()
    loss_fm = feature_matching_loss(emb_fake, y_target, emb_real, y_real, bank)
    loss_var = variance_matching_loss(emb_fake, y_target, emb_real, y_real, bank)
    loss_div = logits_fake.new_tensor(0.0)
    if generator is not None and z is not None and epoch >= config.diversity_start_epoch:
        loss_div = mode_seeking_diversity_loss(generator, z, y_target, len_a, len_v)
    loss_stat = logits_fake.new_tensor(0.0)
    if fake_audio is not None and fake_video is not None and real_audio is not None and real_video is not None:
        loss_stat = stat_matching_loss(fake_audio, fake_video, real_audio, real_video, len_a, len_v)

    loss_audio_dist = logits_fake.new_tensor(0.0)
    if fake_audio is not None and real_audio is not None and config.lambda_audio_dist > 0:
        loss_audio_dist = audio_distribution_loss(
            fake_audio,
            real_audio,
            len_a,
            kurtosis_weight=config.audio_kurtosis_weight,
        )

    diversity_ramp = 0.0
    if epoch >= config.diversity_start_epoch:
        diversity_ramp = min(
            1.0,
            float(epoch - config.diversity_start_epoch + 1)
            / float(config.diversity_warmup_epochs),
        )
    var_weight = config.lambda_var * diversity_ramp
    div_weight = config.lambda_div * diversity_ramp
    total = (
        config.lambda_adv * loss_target
        + config.lambda_avoid * loss_avoid
        + config.lambda_fm * loss_fm
        + var_weight * loss_var
        + div_weight * loss_div
        + config.lambda_stat * loss_stat
        + config.lambda_audio_dist * loss_audio_dist
    )
    metrics = {
        "g_loss": float(total.detach().cpu()),
        "g_target": float(loss_target.detach().cpu()),
        "g_avoid": float(loss_avoid.detach().cpu()),
        "g_fm": float(loss_fm.detach().cpu()),
        "g_var": float(loss_var.detach().cpu()),
        "g_div": float(loss_div.detach().cpu()),
        "g_stat": float(loss_stat.detach().cpu()),
        "g_audio_dist": float(loss_audio_dist.detach().cpu()),
        "g_diversity_ramp": float(diversity_ramp),
        "fake_class_prob": float(prob[:, int(fake_class)].mean().detach().cpu()),
        "target_prob": float(prob.gather(1, y_target.view(-1, 1)).mean().detach().cpu()),

    }
    return total, metrics

def audio_distribution_loss(
    fake_audio,
    real_audio,
    lengths=None,
    kurtosis_weight=0.1,
    eps=1e-5,
):
    """Match real audio moments without normalizing each generated sample."""
    def values(x):
        if lengths is None:
            return x.reshape(-1, x.size(-1))
        mask = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0)
            < lengths.to(x.device).unsqueeze(1)
        )
        return x[mask]

    fake = values(fake_audio)
    real = values(real_audio).detach()
    if fake.numel() == 0 or real.numel() == 0:
        return fake_audio.new_tensor(0.0)

    fake_mean = fake.mean(dim=0)
    real_mean = real.mean(dim=0)
    fake_std = fake.std(dim=0, unbiased=False).clamp_min(eps)
    real_std = real.std(dim=0, unbiased=False).clamp_min(eps)
    loss = F.l1_loss(fake_mean, real_mean) + F.l1_loss(fake_std, real_std)

    if kurtosis_weight > 0 and fake.size(0) > 3 and real.size(0) > 3:
        fake_kurt = (((fake - fake_mean) / fake_std) ** 4).mean(dim=0) - 3.0
        real_kurt = (((real - real_mean) / real_std) ** 4).mean(dim=0) - 3.0
        loss = loss + kurtosis_weight * F.l1_loss(
            fake_kurt.clamp(-10.0, 10.0),
            real_kurt.clamp(-10.0, 10.0),
        )
    return loss


def r1_gradient_penalty(
    logits_real,
    y_real,
    real_inputs,
    lengths=None,
    gamma=10.0,
):
    """Multimodal R1 penalty on the real-class score, normalized per element."""
    scores = logits_real.gather(1, y_real.view(-1, 1)).sum()
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=tuple(real_inputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )
    penalties = []
    lengths = lengths or [None] * len(gradients)
    for gradient, sequence_lengths in zip(gradients, lengths):
        if gradient is None:
            continue
        if sequence_lengths is None:
            penalties.append(gradient.flatten(1).pow(2).mean(dim=1))
            continue
        mask = (
            torch.arange(gradient.size(1), device=gradient.device).unsqueeze(0)
            < sequence_lengths.to(gradient.device).unsqueeze(1)
        )
        while mask.dim() < gradient.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(gradient)
        numerator = (gradient * mask).flatten(1).pow(2).sum(dim=1)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1)
        penalties.append(numerator / denominator)
    if not penalties:
        return logits_real.new_tensor(0.0)
    return 0.5 * float(gamma) * torch.stack(penalties, dim=0).sum(dim=0).mean()
