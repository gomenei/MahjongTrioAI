"""Shared supervised-learning loss, seed and device helpers."""
import random
import numpy as np
import torch
import torch.nn.functional as F


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_cross_entropy(logits, targets, action_mask, label_smoothing):
    """仅在合法动作之间做标签平滑，避免给非法动作分配概率。"""
    if label_smoothing <= 0:
        return F.cross_entropy(logits, targets)
    legal = action_mask > 0
    log_probabilities = F.log_softmax(logits, dim=1)
    negative_log_likelihood = F.nll_loss(log_probabilities, targets, reduction="none")
    legal_log_probabilities = log_probabilities.masked_fill(~legal, 0.0)
    legal_counts = legal.sum(dim=1).clamp_min(1)
    smooth_loss = -legal_log_probabilities.sum(dim=1) / legal_counts
    return (
        (1.0 - label_smoothing) * negative_log_likelihood
        + label_smoothing * smooth_loss
    ).mean()
