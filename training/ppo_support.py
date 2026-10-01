"""Frozen opponents and numerically stable policy retention for PPO."""

from __future__ import annotations

import copy

import torch
from torch import nn


class OpponentPool:
    """FIFO history; opponents stay fixed throughout each collected match."""

    def __init__(self, actor: nn.Module, capacity: int = 1):
        if capacity < 1:
            raise ValueError("Opponent pool capacity must be positive")
        self.capacity = capacity
        self.models: list[nn.Module] = []
        self.add(actor)

    def add(self, actor: nn.Module) -> None:
        frozen = copy.deepcopy(actor).eval().requires_grad_(False)
        self.models.append(frozen)
        if len(self.models) > self.capacity:
            self.models.pop(0)

    def policies(self) -> dict[str, nn.Module]:
        return {f"snapshot_{i}": model for i, model in enumerate(self.models)}

    def sample(self, rng) -> str:
        # Preserve the old RNG sequence with a single historical policy.
        index = int(rng.integers(len(self.models))) if len(self.models) > 1 else 0
        return f"snapshot_{index}"

    def state_dict(self):
        return self.models[-1].state_dict()

    def checkpoint_states(self):
        return [model.state_dict() for model in self.models]

    def restore(self, states) -> None:
        if not states or len(states) > self.capacity:
            raise ValueError("Checkpoint opponent pool does not fit --opponent-pool-size")
        template = self.models[-1]
        restored = []
        for state in states:
            model = copy.deepcopy(template).eval().requires_grad_(False)
            model.load_state_dict(state)
            restored.append(model)
        self.models = restored


def masked_reference_kl(logits, reference_logits, action_mask, temperature=1.0):
    """KL(reference || actor), excluding illegal actions even at low temperature.

    Compute in float32; zero both log probabilities on illegal actions *before*
    subtracting them, so masked -inf values cannot produce 0 * NaN gradients.
    """
    legal = action_mask > 0
    log_probs = (logits.float() / temperature).masked_fill(~legal, -torch.inf)
    reference = (reference_logits.detach().float() / temperature).masked_fill(
        ~legal, -torch.inf
    )
    log_probs = log_probs.log_softmax(-1).masked_fill(~legal, 0.0)
    ref_log_probs = reference.log_softmax(-1).masked_fill(~legal, 0.0)
    ref_probs = ref_log_probs.exp().masked_fill(~legal, 0.0)
    return (ref_probs * (ref_log_probs - log_probs)).sum(-1).mean()
