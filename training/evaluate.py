import time
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F


ACTION_GROUPS = (
    ("出牌", 0, 29),
    ("碰", 29, 58),
    ("明杠", 58, 87),
    ("暗杠", 87, 116),
    ("加杠", 116, 145),
    ("立直", 145, 174),
    ("拔北", 174, 175),
    ("和牌", 175, 176),
    ("跳过", 176, 177),
)


def action_group_ids(actions: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(actions)
    for group_id, (_, start, end) in enumerate(ACTION_GROUPS):
        result[(actions >= start) & (actions < end)] = group_id
    return result


@torch.inference_mode()
def evaluate_model(
    model,
    loader,
    device,
    max_batches: Optional[int] = None,
) -> Dict:
    model.eval()
    totals = defaultdict(float)
    group_total = torch.zeros(len(ACTION_GROUPS), dtype=torch.long)
    group_correct = torch.zeros(len(ACTION_GROUPS), dtype=torch.long)
    started = time.perf_counter()

    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        batch_size = targets.shape[0]
        sample_losses = F.cross_entropy(logits, targets, reduction="none")
        totals["loss"] += sample_losses.sum().item()
        predictions = logits.argmax(dim=1)
        totals["correct"] += (predictions == targets).sum().item()
        decision_mask = (inputs["action_mask"] > 0).sum(dim=1) > 1
        totals["decision_samples"] += decision_mask.sum().item()
        totals["decision_correct"] += (
            (predictions == targets) & decision_mask
        ).sum().item()
        totals["decision_loss"] += sample_losses[decision_mask].sum().item()
        totals["top3"] += (
            logits.topk(min(3, logits.shape[1]), dim=1).indices == targets[:, None]
        ).any(dim=1).sum().item()
        totals["top5"] += (
            logits.topk(min(5, logits.shape[1]), dim=1).indices == targets[:, None]
        ).any(dim=1).sum().item()
        predicted_groups = action_group_ids(predictions)
        target_groups = action_group_ids(targets)
        totals["type_correct"] += (predicted_groups == target_groups).sum().item()
        totals["samples"] += batch_size
        for group_id in range(len(ACTION_GROUPS)):
            selected = target_groups == group_id
            group_total[group_id] += selected.sum().cpu()
            group_correct[group_id] += ((predictions == targets) & selected).sum().cpu()

    elapsed = max(time.perf_counter() - started, 1e-9)
    samples = int(totals["samples"])
    decision_samples = int(totals["decision_samples"])
    per_group = {}
    available_accuracies = []
    for group_id, (name, _, _) in enumerate(ACTION_GROUPS):
        count = int(group_total[group_id])
        accuracy = float(group_correct[group_id] / count) if count else None
        per_group[name] = {"samples": count, "accuracy": accuracy}
        if accuracy is not None:
            available_accuracies.append(accuracy)

    return {
        "samples": samples,
        "loss": totals["loss"] / max(samples, 1),
        "accuracy": totals["correct"] / max(samples, 1),
        "decision_samples": decision_samples,
        "forced_samples": samples - decision_samples,
        "decision_loss": totals["decision_loss"] / max(decision_samples, 1),
        "decision_accuracy": totals["decision_correct"] / max(decision_samples, 1),
        "top3_accuracy": totals["top3"] / max(samples, 1),
        "top5_accuracy": totals["top5"] / max(samples, 1),
        "action_type_accuracy": totals["type_correct"] / max(samples, 1),
        "macro_action_accuracy": sum(available_accuracies) / max(len(available_accuracies), 1),
        "samples_per_second": samples / elapsed,
        "per_action_group": per_group,
    }
