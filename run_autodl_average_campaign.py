"""Predeclared uniform averaging experiment; fresh holdouts, no checkpoint search."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from run_autodl_training import (
    BASE, ROOT, TRAIN_SOURCES, battle, candidates, read_json, sha256, write_json,
)


RECIPE = "average_v8_fix1"
SOURCE_RUNS = (
    Path("/root/autodl-tmp/MahjongTrioAI-rank-v6/ppo_runs/rank_v6_130260916"),
    Path("/root/autodl-tmp/MahjongTrioAI-rank-v7/ppo_runs/rank_v7_150260916"),
)
SMOKE_SEED = 2_631_000_000
CONFIRM_SEEDS = (2_741_000_000, 2_851_000_000)


def now():
    return datetime.now(timezone.utc).isoformat()


def average_checkpoints(sources, destination, *, recipe=RECIPE):
    import torch
    from model import create_model, load_checkpoint_model

    payloads = [torch.load(p, map_location="cpu", weights_only=True) for p in sources]
    architecture = (payloads[0]["model_name"], int(payloads[0]["obs_channels"]))
    observation_key = payloads[0]["observation_key"]
    states = [p["model_state_dict"] for p in payloads]
    for payload, state in zip(payloads, states):
        if (payload["model_name"], int(payload["obs_channels"])) != architecture:
            raise ValueError("Source architectures differ")
        if payload["observation_key"] != observation_key:
            raise ValueError("Source observation features differ")
        if state.keys() != states[0].keys():
            raise ValueError("Source state keys differ")
    model = create_model(architecture[0], obs_channels=architecture[1]).eval()
    # BatchNorm running statistics need a separate recalibration protocol.
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("BatchNorm averaging is not part of this experiment")
    average = {}
    for name, first in states[0].items():
        values = [state[name] for state in states]
        if any(v.shape != first.shape or v.dtype != first.dtype for v in values):
            raise ValueError(f"Incompatible tensor: {name}")
        if first.is_floating_point():
            if not all(torch.isfinite(v).all() for v in values):
                raise ValueError(f"Nonfinite source tensor: {name}")
            average[name] = torch.stack([v.double() for v in values]).mean(0).to(first.dtype)
        else:
            if not all(torch.equal(v, first) for v in values):
                raise ValueError(f"Nonfloating buffer differs: {name}")
            average[name] = first.clone()
    model.load_state_dict(average, strict=True)
    output = {"model_name": architecture[0], "obs_channels": architecture[1],
              "observation_key": observation_key,
              "model_state_dict": average, "averaging_method": "uniform_fp64_accumulation",
              "averaging_sources": [str(p) for p in sources], "recipe": recipe}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    reloaded, metadata = load_checkpoint_model(destination)
    if metadata["observation_key"] != observation_key:
        raise ValueError("Serialization lost observation feature selection")
    reloaded.eval()
    for name, tensor in reloaded.state_dict().items():
        if not torch.equal(tensor, average[name]):
            raise ValueError(f"Reloaded average differs: {name}")
    generator = torch.Generator().manual_seed(361_000_000)
    observation = torch.rand(8, architecture[1], 30, generator=generator)
    mask = torch.zeros(8, 177)
    mask[:, ::3] = 1
    inputs = {"observation": observation, "action_mask": mask}
    with torch.inference_mode():
        expected, actual = model(inputs), reloaded(inputs)
    if not torch.equal(expected, actual) or not torch.isfinite(actual).all():
        raise ValueError("Serialization changed inference or produced nonfinite logits")
    if not mask.gather(1, actual.argmax(-1, keepdim=True)).all():
        raise ValueError("Averaged policy selected an illegal action")
    return {"sources": len(sources), "architecture": list(architecture),
            "observation_key": observation_key,
            "parameters": sum(p.numel() for p in model.parameters()),
            "roundtrip_exact": True, "masked_inference_finite_and_legal": True,
            "contains_batchnorm": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    import fcntl
    torch.set_num_threads(4)
    output = ROOT / "campaigns" / RECIPE
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = []
    for run_dir in SOURCE_RUNS:
        run_sources = candidates(run_dir)
        if len(run_sources) != 4:
            raise ValueError("Expected exactly four distinct training-time checkpoints per seed")
        sources.extend(run_sources)
    identity = {
        "recipe": RECIPE, "method": "uniform average of all four times from both runs",
        "inputs": [{"path": str(p), "sha256": sha256(p), "weight": 1 / 8} for p in sources],
        "base_sha256": sha256(ROOT / BASE),
        "source_sha256": {name: sha256(ROOT / name) for name in
                          (*TRAIN_SOURCES, "run_autodl_training.py", Path(__file__).name)},
        "smoke_seed": SMOKE_SEED, "smoke_groups_per_lineup": 64,
        "smoke_gate": "validity only; no strength-based candidate selection",
        "confirm_seeds": list(CONFIRM_SEEDS), "confirm_groups_per_lineup": 1000,
        "promotion_gate": "both holdouts must pass unchanged assess_report criteria",
        "candidate_count": 1, "env_workers": args.env_workers, "device": args.device,
        "no_additional_training": True, "keep_server_running": True,
        "method_reference": "https://proceedings.mlr.press/v162/wortsman22a.html",
    }
    manifest = output / "experiment.json"
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("Experiment identity changed; use a fresh experiment")
    if not manifest.exists():
        write_json(manifest, {"identity": identity, "declared_at": now()})
    state_path = output / "status.json"
    state = read_json(state_path) if state_path.exists() else {"completed_stages": []}
    if state.get("status") == "completed":
        return

    def status(stage):
        state.update(status="running", stage=stage, pid=os.getpid(), updated_at=now())
        write_json(state_path, state)

    eval_dir = Path("battle_results") / RECIPE
    candidate = Path("ppo_runs") / RECIPE / "uniform_eight_frozen.pt"
    selection_path = ROOT / eval_dir / "selection.json"
    try:
        if not selection_path.exists():
            status("average")
            validation = average_checkpoints(sources, ROOT / candidate)
            write_json(output / "validation.json", validation)
            write_json(selection_path, {"selected": {"checkpoint": candidate.as_posix(),
                       "sha256": sha256(ROOT / candidate)}, "frozen_at": now(),
                       "base_sha256": identity["base_sha256"],
                       "confirm_seeds": list(CONFIRM_SEEDS), "selection": "single predeclared candidate"})
        selection = read_json(selection_path)

        def verify_frozen():
            if sha256(ROOT / candidate) != selection["selected"]["sha256"]:
                raise ValueError("Frozen candidate changed")
            if sha256(ROOT / BASE) != identity["base_sha256"]:
                raise ValueError("Frozen baseline changed")
            for item in identity["inputs"]:
                if sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError("An averaging input changed")

        verify_frozen()
        if "smoke" not in state["completed_stages"]:
            status("smoke")
            smoke = battle(args, candidate, eval_dir / "smoke", SMOKE_SEED, 64)
            write_json(output / "smoke.json", smoke)
            if not smoke["valid"]:
                raise ValueError("Smoke validity gate failed")
            state["completed_stages"].append("smoke")
            status("smoke")
        rows = []
        for seed in CONFIRM_SEEDS:
            verify_frozen()
            status(f"confirm:{seed}")
            rows.append(battle(args, candidate, eval_dir / "confirm" / str(seed), seed, 1000))
        verify_frozen()
        proven = len(rows) == 2 and all(row["pass"] for row in rows)
        result = {"status": "pass" if proven else "not_proven", "candidate": selection["selected"],
                  "results": rows, "strength_improvement_proven": proven,
                  "automatically_installed": False, "finished_at": now()}
        write_json(ROOT / eval_dir / "promotion.json", result)
        write_json(output / "summary.json", result)
        state.update(status="completed", stage=None, updated_at=now(),
                     strength_improvement_proven=proven)
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
