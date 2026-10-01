from dataclasses import asdict

import numpy as np

from mahjong_env.match import SouthMatch
from training.parallel_env import forced_actions, step_to_decision


def observation(*legal):
    mask = np.zeros(177, dtype=np.float32)
    mask[list(legal)] = 1
    return {"player_0": {"action_mask": mask}}


def test_forced_actions_never_skip_a_real_choice_or_guess_an_empty_mask():
    assert forced_actions(observation(174)) == {"player_0": 174}
    assert forced_actions(observation(176)) == {"player_0": 176}
    assert forced_actions(observation(0, 1)) is None
    assert forced_actions(observation()) is None
    assert forced_actions({}) is None


def test_forced_forward_counts_real_steps_and_stops_at_decisions_and_limits():
    class Game:
        def __init__(self):
            self.actions = []

        def step(self, actions):
            self.actions.append(actions)
            result = observation(174) if len(self.actions) == 1 else observation(0, 1)
            return result, {}, False

    game = Game()
    obs, done, steps, error = step_to_decision(game, {"player_0": 176}, 0, 5, True)
    assert game.actions == [{"player_0": 176}, {"player_0": 174}]
    assert not done and steps == 2 and error is None
    assert forced_actions(obs) is None
    game = Game()
    _, done, steps, error = step_to_decision(game, {"player_0": 176}, 4, 5, True)
    assert done and steps == 5 and error is not None
    assert len(game.actions) == 1


def play(seed, fast):
    match = SouthMatch()
    observations = match.reset(seed=seed)
    steps = 0
    while True:
        actions = {}
        for name, obs in observations.items():
            legal = np.flatnonzero(obs["action_mask"])
            actions[name] = 175 if 175 in legal else int(legal[0])
        observations, done, steps, error = step_to_decision(match, actions, steps, 10000, fast)
        assert error is None
        if done:
            return match.scores, match.final_ranks, steps, [asdict(row) for row in match.history]


def test_forced_forward_preserves_complete_south_match_results():
    for seed in (5101, 5102, 5103):
        assert play(seed, False) == play(seed, True)
