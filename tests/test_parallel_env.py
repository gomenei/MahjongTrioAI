import os
import unittest

import numpy as np

from mahjong_env.game import ThreePlayerMahjong
from training.parallel_env import (
    GameSpec,
    ParallelGamePool,
    recommended_env_workers,
)


class ParallelEnvironmentTest(unittest.TestCase):
    def test_i7_12700h_worker_recommendation(self):
        self.assertEqual(recommended_env_workers(logical_processors=20), 12)

    @unittest.skipIf(
        os.environ.get("CI_SKIP_MULTIPROCESS") == "1",
        "当前 CI 禁用多进程测试",
    )
    def test_shared_observation_pool_completes_games(self):
        specs = [GameSpec(slot=index, seed=7000 + index) for index in range(4)]
        with ParallelGamePool(capacity=4, workers=2, max_steps=300) as pool:
            started = pool.start(specs)
            self.assertEqual(len(started), 4)
            completed = [result for result in started if result["done"]]
            while pool.active_slots:
                actions = {}
                for slot in sorted(pool.active_slots):
                    game_actions = {}
                    observations = pool.observations(slot)
                    for agent_name, observation in observations.items():
                        legal = np.flatnonzero(observation["action_mask"] > 0)
                        game_actions[agent_name] = int(legal[0])
                    actions[slot] = game_actions
                results = pool.step(actions)
                completed.extend(
                    result for result in results if result["done"]
                )
            self.assertEqual(len(completed), 4)
            for result in completed:
                self.assertTrue(result["done"])
                self.assertIsInstance(result["game"], ThreePlayerMahjong)
                self.assertIsNone(result["error"])

    def test_async_pool_preserves_observations_steps_and_results_across_batches(self):
        def play(async_min_workers):
            import hashlib

            rounds = []
            with ParallelGamePool(capacity=7, workers=3, max_steps=3000,
                                  fast_forward_forced=True,
                                  async_min_workers=async_min_workers) as pool:
                for batch in range(2):
                    hashes = {slot: hashlib.sha256() for slot in range(7)}
                    completed = {}
                    results = pool.start([GameSpec(slot, 7401 + slot + 7 * batch,
                                                  game_mode="south") for slot in range(7)])
                    while pool.active_slots:
                        actions = {}
                        for result in results:
                            slot = result["slot"]
                            if result["done"]:
                                assert result.get("error") is None
                                game = result["game"]
                                completed[slot] = (result["steps"], game.scores, game.final_ranks)
                                continue
                            actions[slot] = {}
                            for name, obs in pool.observations(slot).items():
                                hashes[slot].update(obs["match_observation"].tobytes())
                                hashes[slot].update(obs["action_mask"].tobytes())
                                legal = np.flatnonzero(obs["action_mask"])
                                actions[slot][name] = 175 if 175 in legal else int(legal[0])
                        results = pool.step(actions)
                    for result in results:
                        assert result["done"] and result.get("error") is None
                        game = result["game"]
                        completed[result["slot"]] = (result["steps"], game.scores, game.final_ranks)
                    assert len(completed) == 7
                    assert not pool.pending_workers and not pool.ready_slots
                    rounds.append((completed, {slot: digest.hexdigest() for slot, digest in hashes.items()}))
            return rounds

        self.assertEqual(play(0), play(1))


if __name__ == "__main__":
    unittest.main()
