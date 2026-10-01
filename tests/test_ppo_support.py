import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from training.models import apply_action_mask
from training.ppo import ActorCritic, Transition, build_ppo_batch, ppo_update
from training.ppo_support import OpponentPool, masked_reference_kl


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.5, -0.5, 1.0]))

    def forward(self, inputs):
        return apply_action_mask(
            self.logits.expand(len(inputs["observation"]), -1), inputs["action_mask"]
        )


class PPOSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pool_freezes_evicts_and_restores_exact_policies(self):
        actor = TinyActor()
        pool = OpponentPool(actor, capacity=2)
        initial = actor.logits.detach().clone()
        with torch.no_grad():
            actor.logits.add_(1)
        torch.testing.assert_close(pool.models[0].logits, initial)
        pool.add(actor)
        with torch.no_grad():
            actor.logits.add_(1)
        pool.add(actor)
        torch.testing.assert_close(pool.models[0].logits, initial + 1)
        restored = OpponentPool(actor, 2)
        restored.restore(copy.deepcopy(pool.checkpoint_states()))
        for left, right in zip(pool.models, restored.models):
            torch.testing.assert_close(left.logits, right.logits)
            self.assertFalse(right.training)
            self.assertFalse(right.logits.requires_grad)
        rng = np.random.default_rng(42)
        self.assertEqual({pool.sample(rng) for _ in range(100)}, set(pool.policies()))

    def test_reference_kl_ignores_illegal_actions_and_has_finite_gradients(self):
        logits = torch.tensor([[1.0, 2.0, -torch.inf]], requires_grad=True)
        reference = torch.tensor([[2.0, 1.0, -torch.inf]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0]])
        for temperature in (0.1, 1.0, 2.0):
            loss = masked_reference_kl(logits, reference, mask, temperature)
            expected = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=reference[:, :2] / temperature),
                torch.distributions.Categorical(logits=logits[:, :2] / temperature),
            ).mean()
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(float(logits.grad[0, 2]), 0.0)
            self.assertIsNone(reference.grad)
            logits.grad = None
        identical = masked_reference_kl(logits, logits, mask)
        self.assertAlmostEqual(float(identical), 0.0)

    def test_long_horizon_reward_reaches_early_decisions(self):
        trajectory = [Transition(np.zeros((6, 30)), np.ones(3), 0, 0, 0)
                      for _ in range(101)]
        completed = [{"episode": SimpleNamespace(trajectory=trajectory), "reward": 1.0}]
        batch = build_ppo_batch(completed, SimpleNamespace(gamma=1.0, gae_lambda=0.99))
        self.assertAlmostEqual(float(batch["returns"][0]), 0.99 ** 100, places=6)
        self.assertAlmostEqual(float(batch["returns"][-1]), 1.0)
        self.assertTrue(np.isfinite(batch["advantages"]).all())

    def test_warmup_keeps_actor_fixed_then_policy_updates_without_reference_gradients(self):
        torch.manual_seed(42)
        actor_critic = ActorCritic(TinyActor())
        reference = copy.deepcopy(actor_critic.actor).requires_grad_(False)
        optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=1e-3)
        observations = np.zeros((8, 6, 30), dtype=np.float32)
        masks = np.ones((8, 3), dtype=np.float32)
        inputs = {"observation": torch.from_numpy(observations),
                  "action_mask": torch.from_numpy(masks)}
        with torch.no_grad():
            logits, values = actor_critic(inputs)
            old_probs = logits.log_softmax(-1)[:, 0].numpy()
        batch = dict(observation=observations, action_mask=masks,
                     actions=np.zeros(8, dtype=np.int64),
                     old_log_probabilities=old_probs, old_values=values.numpy(),
                     returns=np.ones(8, dtype=np.float32),
                     advantages=np.ones(8, dtype=np.float32))
        args = SimpleNamespace(ppo_epochs=1, minibatch_size=8, policy_temperature=1.0,
                               clip_ratio=0.15, value_clip=0.2, value_coef=0.5,
                               entropy_coef=0.003, max_grad_norm=0.5, target_kl=0.02,
                               reference_kl_coef=0.05, critic_warmup_updates=1)
        before_actor = actor_critic.actor.logits.detach().clone()
        before_critic = next(actor_critic.critic.parameters()).detach().clone()
        losses = ppo_update(actor_critic, optimizer, batch, args, torch.device("cpu"),
                            reference=reference, update=1)
        torch.testing.assert_close(before_actor, actor_critic.actor.logits)
        self.assertFalse(torch.equal(before_critic, next(actor_critic.critic.parameters())))
        self.assertTrue(losses["critic_warmup"])
        ppo_update(actor_critic, optimizer, batch, args, torch.device("cpu"),
                   reference=reference, update=2)
        self.assertFalse(torch.equal(before_actor, actor_critic.actor.logits))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))


if __name__ == "__main__":
    unittest.main()
