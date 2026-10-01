import copy

import pytest
import torch

from model import TileTransformer
from ppo_train import ActorCritic
from training.cuda_rollout import CudaRolloutCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph verification requires a GPU")
def test_cuda_rollout_preserves_logits_values_masks_and_weight_updates():
    torch.manual_seed(42)
    model = ActorCritic(TileTransformer(obs_channels=121), obs_channels=121).cuda().eval()
    cache = CudaRolloutCache(enabled=True)
    for size in (5, 7, 3, 7):
        observation = torch.rand(size, 121, 30, device="cuda")
        mask = torch.zeros(size, 177, device="cuda")
        mask[:, [1, 5, 9, 174]] = 1
        inputs = {"observation": observation, "action_mask": mask}
        with torch.inference_mode():
            expected = model(inputs)
            actual = cache(model, inputs)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
            assert torch.equal(expected[0].argmax(-1), actual[0].argmax(-1))
            # Graph replay must read live parameter storage after an optimizer update.
            next(model.actor.parameters()).add_(0.002)
            expected = model(inputs)
            actual = cache(model, inputs)
            torch.testing.assert_close(expected[0], actual[0], rtol=2e-5, atol=2e-5)
    # Graph bookkeeping must not make policy snapshots uncopyable.
    copy.deepcopy(model.actor)
