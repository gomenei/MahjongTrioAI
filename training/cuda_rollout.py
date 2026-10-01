"""CUDA graph replay for small, repeated inference batches during rollout only."""

import weakref

import torch


class CudaRolloutCache:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.models = weakref.WeakKeyDictionary()

    @torch.inference_mode()
    def __call__(self, model, inputs):
        observation = inputs["observation"]
        if not self.enabled or observation.device.type != "cuda":
            return model(inputs)
        if model.training:
            raise ValueError("CUDA rollout graphs require eval mode")
        size = observation.shape[0]
        bucket = 1 << (size - 1).bit_length()
        key = (bucket, tuple(observation.shape[1:]), observation.dtype, observation.device)
        cache = self.models.setdefault(model, {})
        if key not in cache:
            static = {name: value.new_zeros((bucket, *value.shape[1:]))
                      for name, value in inputs.items()}
            static["action_mask"][:, 0] = 1
            for name, value in inputs.items():
                static[name][:size].copy_(value)
            stream = torch.cuda.Stream(device=observation.device)
            stream.wait_stream(torch.cuda.current_stream(observation.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    model(static)
            torch.cuda.current_stream(observation.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = model(static)
            torch.cuda.current_stream(observation.device).wait_stream(stream)
            cache[key] = (static, graph, output)
        static, graph, output = cache[key]
        for name, value in inputs.items():
            static[name][:size].copy_(value)
        graph.replay()
        if isinstance(output, tuple):
            return tuple(value[:size] for value in output)
        return output[:size]


ROLLOUT_INFERENCE = CudaRolloutCache()
