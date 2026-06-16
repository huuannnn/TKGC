from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.profiler import ProfilerActivity, profile, record_function


class FLOPsProfiler:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        forward_batch_fn: Callable,
        logger=None,
    ) -> None:
        self.model = model
        self.device = device
        self.forward_batch_fn = forward_batch_fn
        self.logger = logger

    def profile_single_forward(self, loader) -> Optional[float]:
        try:
            batch = next(iter(loader))

            activities = [ProfilerActivity.CPU]

            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)

            self.model.eval()

            with torch.no_grad():
                with profile(activities=activities, with_flops=True, record_shapes=True) as prof:
                    with record_function("model_inference"):
                        self.forward_batch_fn(batch)

            total_flops = sum(
                e.flops
                for e in prof.key_averages()
                if hasattr(e, "flops") and e.flops > 0
            )

            return total_flops / 1e9

        except Exception as e:
            if self.logger is not None:
                self.logger.log("ERROR", f"[FLOPs] Could not measure FLOPs: {e}")
            return None
