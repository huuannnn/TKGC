from __future__ import annotations

import torch

from .flops_profiler import FLOPsProfiler
from .vram_profiler import VRAMProfiler


class EfficiencyProfiler:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        forward_batch_fn,
        logger=None,
    ) -> None:
        self.vram = VRAMProfiler(
            model=model,
            device=device,
            forward_batch_fn=forward_batch_fn,
            logger=logger,
        )
        self.flops = FLOPsProfiler(
            model=model,
            device=device,
            forward_batch_fn=forward_batch_fn,
            logger=logger,
        )
        self.logger = logger

    def log_efficiency_stats(self, loader) -> None:
        if self.logger is None:
            return

        self.logger.log("SETUP", "[Efficiency Stats]")

        peak_vram = self.vram.measure_peak_single_batch(loader)
        if peak_vram is not None:
            self.logger.log("SETUP", f"  Peak VRAM (single batch): {peak_vram:.4f} GB")
        else:
            self.logger.log("SETUP", "  Peak VRAM: N/A (CPU mode)")

        gflops = self.flops.profile_single_forward(loader)
        if gflops is not None:
            self.logger.log("SETUP", f"  FLOPs (single forward pass): {gflops:.4f} GFLOPs")
        else:
            self.logger.log("SETUP", "  FLOPs: N/A")
