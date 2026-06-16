from __future__ import annotations

from typing import Callable, Optional

import torch


class VRAMProfiler:
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

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def reset_peak(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_message(self, tag: str) -> Optional[str]:
        if not self.is_cuda:
            return None

        peak_vram = torch.cuda.max_memory_allocated(self.device) / 1e9
        reserved_vram = torch.cuda.memory_reserved(self.device) / 1e9

        return f"{tag} Peak VRAM: {peak_vram:.4f} GB | Reserved VRAM: {reserved_vram:.4f} GB"

    def log(self, tag: str) -> None:
        msg = self.get_message(tag)

        if msg is not None and self.logger is not None:
            self.logger.log_vram(msg)

    def log_messages(self, *messages: Optional[str]) -> None:
        if self.logger is None:
            return

        for msg in messages:
            if msg is not None:
                self.logger.log_vram(msg)

    def measure_peak_single_batch(self, loader) -> Optional[float]:
        if not self.is_cuda:
            return None

        try:
            batch = next(iter(loader))
            self.reset_peak()
            self.model.eval()

            with torch.no_grad():
                self.forward_batch_fn(batch)

            return torch.cuda.max_memory_allocated(self.device) / 1e9

        except Exception as e:
            if self.logger is not None:
                self.logger.log("ERROR", f"[VRAM] Could not measure peak VRAM: {e}")
            return None
