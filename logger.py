import csv
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger as _logger


class Logger:
    """Log train / val / test / setup into save_dir/version_x."""

    def __init__(self, save_dir: str, version: Optional[Union[int, str]] = None, cons_log: bool = True):
        self._save_dir = Path(save_dir)
        self._version = version
        self._cons_log = cons_log

        _ = self.version
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = str(self.log_dir / f"{self._save_dir.name.lower()}.log")
        self.metrics_file = str(self.log_dir / "metrics.csv")

        self.console_pattern = "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> <level>{level:<6}</level>: <level>{message}</level>"
        self.file_pattern = "[{time:YYYY-MM-DD HH:mm:ss.SSS}] {level:<6}: {message}"

        self.logger = self._configure_logger(self.log_file)

        self._metrics_csv_f = open(self.metrics_file, "w", newline="")
        self._metrics_writer = csv.writer(self._metrics_csv_f)
        self._metrics_writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        self._metrics_csv_f.flush()

    @property
    def version(self) -> Union[int, str]:
        if self._version is not None:
            return self._version

        if not self._save_dir.exists():
            self._version = 0
            return self._version

        versions = []

        for p in self._save_dir.iterdir():
            if p.is_dir() and p.name.startswith("version_"):
                try:
                    versions.append(int(p.name.split("_")[-1]))
                except ValueError:
                    pass

        self._version = max(versions) + 1 if versions else 0
        return self._version

    @property
    def log_dir(self) -> Path:
        return self._save_dir / f"version_{self.version}"

    def get_log_dir(self) -> str:
        return str(self.log_dir)

    def _configure_logger(self, log_path: str):
        _logger.remove()

        levels = {
            "SETUP": {"no": 60, "color": "<cyan>"},
            "TRAIN": {"no": 61, "color": "<green>"},
            "VAL": {"no": 62, "color": "<blue>"},
            "TEST": {"no": 63, "color": "<magenta>"},
            "VRAM": {"no": 64, "color": "<yellow>"},
            "SAVE": {"no": 65, "color": "<black>"},
            "DONE": {"no": 66, "color": "<black>"},
            "ERROR": {"no": 67, "color": "<red>"},
        }

        for level_name, cfg in levels.items():
            try:
                _logger.level(level_name, no=cfg["no"], color=cfg["color"])
            except ValueError:
                pass

        if self._cons_log:
            _logger.add(sys.stdout, level="INFO", format=self.console_pattern, colorize=True)

        _logger.add(log_path, level="INFO", format=self.file_pattern, colorize=False, backtrace=False, diagnose=False)

        return _logger

    def log(self, tag: str, message: str) -> None:
        self.logger.log(tag.upper(), message)

    def write(self, message: str, plain: Optional[str] = None) -> None:
        self.logger.info(plain if plain is not None else message)

    def log_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        self._metrics_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}" if val_loss is not None else ""])
        self._metrics_csv_f.flush()

    def _format_config(self, cfg: dict, indent: int = 2) -> list[str]:
        lines = []

        for key, value in cfg.items():
            prefix = " " * indent

            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._format_config(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {value}")

        return lines

    def log_config(self, model_config: dict, train_config: dict, dataset_config: dict) -> None:
        self.logger.log("SETUP", "========== CONFIG ==========")

        self.logger.log("SETUP", "[Model Config]")
        for line in self._format_config(model_config):
            self.logger.log("SETUP", line)

        self.logger.log("SETUP", "[Training Config]")
        for line in self._format_config(train_config):
            self.logger.log("SETUP", line)

        self.logger.log("SETUP", "[Dataset Config]")
        for line in self._format_config(dataset_config):
            self.logger.log("SETUP", line)

        self.logger.log("SETUP", "============================")

    def log_train_epoch(self, epoch: int, loss: float, epoch_time: float) -> None:
        self.logger.log("TRAIN", f"Epoch {epoch} | Loss = {loss:.6f} | Time = {epoch_time:.2f}s")

    def log_val_epoch(self, epoch: int, loss: float, epoch_time: float) -> None:
        self.logger.log("VAL", f"Epoch {epoch} | Loss = {loss:.6f} | Time = {epoch_time:.2f}s")

    def log_test(self, loss: float, test_time: float) -> None:
        self.logger.log("TEST", f"Loss = {loss:.6f} | Time = {test_time:.2f}s")

    def log_vram(self, message: str) -> None:
        self.logger.log("VRAM", message)

    def log_checkpoint_saved(self, path: str) -> None:
        self.logger.log("SAVE", f"Checkpoint saved to {path}")

    def log_checkpoint_loaded(self, path: str) -> None:
        self.logger.log("SAVE", f"Checkpoint loaded from {path}")

    def log_training_completed(self) -> None:
        self.logger.log("DONE", "Training completed!")

    def close(self) -> None:
        if hasattr(self, "_metrics_csv_f") and self._metrics_csv_f:
            self._metrics_csv_f.close()
            self._metrics_csv_f = None

        self.logger.complete()