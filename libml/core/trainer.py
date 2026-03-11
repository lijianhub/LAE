# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from absl import flags, logging
from omegaconf import OmegaConf as oc

from ..utils.config import build_config  # kept for parity with your original imports
from ..utils.tensorboard import TensorboardWriter
from .abc import DataModuleABC, ModuleABC, TrainerABC
from .datamodule import DataConf
from .module import ModuleConf


@dataclass
class TrainerConf:
    seed_everything: int = 42
    accelerator: str = "gpu"
    strategy: Optional[str] = None
    devices: int = 1
    num_nodes: int = 1
    precision: int = 32
    benchmark: bool = True
    deterministic: bool = True
    checkpoint: Optional[str] = None


_FABRIC_FLAGS = ["accelerator", "strategy", "devices", "num_nodes", "precision"]


def get_lite_flags(cfg: TrainerConf) -> Dict[str, Any]:
    return {k: getattr(cfg, k) for k in _FABRIC_FLAGS}


class Trainer(TrainerABC):
    """
    Trainer orchestrates per-task setup, (optionally) checkpoint loading, and running
    train/eval loops across all tasks. In eval mode, it automatically loads the latest
    checkpoint found under <base_dir>/task_<id>/checkpoints/epoch_*.ckpt for each task.
    """

    log_dir: Path
    ckpt_dir: Path

    def __init__(self, cfg: TrainerConf, base_dir: str, **kwargs):
        # Normalize Fabric/Lite flags against the provided config
        kwargs = dict(**get_lite_flags(cfg), **kwargs)
        if kwargs.get("devices", 1) < 2 and kwargs.get("num_nodes", 1) < 2:
            strategy = kwargs.get("strategy", None)
            if strategy is not None:
                logging.warning(
                    "Strategy is set as %s, but there is <=1 device/node; resetting to None.",
                    strategy,
                )
            kwargs["strategy"] = None

        super().__init__(**kwargs)
        self.cfg = cfg
        self.base_dir = base_dir

    # --------------------------------------------------------------------------
    # Checkpoint helpers
    # --------------------------------------------------------------------------
    def _resolve_task_latest_ckpt(self) -> Optional[Path]:
        """
        Resolve the latest checkpoint for the current task:
            <base_dir>/task_<id>/checkpoints/epoch_*.ckpt
        Returns the Path or None if not found.
        """
        ckpt_dir = getattr(self, "ckpt_dir", None)
        if ckpt_dir is None:
            return None
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"))
        return ckpts[-1] if ckpts else None

    def load_checkpoint(self, path: Optional[Path] = None, *, for_eval: bool = False) -> bool:
        """
        Load a checkpoint.

        Priority:
            1) explicit 'path' if provided,
            2) (eval only) latest checkpoint for the current task,
            3) cfg.checkpoint if it points to a valid file.

        Returns:
            True if something was loaded; otherwise False.
        """
        # 1) explicit path
        if path is not None:
            p = Path(path)
            if p.is_file():
                ckpt = self.load(p)
                self.module.on_load_checkpoint(ckpt)
                self.datamodule.on_load_checkpoint(ckpt)
                logging.info(
                    "[Trainer] Loaded checkpoint: %s (task=%d)",
                    str(p),
                    self.datamodule.current_task,
                )
                return True

        # 2) eval-time auto resolve
        if for_eval:
            latest = self._resolve_task_latest_ckpt()
            if latest is not None and latest.is_file():
                ckpt = self.load(latest)
                self.module.on_load_checkpoint(ckpt)
                self.datamodule.on_load_checkpoint(ckpt)
                logging.info(
                    "[Trainer] Loaded checkpoint: %s (task=%d)",
                    str(latest),
                    self.datamodule.current_task,
                )
                return True
            else:
                logging.info(
                    "[Trainer] No checkpoint found for task_%d in %s; evaluating in-memory weights.",
                    self.datamodule.current_task,
                    str(getattr(self, "ckpt_dir", "")),
                )

        # 3) cfg.checkpoint as fallback
        cfg_path = getattr(self.cfg, "checkpoint", None)
        if cfg_path:
            p = Path(cfg_path)
            if p.is_file():
                ckpt = self.load(p)
                self.module.on_load_checkpoint(ckpt)
                self.datamodule.on_load_checkpoint(ckpt)
                logging.info(
                    "[Trainer] Loaded checkpoint (cfg.checkpoint): %s (task=%d)",
                    str(p),
                    self.datamodule.current_task,
                )
                return True

        logging.info(
            "[Trainer] No checkpoint loaded (task=%d).",
            self.datamodule.current_task,
        )
        return False

    # --------------------------------------------------------------------------
    # Per-task setup / save
    # --------------------------------------------------------------------------
    def setup_task(self):
        task_id = self.datamodule.current_task
        self.log_dir = Path(self.base_dir) / f"task_{task_id}"
        self.ckpt_dir = self.log_dir / "checkpoints"

        if self.is_global_zero:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            logging.get_absl_handler().use_absl_log_file("log", self.log_dir)
            self.module.logger = TensorboardWriter(self.log_dir)
            self.save_config()

        # Build/prepare module for this task
        self.module.setup()
        self.module.current_epoch, self.module.global_step = 0, 0

        # Train mode: keep original behavior; Eval mode: load latest checkpoint of this task
        if getattr(self, "mode", "train") == "train":
            if task_id == self.start_task and self.cfg.checkpoint:
                self.load_checkpoint()  # uses cfg.checkpoint
        else:
            self.load_checkpoint(for_eval=True)  # prefer latest task ckpt

    def save_checkpoint(self):
        if self.is_global_zero:
            ckpt, current_epoch = dict(), self.module.current_epoch
            self.module.on_save_checkpoint(ckpt)
            self.datamodule.on_save_checkpoint(ckpt)
            self.save(ckpt, self.ckpt_dir / f"epoch_{current_epoch}.ckpt")

    def save_config(self):
        cfg = oc.create(oc.to_container(self.cfg))
        self.datamodule.on_save_config(cfg)
        self.module.on_save_config(cfg)
        oc.save(cfg, self.log_dir / "config.yaml")

    # --------------------------------------------------------------------------
    # Task/run wiring
    # --------------------------------------------------------------------------
    def run_task(self, mode: str = "train"):
        self.module.run(mode)

    def run(self, mode: str = "train"):
        # Store mode so setup_task() can branch on train/eval behavior
        self.mode = mode

        self.launch()
        self.seed_everything(self.cfg.seed_everything)

        # Make deterministic/benchmark consistent
        torch.backends.cudnn.deterministic = self.cfg.deterministic
        torch.backends.cudnn.benchmark = self.cfg.benchmark and not self.cfg.deterministic

        # Abseil logging preferences
        flags.FLAGS.alsologtostderr = True
        flags.FLAGS.showprefixforinfo = False

        # Reduce logging on non-zero ranks, guard rmdir
        if not self.is_global_zero:
            flags.FLAGS.verbosity = -1000  # disable logging
            try:
                Path(self.base_dir).rmdir()  # may fail if not empty
            except Exception:
                pass

        # Broadcast base_dir across ranks (DDP launch issue workaround)
        self.base_dir = self.broadcast(self.base_dir)

        # Wire runtime objects
        self.module.datamodule = self.datamodule
        self.module.trainer = self
        self.module.device = self.device

        dm = self.datamodule
        # Iterate all tasks from current to last
        for dm.current_task in range(dm.current_task, dm.num_tasks):
            self.setup_task()
            self.run_task(mode)
            # Do NOT break in eval; we want to process all tasks