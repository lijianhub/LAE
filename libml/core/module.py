# -*- coding: utf-8 -*-

import gc
import wandb
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from absl import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy

from ..model import Model, ModelConf
from ..optim import (
    OptimizerConf,
    SchedulerConf,
    create_optimizer,
    create_scheduler,
)
from ..utils.tensorboard import TensorboardWriter
from .abc import DataModuleABC, ModuleABC, TrainerABC


@dataclass
class ModuleConf:
    current_epoch: int = 0
    num_epochs: int = 5
    model: ModelConf = ModelConf()
    optimizer: OptimizerConf = OptimizerConf()
    scheduler: SchedulerConf = SchedulerConf()
    schedule_unit: str = "none"  # enum epoch, step
    clip_grad_norm: float = 0.0
    clip_grad_value: float = 0.0
    accumulate_grad: int = 1
    eval_every_n_epoch: int = 1
    log_every_n_step: int = 10
    summary_depth: int = 5


class Module(ModuleABC):
    model: nn.Module
    model_wrap: Union[nn.Module, DistributedDataParallel]
    train_loader: DataLoader
    val_loader: DataLoader
    val_acc: Accuracy
    val_task_accs: nn.ModuleList
    val_task_local_accs: nn.ModuleList

    def __init__(self, cfg: ModuleConf):
        super().__init__()
        self.cfg = cfg
        self.model = None

    def setup(self):
        self.setup_loader()
        self.setup_model()
        optimizer, self.scheduler = self.configure_optimizer()
        self.model_wrap, self.optimizer = self.wrap(self.model, optimizer)

        logging.info(f"==> model overview:\n{self.model}")

        if self.cfg.summary_depth < 0:
            return

        input_size = next(iter(self.val_loader))[0][:1].shape
        with self.no_wrap():
            kwargs = dict(row_settings=("var_names",), verbose=0)
            kwargs["depth"] = self.cfg.summary_depth
            model_summary = summary(self, input_size, **kwargs)
        logging.info(f"==> model summary:\n{model_summary}\n")

    def setup_loader(self):
        torch.cuda.empty_cache()
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()
        gc.collect()

    def setup_model(self):
        if self.model is None:
            self.model = Model(self.cfg.model)
            for task_id in range(self.current_task):
                self.model.head.append(self.datamodule.num_classes_of(task_id))

        self.model.head.append(self.datamodule.num_new_classes)
        self.model.to(self.device).train()

        n_tasks = self.current_task + 1
        self.val_acc = Accuracy()
        self.val_task_accs = nn.ModuleList([Accuracy() for _ in range(n_tasks)])
        self.val_task_local_accs = nn.ModuleList(
            [Accuracy() for _ in range(n_tasks)]
        )

    def filter_state_dict(self, state_dict):
        return {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("model_wrap.")
        }

    def on_load_checkpoint(self, ckpt):
        state_dict = self.filter_state_dict(ckpt["state_dict"])
        self.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.current_epoch = ckpt["epoch"] + 1

    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = self.filter_state_dict(self.state_dict())
        ckpt["optimizer"] = self.optimizer.state_dict()
        ckpt["scheduler"] = self.scheduler.state_dict()
        ckpt["epoch"] = self.current_epoch

    @contextmanager
    def no_wrap(self):
        model_wrap, self.model_wrap = self.model_wrap, None
        try:
            yield
        finally:
            self.model_wrap = model_wrap

    def forward(self, *args, **kwargs):
        model = getattr(self, "model_wrap", None) or self.model
        return model(*args, **kwargs)

    def on_save_config(self, cfg: DictConfig):
        cfg.module = oc.create(oc.to_container(self.cfg))

    def run(self, mode: str = "train"):
        if mode == "eval":
            return self.evaluate()

        loader = self.trainer.setup_dataloaders(self.train_loader)
        num_batches = len(loader)

        self.train()
        num_epochs = self.cfg.num_epochs
        for self.current_epoch in range(self.current_epoch, num_epochs):
            self.pre_train_epoch()
            self.train_epoch(loader, num_batches)
            if self.cfg.schedule_unit == "epoch":
                self.scheduler.step()

            next_epoch = self.current_epoch + 1
            interval = self.cfg.eval_every_n_epoch
            if next_epoch % interval == 0 or next_epoch == self.cfg.num_epochs:
                self.evaluate(getattr(self, "val_loader", None))
                self.trainer.save_checkpoint()
            if self.global_rank == 0:
                kwargs = {"epoch": next_epoch}
                self.logger.write_scalars(self.global_step, kwargs)
            self.post_train_epoch()

        if self.global_rank == 0 and self.datamodule.memory is not None:
            model = self.model.clone(freeze=True).eval()
            model.head.feature_mode = True
            kwargs = dict(mock_test=True, with_memory=False)
            loader_train = self.datamodule.train_dataloader(**kwargs)
            self.datamodule.memory.update(model, loader_train)

        # === Evaluate hyperparameter sweeps at the very end (if configured) ===
        try:
            sweeps = getattr(self.cfg, 'eval_sweeps', None)
            if sweeps is not None:
                from omegaconf import OmegaConf as oc
                sweeps_dict = oc.to_container(sweeps, resolve=True)
                if isinstance(sweeps_dict, dict) and len(sweeps_dict) > 0:
                    self.evaluate_sweeps(getattr(self, 'val_loader', None))
        except Exception as e:
            logging.error(f'Failed to run eval_sweeps: {e}')

    def train_epoch(self, loader, num_batches):
        nt = len(str(self.datamodule.scenario_train.nb_tasks - 1))
        ne = len(str(self.cfg.num_epochs - 1))
        nb = len(str(num_batches - 1))
        na = max(self.cfg.accumulate_grad, 1)
        fmt = f"T%0{nt}d-E%0{ne}d-B%0{nb}d | "
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(loader):
            if num_batches - batch_idx < na:  # skip
                continue

            self.pre_train_step()
            batch = self.to_device(batch[:2])
            skip_sync = na > 1 and (batch_idx + 1) % na != 0
            with self.trainer.no_backward_sync(self.model_wrap, skip_sync):
                with self.autocast():
                    loss, loss_dict = self.train_step(batch, batch_idx)
                self.backward(loss / na)

            if (batch_idx + 1) % na == 0:
                if self.cfg.clip_grad_norm > 0:
                    args = (self.parameters(), self.cfg.clip_grad_norm)
                    loss_dict["grad_norm"] = nn.utils.clip_grad_norm_(*args)
                if self.cfg.clip_grad_value > 0:
                    args = (self.parameters(), self.cfg.clip_grad_value)
                    nn.utils.clip_grad_value_(self.parameters(), *args)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.cfg.schedule_unit == "step":
                self.scheduler.step()

            next_step, interval = batch_idx + 1, self.cfg.log_every_n_step
            if next_step % interval == 0 or next_step == num_batches:
                msg = fmt % (self.current_task, self.current_epoch, batch_idx)
                msg += " | ".join(
                    [f"{k}: {v:.4f}" for k, v in loss_dict.items()]
                )
                msg += f" | lr: {self.optimizer.param_groups[0]['lr']:.8f}"
                logging.info(msg)
                loss_dict = {f"loss/{k}": v for k, v in loss_dict.items()}
                if self.global_rank == 0:
                    self.logger.write_scalars(self.global_step, loss_dict)
            self.global_step += 1
            self.post_train_step()

    def pre_eval_epoch(self):
        self.model.eval()

    def eval_epoch(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        for batch_idx, batch in enumerate(loader):
            output = self.eval_step(batch, batch_idx)
            target = batch[1]
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])
                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

    def eval_step(self, batch, batch_idx):
        return self(batch[0])

    def post_eval_epoch(self, *_):
        self.model.train()

    def _iter_eval_sweep_combos(self):
        """Yield dicts of all combinations in cfg.eval_sweeps (cartesian product).
        Returns [] if no sweeps configured.
        """
        from itertools import product
        sweeps = getattr(self.cfg, 'eval_sweeps', None)
        try:
            from omegaconf import OmegaConf as oc
        except Exception:
            oc = None
        if sweeps is None or (hasattr(sweeps, '__len__') and len(sweeps) == 0):
            return []
        # Convert OmegaConf -> Python dict
        if oc is not None:
            sweeps = oc.to_container(sweeps, resolve=True)
        keys = list(sweeps.keys())
        vals = [sweeps[k] for k in keys]
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
        for combo_vals in product(*vals):
            yield dict(zip(keys, combo_vals))

    def _format_combo_label(self, combo: dict) -> str:
        # Build a compact label that will appear in console and W&B
        def g(k, default):
            return combo.get(k, getattr(self.cfg, k, default))
        gate = g('gate', 'hybrid')
        T = g('ood_T', 1.0)
        alpha = g('hybrid_alpha', 0.7)
        two = g('two_stage', True)
        low = g('two_stage_low_q', 0.2)
        high = g('two_stage_high_q', 0.8)
        band = g('two_stage_band', 'w')
        return f"gate={gate}|T={T}|alpha={alpha}|two={two}|q={low}-{high}|band={band}"

    @torch.no_grad()
    def evaluate_sweeps(self, loader: DataLoader = None):
        """Run evaluation over all cartesian combinations of module.eval_sweeps.
        Logs an identifying label per combo and emits W&B fields under the prefix 'EvalSweep/'.
        """
        combos = list(self._iter_eval_sweep_combos())
        if not combos:
            return
        loader = self.val_loader if loader is None else loader

        # Stash original cfg values we will override
        keys = sorted({k for c in combos for k in c.keys()})
        orig = {k: getattr(self.cfg, k, None) for k in keys}

        logging.info(f"==> EvalSweeps: {len(combos)} combinations to evaluate …")
        for i, combo in enumerate(combos, 1):
            # Apply override for this combo
            for k, v in combo.items():
                setattr(self.cfg, k, v)
            label = self._format_combo_label(combo)
            logging.info(f"==> [SWEEP {i}/{len(combos)}] {label}")

            # Prefer evaluate_v2 (更多指标); fallback evaluate
            eval_fn = getattr(self, 'evaluate_v2', None) or getattr(self, 'evaluate', None)
            eval_fn(loader)

            # Emit sweep-identifying fields to W&B
            sweep_log = {f"EvalSweep/{k}": v for k, v in combo.items()}
            sweep_log["EvalSweep/label"] = label
            sweep_log["EvalSweep/index"] = i
            try:
                wandb.log(sweep_log)
            except Exception:
                pass

        # Restore original config
        for k, v in orig.items():
            setattr(self.cfg, k, v)
        logging.info("==> EvalSweeps: completed and restored base config.")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None):
        loader = self.val_loader if loader is None else loader
        loader = self.trainer.setup_dataloaders(loader)

        self.val_acc.reset()
        _ = [acc.reset() for acc in self.val_task_accs]
        _ = [acc.reset() for acc in self.val_task_local_accs]

        self.pre_eval_epoch()
        result = self.eval_epoch(loader)
        self.post_eval_epoch(result)

        n_tasks = self.current_task + 1
        render = lambda ms: ", ".join([f"{m.compute() * 100:.2f}" for m in ms])
        logging.info(
            "==> Evaluation results %d"
            "\n\tAcc: %.2f"
            "\n\tGlobal Per Task Accs: %s"
            "\n\tGlobal Task Accs Avg: %.2f"
            "\n\tLocal Per Task Accs: %s",
            self.current_epoch,
            self.val_acc.compute() * 100,
            render(self.val_task_accs),
            sum([acc.compute() * 100 for acc in self.val_task_accs]) / n_tasks,
            render(self.val_task_local_accs),
        )

        val_acc_pct = self.val_acc.compute().cpu().item() * 100
        task_accs_pct = [acc.compute().cpu().item() * 100 for acc in self.val_task_accs]
        global_task_acc_avg = sum(task_accs_pct) / n_tasks

        val_acc_pct_rounded = round(val_acc_pct, 2)
        global_task_acc_avg_rounded = round(global_task_acc_avg, 2)

        # ToDo: add learning forgetting rate
        # loss: 0.3238 accu
        wandb.log({
            "Evaluation/taskId": self.current_task,
            "Evaluation/Epoch": self.current_epoch,
            "Evaluation/Accuracy(%)": val_acc_pct_rounded,  # 已格式化的数值
            "Evaluation/GlobalPerTaskAccs": render(self.val_task_accs),
            "Evaluation/GlobalTaskAccsAvg(%)": global_task_acc_avg_rounded,  # 已格式化的数值
            "Evaluation/LocalPerTaskAccs": render(self.val_task_local_accs)
        })

    @torch.no_grad()
    def evaluate_v2(self, loader: DataLoader = None):
        """
        Full evaluation for one epoch.
        - Keeps your existing global/per-task metrics.
        - Adds optional NLL/ECE/Brier/SwitchRate and (if available) gating AUC/PR.
        - Logs both console and W&B.
        """
        # Prepare loader
        loader = self.val_loader if loader is None else loader
        loader = self.trainer.setup_dataloaders(loader)

        # Reset public metrics you already maintain
        self.val_acc.reset()
        _ = [acc.reset() for acc in self.val_task_accs]
        _ = [acc.reset() for acc in self.val_task_local_accs]

        # Forward evaluation pass (your eval_epoch will also record per-batch outputs)
        self.pre_eval_epoch()
        result = self.eval_epoch(loader)  # (pred_on, pred_off, pred_ens, gts)
        self.post_eval_epoch(result)

        # ===== Your existing logging (global + per-task) =====
        n_tasks = self.current_task + 1
        render = lambda ms: ", ".join([f"{m.compute() * 100:.2f}" for m in ms])
        logging.info(
            "==> Evaluation results %d"
            "\n\tAcc: %.2f"
            "\n\tGlobal Per Task Accs: %s"
            "\n\tGlobal Task Accs Avg: %.2f"
            "\n\tLocal Per Task Accs: %s",
            self.current_epoch,
            self.val_acc.compute() * 100,
            render(self.val_task_accs),
            sum([acc.compute() * 100 for acc in self.val_task_accs]) / n_tasks,
            render(self.val_task_local_accs),
        )

        val_acc_pct = self.val_acc.compute().cpu().item() * 100
        task_accs_pct = [acc.compute().cpu().item() * 100 for acc in self.val_task_accs]
        global_task_acc_avg = sum(task_accs_pct) / n_tasks

        val_acc_pct_rounded = round(val_acc_pct, 2)
        global_task_acc_avg_rounded = round(global_task_acc_avg, 2)

        # ===== Optional: compute additional reliability/stability metrics =====
        # We depend on eval_epoch(...) having recorded per-batch outputs into self._last_eval.
        extra = getattr(self, "get_last_epoch_outputs", None)
        nll_val = float("nan")
        ece_val = float("nan")
        brier_val = float("nan")
        switch_val = float("nan")
        auc_gate = float("nan")
        aupr_gate = float("nan")

        if extra is not None:
            cache = self.get_last_epoch_outputs() or {}
            probs_list = cache.get("probs", [])
            targets_list = cache.get("targets", [])
            preds_list = cache.get("preds", [])
            weights_list = cache.get("weights", [])
            logits_on_ls = cache.get("logits_on", [])
            logits_off_ls = cache.get("logits_off", [])

            # Global sequences for switch-rate from predictions you already returned
            pred_on, pred_off, pred_ens, gts = result
            if pred_ens:
                # Flatten to a single sequence in loader order
                y = torch.cat(pred_ens, dim=0)
                if y.numel() > 1:
                    switch_val = (y[1:] != y[:-1]).float().mean().item()

            # Reliability metrics if probabilities are available
            if probs_list and targets_list:
                probs = torch.cat(probs_list, dim=0)
                targets = torch.cat(targets_list, dim=0)
                # NLL
                p = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-12)
                nll_val = float((-p.log()).mean().item())
                # ECE (15-bin default; adjust if you prefer)
                conf, pred = probs.max(dim=1)
                acc = pred.eq(targets)
                n_bins = 15
                bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
                ece_tmp = torch.zeros(1, device=probs.device)
                for i in range(n_bins):
                    m = (conf > bins[i]) & (conf <= bins[i + 1])
                    prop = m.float().mean()
                    if prop.item() > 0:
                        ece_tmp += (acc[m].float().mean() - conf[m].mean()).abs() * prop
                ece_val = float(ece_tmp.item())
                # Brier
                one_hot = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1.0)
                brier_val = float(((probs - one_hot) ** 2).sum(dim=1).mean().item())

            # Optional gating AUC / AUPRC if you expose: logits_on/off + weights
            if logits_on_ls and logits_off_ls and weights_list and targets_list:
                try:
                    logits_on = torch.cat(logits_on_ls, dim=0)
                    logits_off = torch.cat(logits_off_ls, dim=0)
                    gates = torch.cat(weights_list, dim=0).view(-1)
                    targets = torch.cat(targets_list, dim=0).view(-1)
                    p_on = torch.softmax(logits_on, dim=1)
                    p_off = torch.softmax(logits_off, dim=1)
                    loss_on = -p_on.gather(1, targets.view(-1, 1)).log().view(-1)
                    loss_off = -p_off.gather(1, targets.view(-1, 1)).log().view(-1)
                    labels = (loss_on < loss_off).int().cpu().numpy()
                    scores = gates.detach().cpu().numpy()
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    auc_gate = float(roc_auc_score(labels, scores))
                    aupr_gate = float(average_precision_score(labels, scores))
                except Exception:
                    pass  # keep NaNs if something is missing

        # ===== W&B logging (keeps your fields, adds optional metrics if available) =====
        wandb.log({
            "Evaluation/taskId": self.current_task,
            "Evaluation/Epoch": self.current_epoch,
            "Evaluation/Accuracy(%)": val_acc_pct_rounded,
            "Evaluation/GlobalPerTaskAccs": render(self.val_task_accs),
            "Evaluation/GlobalTaskAccsAvg(%)": global_task_acc_avg_rounded,
            "Evaluation/LocalPerTaskAccs": render(self.val_task_local_accs),
            # Optional reliability/stability fields (may be NaN if not recorded)
            "Evaluation/NLL": nll_val,
            "Evaluation/ECE": ece_val,
            "Evaluation/Brier": brier_val,
            "Evaluation/SwitchRate": switch_val,
            "Evaluation/GatingAUC": auc_gate,
            "Evaluation/GatingAUPRC": aupr_gate,
        })

    def configure_optimizer(
        self, *modules: List[nn.Module]
    ) -> Tuple[Optimizer, LRScheduler]:
        modules = modules if modules else [self.model]
        optimizer = create_optimizer(self.cfg.optimizer, *modules)
        num_steps = self.cfg.num_epochs
        if self.cfg.schedule_unit == "step":
            num_steps *= len(self.datamodule.train_dataloader())
        elif self.cfg.schedule_unit == "none":
            num_steps = float("inf")
        scheduler = create_scheduler(self.cfg.scheduler, optimizer, num_steps)
        return optimizer, scheduler
