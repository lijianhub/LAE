# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import wandb
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from absl import logging, flags
from timm.utils.model_ema import ModelEmaV2
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from libml.core import App, Conf, DataModule, Module, ModuleConf
from libml.data import build_imagenet_transform
from libml.model.pet import Adapter, Conv2dAdapter, KVLoRA, Prefix
from libml.model.utils import freeze, unfreeze
from libml.model.backbone.swin import SwinTransformer
from libml.model.backbone.convnext import ConvNeXt
from ood.GeneralizedOODFactory import GeneralizedOODFactory


@dataclass
class MyModuleConf(ModuleConf):
    adapt_blocks: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pet_cls: str = "Adapter"  # enum: Adapter, LoRA, Prefix
    pet_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    num_emas: int = 1
    ema_decay: float = 0.9999
    num_freeze_epochs: int = 3
    eval_only_emas: bool = False


@dataclass
class MyConf(Conf):
    module: MyModuleConf = MyModuleConf()


class MyDataModule(DataModule):
    def train_transform(self):
        return build_imagenet_transform(train=True, norm=False)

    def val_transform(self):
        return build_imagenet_transform(train=False, norm=False)


class MyModule(Module):
    pets: nn.Module  # online pet modules
    pets_emas: nn.ModuleList  # offline pet modules
    original_vit: nn.Module
    train_acc: Accuracy
    loss_fn: nn.Module

    def __init__(self, cfg: ModuleConf):
        super().__init__(cfg)
        self.ood_factory = None

    def setup_model(self):
        super().setup_model()

        # Instantiate the Factory
        self.ood_factory = GeneralizedOODFactory(
            tau=0.04,
            lambda_oe=0.5,
            gradnorm_gamma=0.01
        ).to(self.device)

        if getattr(self, "pets_emas", None) is None:
            freeze(self.model.backbone)
            self.pets_emas = nn.ModuleList([])
            self.pets = self.create_pets()
            logging.info(f"==> pets:\n{self.pets}")
        elif len(self.pets_emas) < self.cfg.num_emas:
            idx = len(self.pets_emas)
            ema = ModelEmaV2(self.pets, decay=self.cfg.ema_decay)
            self.pets_emas.append(ema)

        self.train_acc = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

        self.attach_pets(self.pets)

    def create_pets_swin(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        kwargs = self.cfg.pet_kwargs
        adapters = [
            Adapter(blocks[idx].dim, **kwargs) for idx in self.cfg.adapt_blocks
        ]
        return nn.ModuleList(adapters)

    def create_pets_convnext(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [list(stage.blocks.children()) for stage in stages.children()]
        blocks = sum(blocks, [])
        adapters = []
        for idx in self.cfg.adapt_blocks:
            dim = blocks[idx].conv_dw.in_channels
            adapter = Conv2dAdapter(dim, dim, **self.cfg.pet_kwargs)
            adapters.append(adapter)
        return nn.ModuleList(adapters)

    def create_pets_vit(self):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix"]

        n = len(self.cfg.adapt_blocks)
        embed_dim = self.model.backbone.embed_dim

        kwargs = dict(**self.cfg.pet_kwargs)
        if self.cfg.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            return nn.ModuleList([KVLoRA(**kwargs) for _ in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])

    def create_pets(self):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.create_pets_swin()

        if isinstance(self.model.backbone, ConvNeXt):
            return self.create_pets_convnext()

        return self.create_pets_vit()

    def attach_pets_swin(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        for i, b in enumerate(self.cfg.adapt_blocks):
            blocks[b].attach_adapter(attn=pets[i])

    def attach_pets_convnext(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [
            [(idx, stage) for idx, _ in enumerate(stage.blocks.children())]
            for stage in stages.children()
        ]
        blocks = sum(blocks, [])
        for i, b in enumerate(self.cfg.adapt_blocks):
            idx, stage = blocks[b]
            stage.attach_adapter(**{f"blocks.{idx}": pets[i]})

    def attach_pets_vit(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix"]

        if self.cfg.pet_cls == "Adapter":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attach_adapter(attn=pets[i])
            return

        if self.cfg.pet_cls == "LoRA":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_adapter(qkv=pets[i])
            return

        for i, b in enumerate(self.cfg.adapt_blocks):
            self.model.backbone.blocks[b].attn.attach_prefix(pets[i])

    def attach_pets(self, pets: nn.ModuleList):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.attach_pets_swin(pets)

        if isinstance(self.model.backbone, ConvNeXt):
            return self.attach_pets_convnext(pets)

        return self.attach_pets_vit(pets)

    def filter_state_dict(self, state_dict):
        ps = ("model_wrap.", "model.backbone.")
        return {k: v for k, v in state_dict.items() if not k.startswith(ps)}

    def forward(self, input):
        return super().forward(input)

    def pre_train_epoch(self):
        self.train_acc.reset()

        if self.current_task == 0 or self.cfg.num_freeze_epochs < 1:
            return

        if self.current_epoch == 0:
            freeze(self.pets)
            logging.info("===> Freeze pets")

        if self.current_epoch == self.cfg.num_freeze_epochs:
            unfreeze(self.pets)
            logging.info("===> Unfreeze pets")

    def train_step(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_old_classes = self.datamodule.num_old_classes
        output = self(batch[0])
        self.train_acc.update(output, batch[1])
        output = output[:, num_old_classes:]
        target = batch[1] - num_old_classes
        loss = self.loss_fn(output, target)
        loss_dict = {"loss": loss, "acc": self.train_acc.compute() * 100}
        return loss, loss_dict

    def post_train_step(self):
        for idx, ema in enumerate(reversed(self.pets_emas)):
            if idx == 0:  # the last one
                ema.update(self.pets)
            else:
                ema.update(self.pets_emas[idx - 1])

    def eval_epoch_old(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []
        for _, batch in enumerate(loader):
            input, target = batch[:2]
            output, bs = self(input), input.shape[0]
            pred_on.append(output.argmax(dim=1))
            output_emas = [output.softmax(dim=1)]
            for ema in self.pets_emas:
                self.attach_pets(ema.module)
                output_emas.append(self(input).softmax(dim=1))

            for oe in output_emas[1:]:
                pred_off.append(oe.argmax(dim=1))
                break

            if self.cfg.eval_only_emas and len(output_emas) > 1:
                output_emas = output_emas[1:]
            self.attach_pets(self.pets)
            output = torch.stack(output_emas, dim=-1).max(dim=-1)[0]
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])
                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)
        return pred_on, pred_off, pred_ens, gts

    def _backbone_logits_and_features(self, x):
        bb = self.model.backbone
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
        elif hasattr(bb, "get_intermediate_layers"):
            tmp = bb.get_intermediate_layers(x, n=1)[-1]
            feats = tmp[0] if isinstance(tmp, (tuple, list)) else tmp
        else:
            raise RuntimeError("Backbone does not expose a features method.")
        logits = self.model.head(feats)
        return logits, feats

    def eval_epoch_v1(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []

        for _, batch in enumerate(loader):
            input, target = batch[:2]

            # 1. Forward Online (Learning Module)
            # We need both logits and features for GradNorm
            self.attach_pets(self.pets)
            logits_on, features = self._backbone_logits_and_features(input)
            logits_on = self.model.head(logits_on)
            pred_on.append(logits_on.argmax(dim=1))

            # 2. Forward Offline (Accumulation Module)
            # We assume the first EMA in pets_emas is the primary stable expert
            if len(self.pets_emas) > 0:
                self.attach_pets(self.pets_emas[0].module)
                logits_off = self.model(input)
                pred_off.append(logits_off.argmax(dim=1))

                # 3. Decision via GradNorm Factory (Ensemble Module)
                # Uses 3.1.3 logic to calculate weight 'w' and combine experts
                output = self.ood_factory.compute_ensemble(
                    logits_on,
                    features,
                    logits_off,
                    self.model.head[-1]  # Pass the current task head
                )
            else:
                # Case for Task 0 (No EMAs yet)
                output = logits_on.softmax(dim=1)
                pred_off.append(logits_on.argmax(dim=1))

            # Reset PETs to Online for safe state management
            self.attach_pets(self.pets)

            # 4. Standard Metrics and Bookkeeping
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])

                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)

        return pred_on, pred_off, pred_ens, gts

    # Define a head function for the "current task"
    def head_fn(self, feats):
        # Most Dynamic*Head modules will route to the corresponding head
        # based on self.current_task. If its forward() requires a task_id,
        # we pass it in.
        try:
            return self.model.head(feats, task_id=self.current_task)
        except TypeError:
            # If it doesn't require task_id, just forward normally
            return self.model.head(feats)

    def eval_epoch_v2(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []

        for _, batch in enumerate(loader):
            input, target = batch[:2]

            # 1. Forward Online (Learning Module)
            self.attach_pets(self.pets)

            bb = self.model.backbone
            if hasattr(bb, "forward_features"):
                features = bb.forward_features(input)
            elif hasattr(bb, "get_intermediate_layers"):
                tmp = bb.get_intermediate_layers(input, n=1)[-1]
                features = tmp[0] if isinstance(tmp, (tuple, list)) else tmp
            else:
                raise RuntimeError("Backbone must expose forward_features or get_intermediate_layers.")

            logits_on = self.model.head(features)
            pred_on.append(logits_on.argmax(dim=1))

            # 2. Forward Offline (Accumulation Module)
            if len(self.pets_emas) > 0:
                self.attach_pets(self.pets_emas[0].module)
                logits_off = self.model(input)
                pred_off.append(logits_off.argmax(dim=1))

                # 3. Ensemble via OOD Factory
                output = self.ood_factory.compute_ensemble(
                    logits_on,
                    features,
                    logits_off,
                    self.head_fn
                )
            else:
                # Task 0（no EMA）
                #Need softmax
                output = logits_on.softmax(dim=1)
                pred_off.append(logits_on.argmax(dim=1))

            # Reset PETs
            self.attach_pets(self.pets)

            # 4. Metrics
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])

                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)

        return pred_on, pred_off, pred_ens, gts

    def _encode_for_head(self, x):
        """
        Returns the vector features (B, D) fed into DynamicSimpleHead.classify(...).
        Automatically handles 4D CNN backbone features or token/CLS features from ViT.
        """
        bb = self.model.backbone
        # Extract “backbone features”
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
        elif hasattr(bb, "get_intermediate_layers"):
            tmp = bb.get_intermediate_layers(x, n=1)[-1]
            feats = tmp[0] if isinstance(tmp, (tuple, list)) else tmp
        else:
            raise RuntimeError("Backbone must expose forward_features or get_intermediate_layers.")

        # Normalize everything into (B, D) vector features for the head's neck
        if isinstance(feats, dict):
            # Some timm ViTs return a dict; prioritize 'pooled' or 'x'
            feats = feats.get("pooled", feats.get("x", None))
            if feats is None:
                raise RuntimeError("Unsupported features dict structure from backbone.")

        if feats.ndim == 3:
            # (B, N, D) token sequence -> take CLS token
            feats = feats[:, 0]  # (B, D)
            x_vec = self.model.head.neck(feats)  # pass through neck
            return x_vec  # (B, D')
        elif feats.ndim == 4:
            # (B, C, H, W) CNN features -> pooling + neck
            x_vec = self.model.head.pool(feats).flatten(1)  # (B, C)
            x_vec = self.model.head.neck(x_vec)  # (B, D')
            return x_vec
        elif feats.ndim == 2:
            # Already (B, D)
            return self.model.head.neck(feats)
        else:
            raise RuntimeError(f"Unsupported backbone features shape: {feats.shape}")

    def _get_current_task_classifier(self):
        return self.model.head.classifiers[self.current_task]  # nn.Linear(num_features, C_t)

    #@torch.no_grad()
    def eval_epoch(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []

        for _, batch in enumerate(loader):
            input, target = batch[:2]

            # ---- Online ----
            self.attach_pets(self.pets)
            feat_on = self._encode_for_head(input)  # (B, D')
            logits_on = self.model.head.classify(feat_on)  # (B, sumC)
            pred_on.append(logits_on.argmax(dim=1))

            # ---- Offline/EMA ----
            if len(self.pets_emas) > 0:
                self.attach_pets(self.pets_emas[0].module)
                feat_off = self._encode_for_head(input)  # (B, D')
                logits_off = self.model.head.classify(feat_off)  # (B, sumC)
                pred_off.append(logits_off.argmax(dim=1))

                # The linear classifier for the current task (used to compute w for GradNorm)
                cur_clf = self._get_current_task_classifier()

                # Weighted ensembling (returns probabilities, dimension = global sumC)
                output = self.ood_factory.compute_ensemble(
                    logits_on,  # (B, sumC) logits before softmax
                    feat_on,  # (B, D')   online vector features
                    logits_off,  # (B, sumC)
                    cur_clf  # nn.Linear: classifier for the current task
                )
            else:
                # Task 0: keep the same semantics as ensemble output (probabilities)
                output = logits_on.softmax(dim=1)
                pred_off.append(logits_on.argmax(dim=1))

            # Restore PET
            self.attach_pets(self.pets)

            # ---- Metrics ----
            self.val_acc.update(output, target)
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel])

                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local)

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)

        return pred_on, pred_off, pred_ens, gts

    def post_eval_epoch(self, result):
        super().post_eval_epoch()
        if self.current_task == 0 or not flags.FLAGS.debug:
            return

        pred_on, pred_off, pred_ens, gts = result
        pred_on = torch.cat(pred_on)
        pred_off = torch.cat(pred_off) if len(pred_off) else pred_on[:0]
        pred_ens = torch.cat(pred_ens)
        gts = torch.cat(gts)
        torch.save(pred_on, f=f"{self.trainer.ckpt_dir}/pred_on.pt")
        torch.save(pred_off, f=f"{self.trainer.ckpt_dir}/pred_off.pt")
        torch.save(pred_ens, f=f"{self.trainer.ckpt_dir}/pred_ens.pt")
        torch.save(gts, f=f"{self.trainer.ckpt_dir}/gts.pt")

    def configure_optimizer(self, *_):
        return super().configure_optimizer(self.model.head, self.pets)

if __name__ == "__main__":
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run = wandb.init(
        project="LAE",
        name=f"version0-20260103-256-5-cifar100",
        config={
            "lr": 0.0028125,
            "batch_size": 256,
            "num_workers": 8,
            "epochs": 5,
            "dataset": "cifar100",
            "backbone": "ViT-B_16"
        },
        tags=["baseline", "cifar100", "image-classification"],
        notes="ViT-B_16 cifar100 baseline",
        mode="offline"  # run mode（online=realtime，offline=local，dryrun=nothing）
    )

    kwargs = dict(
        datamodule_cls=MyDataModule, module_cls=MyModule, config_cls=MyConf
    )
    App(**kwargs).run()

    wandb.finish()
