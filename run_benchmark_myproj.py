#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, csv, json, argparse, random
from types import SimpleNamespace
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Import your project symbols
from main import MyDataModule, MyModule, MyConf  # <- your classes  (main.py)

# ------------------ basic utils ------------------

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def top1_acc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    return (probs.argmax(1) == targets).float().mean().item()

@torch.no_grad()
def nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    p = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-12)
    return (-p.log()).mean().item()

@torch.no_grad()
def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = pred.eq(targets)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece_val = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        prop = m.float().mean()
        if prop.item() > 0:
            ece_val += (acc[m].float().mean() - conf[m].mean()).abs() * prop
    return ece_val.item()

@torch.no_grad()
def brier(probs: torch.Tensor, targets: torch.Tensor) -> float:
    one_hot = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1.0)
    return ((probs - one_hot) ** 2).sum(1).mean().item()

def switch_rate(pred_seq: List[torch.Tensor]) -> float:
    if not pred_seq: return 0.0
    y = torch.cat(pred_seq, 0)
    if y.numel() <= 1: return 0.0
    return (y[1:] != y[:-1]).float().mean().item()

# ------------------ adapters to your project ------------------

def build_datamodule(args) -> MyDataModule:
    # Instantiate your Conf if needed; here MyDataModule has no custom __init__
    # If your DataModule requires a Conf, adapt accordingly.
    dm = MyDataModule(MyConf())  # safe for your main.py base classes
    # Try to set number of tasks if configurable
    if hasattr(dm, "num_tasks") and args.num_tasks is not None:
        try:
            dm.num_tasks = args.num_tasks
        except Exception:
            pass
    return dm

def build_trainer(args, datamodule: MyDataModule) -> MyModule:
    # Build module with its ModuleConf from MyConf
    conf = MyConf()
    module = MyModule(conf.module)
    # Bind datamodule
    module.datamodule = datamodule
    # Device & model setup
    module.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(module.device)
    module.setup_model()  # will create ood_factory etc.
    # Minimal trainer-like namespace (for file save compatibility)
    module.trainer = SimpleNamespace(ckpt_dir=args.out_dir)
    # Default gating hyperparams (can be updated per-gate later)
    setattr(module, "cfg", SimpleNamespace(
        gate="gradnorm",
        two_stage=True,
        ood_T=1.0
    ))
    return module

def get_eval_loader(datamodule: MyDataModule, task_idx: int):
    """
    Try common names for 'all seen tasks' validation loader.
    Fallback to datamodule.val_dataloader().
    """
    # A few heuristics to locate the correct loader
    for name in [
        "val_dataloader_all_seen",
        "val_dataloader_union",
        "eval_dataloader",
        "val_dataloader"
    ]:
        if hasattr(datamodule, name):
            loader = getattr(datamodule, name)
            try:
                dl = loader() if callable(loader) else loader
                return dl
            except TypeError:
                # some loaders accept task index
                try:
                    return loader(task_idx)
                except Exception:
                    continue
    raise RuntimeError("No suitable eval loader found on your DataModule.")

# ------------------ main benchmark loop ------------------

def run_suite(args: argparse.Namespace):
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(args.out_dir, f"{args.dataset}_{stamp}")
    os.makedirs(root, exist_ok=True)

    summary_rows = []

    for seed in args.seeds:
        set_global_seed(seed)
        print(f"\n== dataset={args.dataset} | seed={seed} | tasks={args.num_tasks} ==")

        dm = build_datamodule(args)
        model = build_trainer(args, dm)

        # Per-seed CSV
        seed_dir = os.path.join(root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        fcsv = open(os.path.join(seed_dir, "metrics_per_task.csv"), "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["seed","gate","task_idx","acc","nll","ece","brier","switch_rate","auc_gate","aupr_gate"])

        for gate in args.gates:
            print(f"\n-- gate = {gate} --")
            # Configure gating mode on the fly
            if gate == "lae-max":
                model.cfg.gate = "lae-max"   # interpreted in your eval logic if you add a branch
                model.cfg.two_stage = False
            elif gate == "deud-energy":
                model.cfg.gate = "energy"
                model.cfg.two_stage = False
            elif gate == "deud-hybrid":
                model.cfg.gate = "hybrid"
                model.cfg.two_stage = bool(args.two_stage)
            else:
                raise ValueError(gate)

            acc_seq, pred_seq_all, targets_all, logits_on_all, logits_off_all, gates_all = [], [], [], [], [], []

            for t in range(args.num_tasks):
                # Let DataModule switch to task t if supported
                if hasattr(dm, "set_task"):
                    try: dm.set_task(t)
                    except Exception: pass

                loader = get_eval_loader(dm, t)
                # Let MyModule know current task (if used internally)
                model.current_task = t

                # Run your eval epoch
                pred_on, pred_off, pred_ens, gts = model.eval_epoch(loader)
                out = model.get_last_epoch_outputs()

                # Aggregate batch lists
                probs_t   = torch.cat(out["probs"],   0) if out["probs"]   else None
                targets_t = torch.cat(out["targets"], 0) if out["targets"] else None
                pred_seq_all += pred_ens
                if probs_t is not None and targets_t is not None:
                    acc_t = top1_acc(probs_t, targets_t)
                    nll_t = nll(probs_t, targets_t)
                    ece_t = ece(probs_t, targets_t, n_bins=args.ece_bins)
                    brier_t = brier(probs_t, targets_t)
                else:
                    # Fallback: accuracy from predictions only
                    preds = torch.cat(pred_ens, 0)
                    tgts  = torch.cat(gts, 0)
                    acc_t = (preds == tgts).float().mean().item()
                    nll_t = ece_t = brier_t = float("nan")

                # Optionally collect for gating AUC/PR
                if out["logits_on"] and out["logits_off"] and out.get("weights", []):
                    logits_on_all.append(torch.cat(out["logits_on"], 0))
                    logits_off_all.append(torch.cat(out["logits_off"], 0))
                    gates_all.append(torch.cat(out["weights"], 0))
                    targets_all.append(targets_t)

                sw = switch_rate(pred_ens)
                acc_seq.append(acc_t)
                writer.writerow([seed, gate, t, acc_t, nll_t, ece_t, brier_t, sw, float("nan"), float("nan")])
                fcsv.flush()
                print(f"task {t:02d}  acc={acc_t:.4f}  nll={nll_t:.4f}  ece={ece_t:.4f}  brier={brier_t:.4f}  sw={sw:.4f}")

            A_N = acc_seq[-1] if acc_seq else float("nan")
            A_bar = float(np.mean(acc_seq)) if acc_seq else float("nan")

            # Optional gating AUC/PR if weights are provided
            auc_gate = aupr_gate = float("nan")
            if gates_all and logits_on_all and logits_off_all and targets_all:
                lg_on = torch.cat(logits_on_all, 0)
                lg_off = torch.cat(logits_off_all, 0)
                tgts = torch.cat(targets_all, 0).view(-1)
                scores = torch.cat(gates_all, 0).view(-1)
                p_on  = F.softmax(lg_on,  dim=1)
                p_off = F.softmax(lg_off, dim=1)
                loss_on  = -p_on.gather(1, tgts.view(-1, 1)).log().view(-1)
                loss_off = -p_off.gather(1, tgts.view(-1, 1)).log().view(-1)
                labels = (loss_on < loss_off).int().cpu().numpy()
                auc_gate  = roc_auc_score(labels, scores.cpu().numpy())
                aupr_gate = average_precision_score(labels, scores.cpu().numpy())

            summary_rows.append({
                "dataset": args.dataset, "seed": seed, "gate": gate,
                "A_N": A_N, "A_bar": A_bar, "AUC_gate": auc_gate, "AUPRC_gate": aupr_gate
            })

        fcsv.close()

    # Save summary
    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(root, "summary_all_seeds.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")
    if not df.empty:
        def mean_std(x): return f"{np.nanmean(x):.4f} ± {np.nanstd(x):.4f}"
        grouped = df.groupby(["dataset","gate"]).agg({"A_N":mean_std, "A_bar":mean_std, "AUC_gate":mean_std, "AUPRC_gate":mean_std})
        print("\n== mean ± std over seeds ==")
        print(grouped.reset_index().to_string(index=False))

def parse_args():
    p = argparse.ArgumentParser("Benchmark runner for your project (LAE vs DEUD).")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--gates", nargs="+", default=["lae-max","deud-energy","deud-hybrid"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--two_stage", type=lambda x: str(x).lower()=="true", default=True)
    p.add_argument("--out_dir", type=str, default="results")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_suite(args)