#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

python compute_forgetting_metrics.py --csv ./global_per_task_accs_cifar100_depi.csv
# 或显示全部阶段
python compute_forgetting_metrics_nostdlib.py --csv C:\path\to\global_per_task_accs_cifar100_original.csv --show_all

Zero-deps (stdlib-only) continual-learning forgetting metrics.

Input CSV:
  - Header: log_filename,Global_Task_0,...,Global_Task_{T-1}
  - Rows are phases; future tasks are empty strings (lower-triangular).

Output:
  - Print to console:
      * Per-phase Average Forgetting (raw & ReLU)
      * Final UFM (raw & ReLU)
      * BWT
"""

import argparse
import csv
import math
import sys
from typing import List, Tuple


def read_triangular_csv(path: str, prefix: str) -> Tuple[List[str], List[List[float]]]:
    """
    Read CSV and return (phase_names, A), where:
      - phase_names[i]: str
      - A: list of rows; each row is a list[float or math.nan] of length n_tasks
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Task columns in numeric order
        task_cols = [c for c in reader.fieldnames if c and c.startswith(prefix + "Task_")]
        # sort by numeric suffix
        try:
            task_cols.sort(key=lambda c: int(c.split("_")[-1]))
        except Exception:
            pass

        phases: List[str] = []
        A: List[List[float]] = []

        for row in reader:
            phases.append(row.get("log_filename", ""))
            vals: List[float] = []
            for c in task_cols:
                cell = row.get(c, "")
                if cell is None or cell == "":
                    vals.append(math.nan)
                else:
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        vals.append(math.nan)
            A.append(vals)

    return phases, A


def nanmax(arr: List[float]) -> float:
    """Return max ignoring NaN; NaN if all NaN."""
    m = None
    for v in arr:
        if v != v:  # NaN check
            continue
        if m is None or v > m:
            m = v
    return m if m is not None else math.nan


def mean_ignore_nan(arr: List[float]) -> float:
    """Mean ignoring NaN; NaN if none."""
    s = 0.0
    c = 0
    for v in arr:
        if v == v:  # not NaN
            s += v
            c += 1
    return (s / c) if c > 0 else math.nan


def compute_per_phase_avg_forgetting(A: List[List[float]], relu: bool) -> List[float]:
    """
    Chaudhry per-phase Average Forgetting:
      F_i = mean_{j<i} ( max_{l<i} a_{l,j} - a_{i,j} )
    relu=True -> clamp each term with max(0, .)
    """
    n_steps = len(A)
    if n_steps == 0:
        return []

    n_tasks = len(A[0]) if n_steps > 0 else 0
    F = [math.nan] * n_steps

    for i in range(1, n_steps):
        diffs: List[float] = []
        limit = min(i, n_tasks)  # only old tasks j < i
        for j in range(limit):
            cur = A[i][j]
            if cur != cur:  # NaN
                continue
            # historical best up to phase i-1
            hist_col = [A[k][j] for k in range(i)]
            best = nanmax(hist_col)
            if best != best:  # NaN
                continue
            d = best - cur
            if relu and d < 0.0:
                d = 0.0
            diffs.append(d)
        F[i] = mean_ignore_nan(diffs)
    return F


def compute_ufm(A: List[List[float]], relu: bool) -> float:
    """
    UFM = mean_j ( max_l a_{l,j} - a_{T,j} )
    Include only tasks with valid final value and at least one value.
    """
    if not A:
        return math.nan
    n_steps = len(A)
    n_tasks = len(A[0]) if n_steps > 0 else 0
    last = A[-1]

    terms: List[float] = []
    for j in range(n_tasks):
        col = [A[i][j] for i in range(n_steps)]
        # skip if all NaN or final is NaN
        all_nan = all((v != v) for v in col)
        if all_nan or last[j] != last[j]:
            continue
        best = nanmax(col)
        d = best - last[j]
        if relu and d < 0.0:
            d = 0.0
        terms.append(d)
    return mean_ignore_nan(terms)


def compute_bwt(A: List[List[float]]) -> float:
    """
    BWT = mean_i ( a_{T,i} - a_{i,i} ), over tasks with valid diagonal and final.
    """
    if not A:
        return math.nan
    n_steps = len(A)
    n_tasks = len(A[0]) if n_steps > 0 else 0
    final_row = A[-1]

    diffs: List[float] = []
    for i in range(min(n_steps, n_tasks)):
        aii = A[i][i]
        aTi = final_row[i]
        if aii == aii and aTi == aTi:  # both not NaN
            diffs.append(aTi - aii)
    return mean_ignore_nan(diffs)


def main():
    parser = argparse.ArgumentParser(
        description="Compute forgetting metrics from a lower-triangular accuracy CSV (stdlib-only)."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV (e.g., global_per_task_accs_cifar100_original.csv)")
    parser.add_argument("--prefix", default="Global_", help="Task column prefix (default: Global_)")
    parser.add_argument("--show_all", action="store_true",
                        help="Print all phases; otherwise print all if phases<=15 or last 10.")
    args = parser.parse_args()

    try:
        phases, A = read_triangular_csv(args.csv, args.prefix)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    n_steps = len(A)
    n_tasks = len(A[0]) if n_steps > 0 else 0

    print("=" * 70)
    print(f"Loaded CSV: {args.csv}")
    print(f"Shape (phases x tasks): {n_steps} x {n_tasks}")
    print("=" * 70)

    # Per-phase
    F_raw = compute_per_phase_avg_forgetting(A, relu=False)
    F_relu = compute_per_phase_avg_forgetting(A, relu=True)

    # Final
    ufm_raw = compute_ufm(A, relu=False)
    ufm_relu = compute_ufm(A, relu=True)
    bwt = compute_bwt(A)

    # Print per-phase table
    def phase_name(i: int) -> str:
        if i < len(phases) and phases[i]:
            return phases[i]
        return f"phase_{i:02d}"

    if args.show_all or n_steps <= 15:
        print("\nPer-phase Average Forgetting (Chaudhry):")
        print("  i | Phase                               | AvgForget | AvgForget_ReLU")
        print("----+--------------------------------------+-----------+---------------")
        for i in range(n_steps):
            fr = "" if (i >= len(F_raw) or F_raw[i] != F_raw[i]) else f"{F_raw[i]:6.2f}"
            frp = "" if (i >= len(F_relu) or F_relu[i] != F_relu[i]) else f"{F_relu[i]:6.2f}"
            print(f"{i:3d} | {phase_name(i):<36} | {fr:>9} | {frp:>13}")
    else:
        print("\nPer-phase Average Forgetting (last 10 phases):")
        print("  i | Phase                               | AvgForget | AvgForget_ReLU")
        print("----+--------------------------------------+-----------+---------------")
        start = max(0, n_steps - 10)
        for i in range(start, n_steps):
            fr = "" if (i >= len(F_raw) or F_raw[i] != F_raw[i]) else f"{F_raw[i]:6.2f}"
            frp = "" if (i >= len(F_relu) or F_relu[i] != F_relu[i]) else f"{F_relu[i]:6.2f}"
            print(f"{i:3d} | {phase_name(i):<36} | {fr:>9} | {frp:>13}")

    # Final summary
    def fmt(x: float) -> str:
        return "NaN" if (x != x) else f"{x:.2f}"

    print("\nFinal summary based on the last phase:")
    print(f"  UFM_final        : {fmt(ufm_raw)}")
    print(f"  UFM_final_ReLU   : {fmt(ufm_relu)}")
    print(f"  BWT              : {fmt(bwt)}")
    print("=" * 70)


if __name__ == "__main__":
    main()