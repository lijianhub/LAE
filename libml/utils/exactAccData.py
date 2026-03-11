# analyzeAccs_fixed.py
# -------------------------------------------------------------
##只统计 Global Per Task Accs（无中括号），输出 10 列：Global_Task_0..Global_Task_9
# python exactAccData.py --log_dir "C:\Users\40518\code\LAE\logs\cifar100\version_4" --out_csv "global_per_task_accs_cifar100_depi.csv"
# 若文件模式不同，自行调整 --glob
# python analyzeAccs_fixed.py --log_dir /path/to/logs --glob "log.lj.*" --out_csv global_per_task_accs_cifar100_original.csv


# Purpose:
#   - Extract GLOBAL "Global Per Task Accs: 97.00, 93.90" (NO brackets).
#   - For each log file, use ONLY the last occurrence line.
#   - Sort logs by mtime to form phases.
#   - Export ONE CSV: log_filename, Global_Task_0..Global_Task_{N-1}.
#   - Future tasks remain empty -> lower-triangular matrix.
# Notes:
#   - This script ONLY handles Global Per Task Accs (no Local, no brackets).
# -------------------------------------------------------------

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import List, Tuple

# --------------------- Logger ---------------------

def setup_logger(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("analyzeAccs_fixed")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# --------------------- Core Helpers ---------------------

# 仅匹配“无中括号”的格式：
#   Global Per Task Accs: 97.00, 93.90
#   Global Per Task Accs:97.00,93.90
#   Global Per Task Accs : 97.00
# 捕获 ':' 之后的整串文本（不含换行）
FIELD = "Global Per Task Accs"
PATTERN = re.compile(rf"{re.escape(FIELD)}\s*:\s*(.+)$")

def parse_float_list_no_brackets(s: str) -> List[float]:
    """Parse '97.00, 93.90' into [97.00, 93.90]. Ignore non-numeric tokens."""
    vals: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            # 忽略非数字
            continue
    return vals

def extract_latest_array_from_file(file_path: Path, logger: logging.Logger) -> List[float]:
    """
    从文件中提取 'Global Per Task Accs: ...' 的最后一次出现，返回浮点数组。
    若未找到，返回 []。
    """
    latest: List[float] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                m = PATTERN.search(line)
                if not m:
                    continue
                payload = m.group(1).strip()  # 例: "97.00, 93.90"
                arr = parse_float_list_no_brackets(payload)
                if arr:
                    latest = arr  # 只保留最后一次
                    logger.info(f"[{file_path.name}] {FIELD} Line {line_no}: {arr}")
    except Exception as e:
        logger.warning(f"[{file_path.name}] read error: {e}")

    if latest:
        logger.info(f"[{file_path.name}] use latest {FIELD}: {latest}")
    else:
        logger.info(f"[{file_path.name}] no valid '{FIELD}' line found.")
    return latest

def collect_phases(log_dir: Path, glob_pat: str, logger: logging.Logger) -> List[Tuple[str, List[float]]]:
    """
    枚举日志文件，按 mtime 升序；每个文件提取最后一次 Global Per Task Accs。
    返回 [(log_filename, acc_array), ...]
    """
    files = [p for p in log_dir.rglob(glob_pat) if p.is_file()]
    if not files:
        logger.error(f"No log files matched pattern '{glob_pat}' under: {log_dir}")
        return []

    files.sort(key=lambda p: p.stat().st_mtime)  # 时间顺序

    phases: List[Tuple[str, List[float]]] = []
    for p in files:
        arr = extract_latest_array_from_file(p, logger)
        if arr:
            phases.append((p.name, arr))
    logger.info(f"Collected {len(phases)} phases with valid arrays.")
    return phases

def write_triangular_csv(
    phases: List[Tuple[str, List[float]]],
    out_csv: Path,
    num_tasks: int,
    logger: logging.Logger
) -> None:
    """
    输出 CSV：
        log_filename, Global_Task_0, ..., Global_Task_{num_tasks-1}
    每行只填该阶段已出现任务的前 k 列，后续留空（左下三角）。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["log_filename"] + [f"Global_Task_{i}" for i in range(num_tasks)]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for name, arr in phases:
            row = {"log_filename": name}
            for i in range(num_tasks):
                if i < len(arr):
                    row[f"Global_Task_{i}"] = round(float(arr[i]), 2)
                else:
                    row[f"Global_Task_{i}"] = ""
            writer.writerow(row)

    logger.info(f"✅ Wrote {len(phases)} rows to: {out_csv}")

# --------------------- CLI ---------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract GLOBAL 'Global Per Task Accs: x, y, ...' (NO brackets) and export a lower-triangular CSV."
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Root directory containing log files.")
    parser.add_argument("--glob", type=str, default="log.lj.*", help="Filename glob pattern (default: log.lj.*).")
    parser.add_argument("--tasks", type=int, default=10, help="Number of task columns to output (default: 10).")
    parser.add_argument("--out_csv", type=str, default="global_per_task_accs_cifar100_original.csv",
                        help="Output CSV file path (default: global_per_task_accs_cifar100_original.csv).")
    parser.add_argument("--log", type=str, default=None, help="Optional log file path.")

    args = parser.parse_args()
    logger = setup_logger(args.log)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory does not exist: {log_dir}")
        return

    phases = collect_phases(log_dir, args.glob, logger)
    if not phases:
        logger.error("No valid phases extracted; CSV will not be written.")
        return

    write_triangular_csv(phases, Path(args.out_csv), args.tasks, logger)

if __name__ == "__main__":
    main()


