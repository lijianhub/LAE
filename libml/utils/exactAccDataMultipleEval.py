"""
 python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/cifar100/version_12" `
   --glob "log.lj.*" `
   --tasks 10 `
   --out_csv "global_per_task_accs_cifar100_12_dedup.csv"
 python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_lora/version_5" `
   --glob "log.lj.*" `
   --tasks 10 `
   --out_csv "global_per_task_accs_vit_lora_v5_dedup.csv"

 python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_lora/version_7" `
   --glob "log.lj.*" `
   --tasks 20 `
   --out_csv "global_per_task_accs_vit_lora_v7_dedup.csv"
 python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_adapter/version_1" `
   --glob "log.lj.*" `
   --tasks 10 `
   --out_csv "global_per_task_accs_vit_adapter_v1_dedup.csv"

 python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_lora/version_8" `
   --glob "log.lj.*" `
   --tasks 20 `
   --out_csv "global_per_task_accs_vit_lora_v8_dedup.csv"
python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_lora/version_9" `
   --glob "log.lj.*" `
   --tasks 10 `
   --out_csv "global_per_task_accs_vit_lora_v9_dedup.csv"
python exactAccDataMultipleEval.py `
   --log_dir "C:/Users/40518/code/LAE/logs/vit_lora/version_11" `
   --glob "log.lj.*" `
   --tasks 10 `
   --out_csv "global_per_task_accs_vit_lora_v11_dedup.csv"
"""
# exactAccDataMultipleEval.py
# -----------------------------------------------------------------------------
# 需求：
# 1) 递归遍历 log_dir（task_0 ...）找日志文件。
# 2) baseline（pre-sweep）只保留最后一次 Evaluation 的 "Global Per Task Accs"。
# 3) 从出现 [SWEEP i/N] 后，逐个 sweep 提取其对应的 "Global Per Task Accs"（一对一）。
# 4) CSV 列：task_name, kind, sweep_idx, sweep_total, sweep_line, Global_Task_0..N
#    - 去掉 accs 与 eval_line 两列；
#    - sweep_line 只保留 “[SWEEP i/N] ...” 这部分（后半段参数保留）。
# 5) 兼容：
#    - "==>" 与 "==&gt;"；
#    - "Evaluation result" 与 "Evaluation results"；
#    - glog 时间戳；
#    - "评估头 与 Global Per Task Accs 在同一行" 的情况。
# -----------------------------------------------------------------------------
import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# --------------------- Logger ---------------------
def setup_logger(log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("extract_accs")
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

# --------------------- Task name helper ---------------------
TASK_NAME_RE = re.compile(r"^task_\d+$", re.IGNORECASE)

def get_task_name(file_path: Path, root: Path) -> str:
    """优先用相对路径中的 task_x 名称；否则上级目录名；再否则文件名 stem。"""
    try:
        rel = file_path.relative_to(root)
    except Exception:
        rel = file_path
    for part in rel.parts:
        if TASK_NAME_RE.match(part):
            return part
    if file_path.parent and file_path.parent.name:
        return file_path.parent.name
    return file_path.stem

# --------------------- Patterns ---------------------
# glog 样式：I/W/E/F + MMDD + ' HH:MM:SS.micros pid file:line]'
TS_LINE = re.compile(
    r"^[IWEF]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{6}\s+\d+\s+\S+:\d+\]",
)

# 兼容 "==>" 与 "==&gt;"，兼容 result / results
PAT_EVAL_HEAD = re.compile(
    r"==\s*(?:>|&gt;)\s*Evaluation\s+result(?:s)?\s*(\d+)",
    re.IGNORECASE,
)

# [SWEEP i/N] + 后半部分参数整行
PAT_SWEEP_HEAD = re.compile(
    r"==\s*(?:>|&gt;)\s*(\[SWEEP\s*(\d+)\s*/\s*(\d+)\]\s*.*)$",
    re.IGNORECASE,
)

FIELD = "Global Per Task Accs"

# 支持数组或单值，并允许前面有其他内容（兼容“拼在同一行”）
# 例如：
#   " ... ==> Evaluation result 4 Global Per Task Accs: 89.50, 88.30"
#   " ... [SWEEP 1/24] gate=... Global Per Task Accs: 93.70, 86.40"
PAT_GLOBAL_ANYWHERE = re.compile(
    rf"{re.escape(FIELD)}\s*:\s*(\[[^\]]*\]|(?:[+-]?\d+(?:\.\d+)?)(?:\s*,\s*[+-]?\d+(?:\.\d+)?)*)(?!\])",
    re.IGNORECASE,
)

def parse_float_list(text: str) -> List[float]:
    """将 [a, b, c] 或 'a, b, c' / 单值 'x' 转为 float 列表。"""
    s = text.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    vals: List[float] = []
    for tok in re.split(r"\s*,\s*|\s+", s):
        tok = tok.strip().strip(",")
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals

def is_ts_line(line: str) -> bool:
    return bool(TS_LINE.match(line))

# --------------------- Core extraction per file ---------------------
def extract_from_file(
    file_path: Path,
    logger: logging.Logger,
) -> Dict[str, List[Dict]]:
    """
    返回：
      {
        "base_last": [ {accs: [...]} ],  # 仅一个（最后一次 baseline）
        "sweeps":    [ {idx, total, sweep_line, accs}, ... ]  # 每个 sweep 一条
      }
    """
    base_last: Optional[Dict] = None
    sweeps: List[Dict] = []

    sweep_started = False
    # baseline 阶段：不断覆盖，最后一个即为 base_last
    pending_base_eval_seen = False  # 标识是否在 baseline 段见过 eval 头
    last_baseline_accs: Optional[List[float]] = None

    # sweep 阶段：记录最近一个 sweep 头，直到拿到 accs 为止
    pending_sweep: Optional[Dict] = None  # {idx, total, sweep_line}

    def finalize_baseline_if_needed():
        nonlocal base_last, last_baseline_accs
        if last_baseline_accs is not None:
            base_last = {"accs": last_baseline_accs}
            last_baseline_accs = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # 1) 如果遇到 SWEEP 头
            m_sw = PAT_SWEEP_HEAD.search(line)
            if m_sw:
                # 进入 sweep 段前，先结算 baseline 的最后一次
                if not sweep_started:
                    finalize_baseline_if_needed()
                sweep_started = True

                sweep_full = m_sw.group(1).strip()  # "[SWEEP i/N] params..."
                # 提取 i, N
                m_idx = re.match(r"\[SWEEP\s*(\d+)\s*/\s*(\d+)\]", sweep_full, flags=re.IGNORECASE)
                if m_idx:
                    idx = int(m_idx.group(1))
                    total = int(m_idx.group(2))
                else:
                    idx, total = "", ""

                # 只保留 “[SWEEP i/N] 后半部分”
                sweep_line = sweep_full
                pending_sweep = {"idx": idx, "total": total, "sweep_line": sweep_line}
                continue

            # 2) baseline 段：记录 eval 头并抓 accs（可能在同一行）
            if not sweep_started:
                if PAT_EVAL_HEAD.search(line):
                    pending_base_eval_seen = True
                    # 若与 accs 同行，立即抓取
                    m_ga_inline = PAT_GLOBAL_ANYWHERE.search(line)
                    if m_ga_inline:
                        last_baseline_accs = parse_float_list(m_ga_inline.group(1))
                    continue

                # baseline 段中也可能出现单独一行的 Global Per Task Accs
                m_ga = PAT_GLOBAL_ANYWHERE.search(line)
                if m_ga and pending_base_eval_seen:
                    last_baseline_accs = parse_float_list(m_ga.group(1))
                    # 不立即 finalize，直到遇到 sweep 或文件结束前随时可能被覆盖为“最后一次”
                continue

            # 3) sweep 段：每个 sweep 捕获一次 accs（可能和 sweep 或 eval 头在同一行）
            if sweep_started:
                # 如果 accs 跟在 sweep 或 eval 头同一行，这里也能抓到
                m_ga = PAT_GLOBAL_ANYWHERE.search(line)
                if m_ga and pending_sweep is not None and "accs" not in pending_sweep:
                    pending_sweep["accs"] = parse_float_list(m_ga.group(1))
                    sweeps.append(pending_sweep)
                    pending_sweep = None
                    continue

                # 有些日志会先出现 "Evaluation result ..." 再出现下一行 accs
                if PAT_EVAL_HEAD.search(line):
                    # 允许 eval 头出现，但真正入列在捕到 accs 时
                    m_inline = PAT_GLOBAL_ANYWHERE.search(line)
                    if m_inline and pending_sweep is not None and "accs" not in pending_sweep:
                        pending_sweep["accs"] = parse_float_list(m_inline.group(1))
                        sweeps.append(pending_sweep)
                        pending_sweep = None
                    continue

    # 文件结束：baseline 最终结算
    if not sweep_started:
        finalize_baseline_if_needed()

    result = {"base_last": [], "sweeps": sweeps}
    if base_last:
        result["base_last"].append(base_last)
    return result

# --------------------- CSV writer ---------------------
def write_combined_csv(
    all_rows: List[Tuple[str, Dict, str]],  # (log_name, payload, task_name)
    out_csv: Path,
    tasks: int,
    logger: logging.Logger,
):
    headers = [
        "task_name", "kind", "sweep_idx", "sweep_total", "sweep_line"
    ] + [f"Global_Task_{i}" for i in range(tasks)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for log_name, payload, task_name in all_rows:
            # baseline（只保留最后一次）
            for base in payload.get("base_last", []):
                row = {
                    "task_name": task_name,
                    "kind": "base_last",
                    "sweep_idx": "",
                    "sweep_total": "",
                    "sweep_line": "",
                }
                arr = base.get("accs", [])
                for i in range(tasks):
                    row[f"Global_Task_{i}"] = (round(float(arr[i]), 4) if i < len(arr) else "")
                writer.writerow(row)

            # sweeps：每个 sweep 一条
            for sw in payload.get("sweeps", []):
                row = {
                    "task_name": task_name,
                    "kind": "sweep",
                    "sweep_idx": sw.get("idx", ""),
                    "sweep_total": sw.get("total", ""),
                    # 只保留 “[SWEEP i/N] ...” 整段（已经是后半部分）
                    "sweep_line": sw.get("sweep_line", ""),
                }
                arr = sw.get("accs", [])
                for i in range(tasks):
                    row[f"Global_Task_{i}"] = (round(float(arr[i]), 4) if i < len(arr) else "")
                writer.writerow(row)

    logger.info(f"✅ Wrote CSV: {out_csv}")

# --------------------- CLI ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract last baseline 'Global Per Task Accs' and each sweep's accs recursively, export CSV with task_name."
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Root directory containing logs (will be scanned recursively).")
    parser.add_argument("--glob", type=str, default="log.*", help="Glob pattern for log files (default: log.*).")
    parser.add_argument("--tasks", type=int, default=10, help="Max task columns to expand (default: 10).")
    parser.add_argument("--out_csv", type=str, default="extracted_accs.csv", help="Output CSV path.")
    parser.add_argument("--log", type=str, default=None, help="Optional run log path.")
    args = parser.parse_args()

    logger = setup_logger(args.log)

    root = Path(args.log_dir)
    if not root.exists():
        logger.error(f"Log directory does not exist: {root}")
        return

    # 递归遍历子目录（task_0...）匹配日志文件
    files = [p for p in root.rglob(args.glob) if p.is_file()]
    if not files:
        logger.error(f"No files matched '{args.glob}' under {root}")
        return

    files.sort(key=lambda p: p.stat().st_mtime)
    all_rows: List[Tuple[str, Dict, str]] = []

    for p in files:
        payload = extract_from_file(p, logger)
        task_name = get_task_name(p, root)
        all_rows.append((p.name, payload, task_name))

    write_combined_csv(all_rows, Path(args.out_csv), args.tasks, logger)

if __name__ == "__main__":
    main()