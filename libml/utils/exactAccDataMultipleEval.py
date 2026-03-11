# exactAccData.py
# -----------------------------------------------------------------------------
# 目标：
# 1) 抽取 pre-sweep 段（首次出现 [SWEEP ...] 之前）最后一次
#    "==> Evaluation results X" 的 Global Per Task Accs（数组）+ eval 头部行。
# 2) 抽取每一组 [SWEEP i/N] 的 sweep_line（自动合并多行）、eval 头部行与 Global Per Task Accs。
# 3) 导出一个合并 CSV：第一列改为 task_name（如 task_0），而不是 log_filename。
#    列：task_name, kind, sweep_idx, sweep_total, sweep_line, eval_line, accs(json),
#        Global_Task_0..Global_Task_{tasks-1}
# -----------------------------------------------------------------------------
"""
python exactAccDataMultipleEval.py `
  --log_dir "C:/Users/40518/code/LAE/logs/cifar100/version_9" `
  --glob "log.lj.*" `
  --tasks 10 `
  --out_csv "global_per_task_accs_cifar100_9_dedup.csv"
"""
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
    """
    从 file_path 相对 root 的路径里，提取第一个 'task_数字' 目录名；
    若不存在，则用上级目录名；再不行用文件 stem。
    """
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
TS_LINE = re.compile(r"^[IWEF]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{6}\s+\d+\s+\S+:\d+\]")
PAT_EVAL_HEAD = re.compile(r"==>\s*Evaluation results\s*(\d+)", re.IGNORECASE)
PAT_SWEEP_HEAD = re.compile(r"==>\s*\[SWEEP\s*(\d+)\s*/\s*(\d+)\]\s*(.*)$", re.IGNORECASE)

FIELD = "Global Per Task Accs"
PAT_GLOBAL_ARRAY = re.compile(
    rf"{re.escape(FIELD)}\s*:\s*(\[[^\]]*\]|[^\n\r]+)$", re.IGNORECASE
)

def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    vals: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals

def is_ts_line(line: str) -> bool:
    return bool(TS_LINE.match(line))

# --------------------- Core extraction per file ---------------------
def extract_from_file(
    file_path: Path,
    logger: logging.Logger,
    sweep_joiner: str = "|",
    tasks_cap: int = 50,
) -> Dict[str, List[Dict]]:
    base_last: Optional[Dict] = None
    sweeps: List[Dict] = []

    sweep_started = False
    pending_base_eval_line: Optional[str] = None
    pending_sweep: Optional[Dict] = None

    def collect_sweep_block(first_line: str, lines_iter) -> str:
        parts = [first_line.strip()]
        for peek in lines_iter:
            if is_ts_line(peek):
                buffer.append(peek)
                break
            parts.append(peek.strip())
        return sweep_joiner.join([p for p in parts if p])

    buffer: List[str] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        def line_generator():
            for raw in f:
                if buffer:
                    yield buffer.pop(0)
                yield raw

        lines = line_generator()
        for line in lines:
            m_sw = PAT_SWEEP_HEAD.search(line)
            if m_sw:
                sweep_started = True
                idx = int(m_sw.group(1)); total = int(m_sw.group(2))
                first_line = line.strip()
                sweep_line = collect_sweep_block(first_line, lines)
                pending_sweep = {"idx": idx, "total": total, "sweep_line": sweep_line}
                pending_base_eval_line = None
                logger.info(f"[{file_path.name}] SWEEP {idx}/{total}: {sweep_line}")
                continue

            m_ev = PAT_EVAL_HEAD.search(line)
            if m_ev:
                if pending_sweep is not None and "eval_line" not in pending_sweep:
                    pending_sweep["eval_line"] = line.strip()
                elif not sweep_started:
                    pending_base_eval_line = line.strip()
                continue

            m_ga = PAT_GLOBAL_ARRAY.search(line)
            if m_ga:
                arr = parse_float_list(m_ga.group(1))
                if pending_sweep is not None and "eval_line" in pending_sweep and "accs" not in pending_sweep:
                    pending_sweep["accs"] = arr
                    sweeps.append(pending_sweep)
                    logger.info(f"[{file_path.name}] SWEEP {pending_sweep['idx']}/{pending_sweep['total']} accs={arr}")
                    pending_sweep = None
                elif not sweep_started and pending_base_eval_line:
                    base_last = {"eval_line": pending_base_eval_line, "accs": arr}
                    logger.info(f"[{file_path.name}] BASE-LAST accs={arr} (eval_line={pending_base_eval_line})")
                    pending_base_eval_line = None
                continue

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
        "task_name", "kind", "sweep_idx", "sweep_total", "sweep_line", "eval_line", "accs"
    ] + [f"Global_Task_{i}" for i in range(tasks)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for log_name, payload, task_name in all_rows:
            for base in payload.get("base_last", []):
                row = {
                    "task_name": task_name,
                    "kind": "base_last",
                    "sweep_idx": "",
                    "sweep_total": "",
                    "sweep_line": "",
                    "eval_line": base.get("eval_line", ""),
                    "accs": json.dumps(base.get("accs", []), ensure_ascii=False),
                }
                arr = base.get("accs", [])
                for i in range(tasks):
                    row[f"Global_Task_{i}"] = (round(float(arr[i]), 4) if i < len(arr) else "")
                writer.writerow(row)

            for sw in payload.get("sweeps", []):
                row = {
                    "task_name": task_name,
                    "kind": "sweep",
                    "sweep_idx": sw.get("idx", ""),
                    "sweep_total": sw.get("total", ""),
                    "sweep_line": sw.get("sweep_line", ""),
                    "eval_line": sw.get("eval_line", ""),
                    "accs": json.dumps(sw.get("accs", []), ensure_ascii=False),
                }
                arr = sw.get("accs", [])
                for i in range(tasks):
                    row[f"Global_Task_{i}"] = (round(float(arr[i]), 4) if i < len(arr) else "")
                writer.writerow(row)

    logger.info(f"✅ Wrote CSV: {out_csv}")

# --------------------- CLI ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract 'Global Per Task Accs' for last pre-sweep eval and all sweeps, export CSV with task_name."
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Root directory or file directory containing logs.")
    parser.add_argument("--glob", type=str, default="log.*", help="Glob pattern for log files (default: log.*).")
    parser.add_argument("--tasks", type=int, default=10, help="Max task columns to expand (default: 10).")
    parser.add_argument("--out_csv", type=str, default="extracted_accs.csv", help="Output CSV path.")
    parser.add_argument("--log", type=str, default=None, help="Optional run log path.")
    parser.add_argument("--sweep_joiner", type=str, default="|", help="Joiner for multi-line SWEEP params.")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    root = Path(args.log_dir)
    if not root.exists():
        logger.error(f"Log directory does not exist: {root}")
        return

    files = [p for p in root.rglob(args.glob) if p.is_file()]
    if not files:
        logger.error(f"No files matched '{args.glob}' under {root}")
        return
    files.sort(key=lambda p: p.stat().st_mtime)

    all_rows: List[Tuple[str, Dict, str]] = []
    for p in files:
        payload = extract_from_file(p, logger, sweep_joiner=args.sweep_joiner, tasks_cap=args.tasks)
        task_name = get_task_name(p, root)
        all_rows.append((p.name, payload, task_name))

    write_combined_csv(all_rows, Path(args.out_csv), args.tasks, logger)

if __name__ == "__main__":
    main()