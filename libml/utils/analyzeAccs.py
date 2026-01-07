import os
import re
import csv
import logging
from pathlib import Path
import numpy as np


# ===================== Log Configuration =====================
def setup_logger(log_path: str = "acc_calculation.log"):
    """Setup logger for calculation verification (simple format)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_path, 'w', 'utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# ===================== Core Extraction & Calculation =====================
def extract_arrays_from_file(file_path: Path, target_field: str) -> list:
    """Extract all arrays of target field (Global/Local) from a single log file"""
    arrays = []
    pattern = re.compile(re.escape(target_field) + r':\s*([\d\.]+(?:, [\d\.]+)*)')

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    # Convert string values to float array
                    values = [float(v.strip()) for v in match.group(1).split(',') if
                              v.strip().replace('.', '').isdigit()]
                    arrays.append(values)
                    logger.info(f"[{file_path.name}] {target_field} Line {line_num}: Extracted {values}")
    except Exception as e:
        logger.error(f"[{file_path.name}] Error reading file: {str(e)}")

    logger.info(f"[{file_path.name}] {target_field} - Total arrays extracted: {len(arrays)}")
    return arrays


def calculate_file_element_avg(arrays: list, file_name: str, field_name: str) -> list:
    """Calculate element-wise average for all arrays in a single file (1 array per file)"""
    if not arrays:
        logger.warning(f"[{file_name}] {field_name} - No valid arrays → empty result")
        return []

    # Step 1: Pad all arrays in the file to the same length (max length in file)
    max_len = max(len(arr) for arr in arrays)
    padded_arrays = [arr + [np.nan] * (max_len - len(arr)) for arr in arrays]

    # Step 2: Calculate element-wise average (ignore NaN)
    avg_array = np.nanmean(np.array(padded_arrays), axis=0)
    # Round to 2 decimals, replace NaN with empty string
    final_avg = [round(val, 2) if not np.isnan(val) else '' for val in avg_array]

    # Log verification details
    logger.info(f"[{file_name}] {field_name} - File-level element-wise average: {final_avg}")
    return final_avg


# ===================== CSV Writing =====================
def write_file_wise_csv(results: dict, csv_path: str, field_prefix: str):
    """Write 1 row per file to CSV (aligned columns across all files)"""
    if not results:
        logger.warning(f"No data to write to {csv_path}")
        return

    # Step 1: Find max column count across all files (align columns)
    max_cols = max(len(arr) for arr in results.values())

    # Step 2: Build headers (log_filename + Task_0, Task_1, ...)
    headers = ['log_filename'] + [f"{field_prefix}Task_{i}" for i in range(max_cols)]

    # Step 3: Write CSV (1 row per file)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for file_name, avg_array in results.items():
            # Pad array to max_cols with empty strings
            padded_avg = avg_array + [''] * (max_cols - len(avg_array))
            row = {'log_filename': file_name}
            for i, val in enumerate(padded_avg):
                row[f"{field_prefix}Task_{i}"] = val
            writer.writerow(row)

    logger.info(f"\n✅ Successfully wrote {len(results)} rows (1 per file) to {csv_path}")
    logger.info(f"   Columns: log_filename + {max_cols} task columns\n")


# ===================== Main Workflow =====================
def main(log_dir: str):
    logger.info("=== START: File-Level Acc Calculation (1 Row per File) ===\n")

    # Step 1: Get all log files (1 entry per file)
    log_files = [f for f in Path(log_dir).rglob("log.lj.*") if f.is_file()]
    if not log_files:
        logger.error(f"No log files found in {log_dir}")
        return
    logger.info(f"Found {len(log_files)} log files (1 row per file in output)\n")

    # Step 2: Process Global Per Task Accs (file-level average)
    global_results = {}
    logger.info("----- Processing Global Per Task Accs -----")
    for file in log_files:
        arrays = extract_arrays_from_file(file, "Global Per Task Accs")
        global_results[file.name] = calculate_file_element_avg(arrays, file.name, "Global Per Task Accs")

    # Step 3: Process Local Per Task Accs (file-level average)
    local_results = {}
    logger.info("\n----- Processing Local Per Task Accs -----")
    for file in log_files:
        arrays = extract_arrays_from_file(file, "Local Per Task Accs")
        local_results[file.name] = calculate_file_element_avg(arrays, file.name, "Local Per Task Accs")

    # Step 4: Write to CSV (1 row per file)
    write_file_wise_csv(global_results, "global_per_task_accs.csv", "Global_")
    write_file_wise_csv(local_results, "local_per_task_accs.csv", "Local_")

    logger.info("=== END: All Calculations Completed ===")
    logger.info(f"Global results: global_per_task_accs.csv (1 row per file)")
    logger.info(f"Local results: local_per_task_accs.csv (1 row per file)")


if __name__ == "__main__":
    # --------------------------
    # ONLY CONFIGURE THIS LINE!
    # --------------------------
    LOG_DIRECTORY = r"C:\Users\40518\code\LAE\logs\cifar100\version_0"  # Your log folder path

    main(LOG_DIRECTORY)