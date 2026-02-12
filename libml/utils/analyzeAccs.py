import os
import re
import csv
import logging
from pathlib import Path
import numpy as np
import time


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


# ===================== Helper Function: Check File Permission =====================
def is_file_writable(file_path: str) -> bool:
    """Check if file is writable (not occupied by other programs)"""
    if not os.path.exists(file_path):
        return True  # New file can be created

    # Try to open file in write mode to test permission
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            return True
    except PermissionError:
        return False
    except Exception as e:
        logger.warning(f"Check file writable failed: {str(e)}")
        return False


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


def calculate_phase_forgetting(acc_matrix: list) -> float:
    """Calculate per-phase average forgetting (your original logic: per training phase)"""
    valid_data = [(idx, acc) for idx, acc in enumerate(acc_matrix) if acc != '' and isinstance(acc, (int, float))]
    if len(valid_data) < 2:
        return 0.0

    task_indices, valid_accs = zip(*valid_data)
    forgetting_values = []

    for k_idx in range(1, len(valid_accs)):
        current_task = task_indices[k_idx]
        current_acc = valid_accs[k_idx]

        historical_accs = []
        for l_idx in range(k_idx):
            historical_task = task_indices[l_idx]
            if historical_task < current_task:
                historical_accs.append(valid_accs[l_idx])

        if not historical_accs:
            continue

        max_historical = max(historical_accs)
        f_kj = max_historical - current_acc
        if f_kj >= 0:
            forgetting_values.append(f_kj)

    if len(forgetting_values) == 0:
        return 0.0
    return round(sum(forgetting_values) / len(forgetting_values), 2)


def calculate_global_final_forgetting(all_acc_data: dict, num_tasks: int) -> float:
    """Calculate global final forgetting (using the provided formula: after all tasks)"""
    if num_tasks < 2:
        logger.info("Global Final Forgetting: Less than 2 tasks → 0.0")
        return 0.0

    # Step 1: Get final phase data (last row: after learning all tasks)
    final_phase_key = list(all_acc_data.keys())[-1]
    final_phase_acc = all_acc_data[final_phase_key]

    # Step 2: Collect A_i^max (best accuracy of each task across all phases)
    task_max_acc = {}
    for phase_acc in all_acc_data.values():
        for task_idx, acc in enumerate(phase_acc):
            if acc == '' or not isinstance(acc, (int, float)):
                continue
            if task_idx not in task_max_acc or acc > task_max_acc[task_idx]:
                task_max_acc[task_idx] = acc

    # Step 3: Calculate forgetting for each task (i=1 to T-1)
    forgetting_sum = 0.0
    valid_task_count = 0
    for task_idx in range(1, num_tasks):  # i=1 to T-1 (per formula)
        if task_idx not in task_max_acc:
            continue
        A_i_max = task_max_acc[task_idx]
        A_i_T = final_phase_acc[task_idx] if (
                    task_idx < len(final_phase_acc) and final_phase_acc[task_idx] != '') else 0.0

        if not isinstance(A_i_T, (int, float)):
            continue

        forgetting = A_i_max - A_i_T
        if forgetting >= 0:
            forgetting_sum += forgetting
            valid_task_count += 1

    # Step 4: Compute average (per formula: 1/(T-1))
    if valid_task_count == 0 or (num_tasks - 1) == 0:
        return 0.0
    global_forget = forgetting_sum / (num_tasks - 1)
    logger.info(f"Global Final Forgetting: Calculated as {global_forget:.2f}")
    return round(global_forget, 2)


# ===================== CSV Writing (Add Global Final Forgetting Row) =====================
def write_file_wise_csv(results: dict, csv_path: str, field_prefix: str):
    """Write 1 row per file + last row: Global Final Forgetting"""
    if not results:
        logger.warning(f"No data to write to {csv_path}")
        return

    # Check file permission
    if not is_file_writable(csv_path):
        logger.error(f"❌ Permission denied: {csv_path} is occupied")
        return

    # Step 1: Determine max task columns & num tasks
    max_task_cols = max(len(arr) for arr in results.values())
    num_tasks = max_task_cols  # Total tasks = max task columns

    # Step 2: Build headers
    task_headers = [f"{field_prefix}Task_{i}" for i in range(max_task_cols)]
    headers = ['log_filename'] + task_headers + ['Average_Forgetting']

    # Step 3: Calculate global final forgetting (using the formula)
    global_final_forget = calculate_global_final_forgetting(results, num_tasks)

    # Step 4: Write CSV
    retry_count = 3
    for retry in range(retry_count):
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                # Write per-phase rows (original data)
                for file_name, avg_array in results.items():
                    padded_acc = avg_array + [''] * (max_task_cols - len(avg_array))
                    phase_forget = calculate_phase_forgetting(padded_acc)

                    row = {'log_filename': file_name}
                    for i, val in enumerate(padded_acc):
                        row[f"{field_prefix}Task_{i}"] = val
                    row['Average_Forgetting'] = phase_forget
                    writer.writerow(row)

                # Write LAST ROW: Global Final Forgetting
                global_row = {
                    'log_filename': 'Global_Final_Forgetting',
                    'Average_Forgetting': global_final_forget
                }
                # Fill task columns with empty (only last column has value)
                for i in range(max_task_cols):
                    global_row[f"{field_prefix}Task_{i}"] = ''
                writer.writerow(global_row)

            logger.info(f"\n✅ Successfully wrote {len(results) + 1} rows to {csv_path}")
            logger.info(f"   Last row: Global Final Forgetting = {global_final_forget}")
            return
        except PermissionError as e:
            if retry < retry_count - 1:
                logger.warning(f"⚠️ Retry {retry + 1}/{retry_count}: {str(e)}")
                time.sleep(1)
            else:
                logger.error(f"❌ Failed to write {csv_path}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error writing {csv_path}: {str(e)}")
            return


# ===================== Main Workflow =====================
def main(log_dir: str):
    logger.info("=== START: Acc Calculation + Forgetting Metrics ===")

    # Get log files
    log_files = [f for f in Path(log_dir).rglob("log.lj.*") if f.is_file()]
    if not log_files:
        logger.error(f"No log files in {log_dir}")
        return
    logger.info(f"Found {len(log_files)} log files\n")

    # Process Global Accs
    global_results = {}
    logger.info("----- Processing Global Per Task Accs -----")
    for file in log_files:
        arrays = extract_arrays_from_file(file, "Global Per Task Accs")
        global_results[file.name] = calculate_file_element_avg(arrays, file.name, "Global Per Task Accs")

    # Process Local Accs
    local_results = {}
    logger.info("\n----- Processing Local Per Task Accs -----")
    for file in log_files:
        arrays = extract_arrays_from_file(file, "Local Per Task Accs")
        local_results[file.name] = calculate_file_element_avg(arrays, file.name, "Local Per Task Accs")

    # Write CSVs (with global final forgetting row)
    write_file_wise_csv(global_results, "global_per_task_accs.csv", "Global_")
    write_file_wise_csv(local_results, "local_per_task_accs.csv", "Local_")

    logger.info("=== END: All Calculations Completed ===")


if __name__ == "__main__":
    LOG_DIRECTORY = r"C:\Users\40518\code\LAE\logs\vit_lora\version_0"
    main(LOG_DIRECTORY)