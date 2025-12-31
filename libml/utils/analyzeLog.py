import re
import csv
import os  # Add os module for path operations
from typing import List, Dict, Optional


# --------------------------
# Core Data Extraction Functions (unchanged)
# --------------------------
def extract_single_task_data(log_file_path: str) -> Optional[Dict[str, List[Dict]]]:
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Log file not found at '{log_file_path}'")
        return None
    except PermissionError:
        print(f"❌ Error: No permission to read log file at '{log_file_path}'")
        return None

    eval_pattern = r'==> Evaluation result (\d+)\s+' \
                   r'Acc: (\d+\.\d+)\s+' \
                   r'Global Per Task Accs: ([\d+\.\d+, ]+)\s+' \
                   r'Global Task Accs Avg: (\d+\.\d+)\s+' \
                   r'Local Per Task Accs: ([\d+\.\d+, ]+)'

    eval_matches = re.findall(eval_pattern, log_content, re.DOTALL)
    if not eval_matches:
        print("❌ No evaluation results found in the log file")
        return None

    task_data = {}
    task_id = None

    for match in eval_matches:
        eval_id = int(match[0])
        overall_acc = float(match[1])
        global_per_task_str = match[2].strip()
        global_avg = float(match[3])
        local_per_task_str = match[4].strip()

        global_per_task_accs = [float(x.strip()) for x in global_per_task_str.split(',') if x.strip()]
        local_per_task_accs = [float(x.strip()) for x in local_per_task_str.split(',') if x.strip()]

        num_tasks = len(global_per_task_accs)
        if len(local_per_task_accs) != num_tasks:
            print(
                f"⚠️ Mismatched task counts in Evaluation {eval_id} (Global: {num_tasks}, Local: {len(local_per_task_accs)})")
            continue

        if task_id is None:
            task_id = 0 if num_tasks == 1 else num_tasks - 1
            task_data[task_id] = []

        result = {
            'evaluation_id': eval_id,
            'task_id': task_id,
            'overall_acc': overall_acc,
            'global_task_accs_avg': global_avg,
            'global_per_task_accs_array': global_per_task_accs,
            'local_per_task_accs_array': local_per_task_accs,
            'task_specific_global_acc': global_per_task_accs[task_id],
            'task_specific_local_acc': local_per_task_accs[task_id],
            'total_tasks_in_log': num_tasks
        }
        task_data[task_id].append(result)

    return task_data if task_data else None


def save_task_data_to_csv(task_data: Dict[str, List[Dict]], csv_file_path: str):
    flattened_data = []
    for task_id, results in task_data.items():
        flattened_data.extend(results)

    if not flattened_data:
        print("❌ No data to save")
        return

    # Create parent directory if it doesn't exist
    csv_dir = os.path.dirname(csv_file_path)
    if csv_dir and not os.path.exists(csv_dir):
        try:
            os.makedirs(csv_dir, exist_ok=True)  # exist_ok=True avoids error if dir exists
            print(f"✅ Created parent directory: '{csv_dir}'")
        except PermissionError:
            print(f"❌ Error: No permission to create directory '{csv_dir}'")
            return

    # Save CSV
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            headers = flattened_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flattened_data)
        print(f"✅ CSV saved successfully to: '{csv_file_path}'")
    except PermissionError:
        print(f"❌ Error: No permission to write CSV file at '{csv_file_path}'")
    except Exception as e:
        print(f"❌ Failed to save CSV: {str(e)}")


def print_single_task_summary(task_data: Dict[str, List[Dict]]):
    task_id = next(iter(task_data.keys()))
    results = task_data[task_id]
    num_evaluations = len(results)

    print(f"\n=== Single-Task Data Summary ===")
    print(f"Task ID detected: {task_id}")
    print(f"Number of evaluations: {num_evaluations}")
    print(f"Total tasks in log: {results[0]['total_tasks_in_log']}")
    print(f"\nTask Performance Across Evaluations:")
    for result in results:
        print(f"Evaluation {result['evaluation_id']}:")
        print(
            f"  - Global Acc: {result['task_specific_global_acc']:.2f} | Local Acc: {result['task_specific_local_acc']:.2f}")
        print(f"  - Overall Acc: {result['overall_acc']:.2f} | Global Avg: {result['global_task_accs_avg']:.2f}")
        print("-" * 50)


# --------------------------
# Main Function with Fixed Path Handling
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # Fix 1: Use Absolute Paths (Avoid Relative Path Confusion)
    # --------------------------
    # Replace with YOUR ACTUAL absolute paths (find via: right-click file → "Copy Path" → paste here)
    # Example Windows absolute path format: "C:/Users/40518/code/LAE/logs/cifar100/version_3/task_9/log.lj.40518.log.INFO.20251231-164719.28296"
    # Example Linux/macOS absolute path format: "/home/40518/code/LAE/logs/cifar100/version_3/task_9/log.lj.40518.log.INFO.20251231-164719.28296"
    LOG_FILE_PATH = "C:/Users/40518/code/LAE/logs/cifar100/version_3/task_9/log.lj.40518.log.INFO.20251231-164719.28296"  # UPDATE THIS
    CSV_SAVE_PATH = "C:/Users/40518/code/LAE/results/task9.csv"  # UPDATE THIS

    # --------------------------
    # Fix 2: Validate Log File Path Before Extraction
    # --------------------------
    print(f"🔍 Checking log file path: '{LOG_FILE_PATH}'")
    if not os.path.exists(LOG_FILE_PATH):
        print(f"❌ Fatal Error: Log file does not exist. Please check the path.")
        print(f"💡 Tip: Use absolute path (right-click log file → 'Copy Path' → paste into LOG_FILE_PATH)")
        exit(1)  # Exit if log file is missing (critical error)
    if not os.path.isfile(LOG_FILE_PATH):
        print(f"❌ Fatal Error: '{LOG_FILE_PATH}' is not a file (it's a directory). Check the path.")
        exit(1)

    # --------------------------
    # Extract and Save Data
    # --------------------------
    print("📥 Extracting data from single-task log file...")
    single_task_data = extract_single_task_data(LOG_FILE_PATH)

    if single_task_data:
        print_single_task_summary(single_task_data)
        print(f"\n💾 Saving data to CSV: '{CSV_SAVE_PATH}'")
        save_task_data_to_csv(single_task_data, CSV_SAVE_PATH)
    else:
        print("❌ Data extraction failed (see errors above for details)")