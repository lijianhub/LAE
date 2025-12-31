import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast  # To parse string-formatted arrays from CSV


# --------------------------
# 1. Load and preprocess data (fixed array parsing)
# --------------------------
def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and preprocess array columns (supports both string/float array elements)
    Fixed: Removed strip() for float elements to avoid AttributeError
    """
    try:
        # Read CSV with core columns (include array columns)
        df = pd.read_csv(
            csv_path,
            usecols=[
                "evaluation_id",
                "task_specific_global_acc",
                "task_id",
                "global_per_task_accs_array",  # Array for all tasks' global acc
                "local_per_task_accs_array"  # Array for all tasks' local acc
            ],
            dtype={
                "evaluation_id": int,
                "task_specific_global_acc": float,
                "task_id": int
            }
        )

        # --------------------------
        # Fixed: Safe array parsing (handles strings and floats)
        # --------------------------
        def safe_parse_array(array_str_or_list):
            """
            Convert CSV array data to list of floats:
            - If input is a string (e.g., "[85.6, 91.2]"), parse with ast
            - If input is already a list, directly convert elements to float
            - Skip strip() for non-string elements
            """
            # Case 1: Input is a string (common in CSV exports)
            if isinstance(array_str_or_list, str):
                # Remove extra quotes if present (e.g., '"[85.6, 91.2]"' → "[85.6, 91.2]")
                clean_str = array_str_or_list.strip().strip('"').strip("'")
                # Parse string to list
                parsed_list = ast.literal_eval(clean_str)
            # Case 2: Input is already a list (rare, but handles edge cases)
            elif isinstance(array_str_or_list, list):
                parsed_list = array_str_or_list
            # Case 3: Unexpected type (return empty list to avoid crash)
            else:
                print(f"⚠️ Unexpected array type: {type(array_str_or_list)}. Using empty list.")
                return []

            # Convert all elements to float (handles ints/floats in the list)
            float_list = []
            for elem in parsed_list:
                try:
                    # No strip() here (elem is float/int, not string)
                    float_list.append(float(elem))
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Skipping invalid element '{elem}': {str(e)}")
                    float_list.append(0.0)  # Replace invalid data with 0.0 (safe fallback)

            return float_list

        # Apply safe parsing to array columns
        df["global_per_task_accs_array"] = df["global_per_task_accs_array"].apply(safe_parse_array)
        df["local_per_task_accs_array"] = df["local_per_task_accs_array"].apply(safe_parse_array)

        # Filter out rows with empty arrays (avoid downstream errors)
        df = df[
            (df["global_per_task_accs_array"].apply(len) > 0) &
            (df["local_per_task_accs_array"].apply(len) > 0)
            ].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid data left after filtering empty arrays")

        # Sort by evaluation ID to ensure correct sequence
        df = df.sort_values("evaluation_id").reset_index(drop=True)

        # Print data verification info
        total_tasks = len(df["global_per_task_accs_array"].iloc[0])
        print(f"✅ Data loaded successfully: {len(df)} evaluation records")
        print(f"📌 Target Task ID: {df['task_id'].iloc[0]}")
        print(f"📊 Total tasks detected (from array): {total_tasks} (Task 0 to {total_tasks - 1})")
        print(f"🔄 Evaluation sequence: {df['evaluation_id'].min()} - {df['evaluation_id'].max()}")

        return df
    except Exception as e:
        print(f"❌ Data processing failed: {str(e)}")
        raise


# --------------------------
# 2. Plot 1: Original task accuracy trend
# --------------------------
def plot_single_task_trend(df: pd.DataFrame, save_path: str = "1_single_task_trend.png"):
    eval_ids = df["evaluation_id"].values
    accuracies = df["task_specific_global_acc"].values
    target_task_id = df["task_id"].iloc[0]

    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main curve
    ax.plot(
        eval_ids, accuracies,
        color="#2E86AB", linestyle="-", linewidth=2.5,
        marker="o", markersize=6, markerfacecolor="#A23B72",
        markeredgecolor="white", markeredgewidth=1,
        label=f"Target Task {target_task_id}"
    )

    # Auxiliary elements
    for eid in eval_ids:
        ax.axvline(x=eid, color="#D3D3D3", linestyle="--", alpha=0.6)
    ax.grid(True, axis="y", color="#F0F0F0", alpha=0.8)

    # Axis and title
    y_margin = (accuracies.max() - accuracies.min()) * 0.05 if len(accuracies) > 1 else 5.0
    ax.set_ylim(accuracies.min() - y_margin, accuracies.max() + y_margin)
    ax.set_xlim(eval_ids.min() - 0.5, eval_ids.max() + 0.5)
    ax.set_xlabel("Training Sequence (Evaluation ID)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(f"Accuracy Trend of Target Task {target_task_id}", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"📸 Plot 1 saved: {save_path}")


# --------------------------
# 3. Plot 2: All-task global accuracy comparison
# --------------------------
def plot_all_tasks_global(df: pd.DataFrame, save_path: str = "2_all_tasks_global_acc.png"):
    eval_ids = df["evaluation_id"].values
    total_tasks = len(df["global_per_task_accs_array"].iloc[0])
    target_task_id = df["task_id"].iloc[0]

    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(12, 7))

    # Distinct colors for tasks
    colors = plt.cm.Set3(np.linspace(0, 1, total_tasks))

    # Plot each task's global accuracy
    for task_idx in range(total_tasks):
        task_accs = [row[task_idx] for row in df["global_per_task_accs_array"].values]

        # Highlight target task
        if task_idx == target_task_id:
            ax.plot(
                eval_ids, task_accs,
                color="#E74C3C", linestyle="-", linewidth=3,
                marker="s", markersize=7, markerfacecolor="#E74C3C",
                label=f"Task {task_idx} (Target)", zorder=5
            )
        else:
            ax.plot(
                eval_ids, task_accs,
                color=colors[task_idx], linestyle="-", linewidth=1.8,
                marker="o", markersize=4, alpha=0.7,
                label=f"Task {task_idx}"
            )

    # Auxiliary elements
    for eid in eval_ids:
        ax.axvline(x=eid, color="#D3D3D3", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", color="#F0F0F0", alpha=0.8)

    # Axis and title
    ax.set_xlabel("Training Sequence (Evaluation ID)", fontweight="bold")
    ax.set_ylabel("Global Accuracy (%)", fontweight="bold")
    ax.set_title(f"Global Accuracy Comparison Across All Tasks (Total: {total_tasks})", fontweight="bold", fontsize=12)

    # Legend (avoid overlap)
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5),
        frameon=True, framealpha=0.9, ncol=2
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"📸 Plot 2 saved: {save_path}")


# --------------------------
# 4. Plot 3: All-task local accuracy comparison
# --------------------------
def plot_all_tasks_local(df: pd.DataFrame, save_path: str = "3_all_tasks_local_acc.png"):
    eval_ids = df["evaluation_id"].values
    total_tasks = len(df["local_per_task_accs_array"].iloc[0])
    target_task_id = df["task_id"].iloc[0]

    plt.rcParams.update({"font.size": 10, "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(12, 7))

    # Distinct colors for tasks (different from global plot)
    colors = plt.cm.Set2(np.linspace(0, 1, total_tasks))

    # Plot each task's local accuracy
    for task_idx in range(total_tasks):
        task_accs = [row[task_idx] for row in df["local_per_task_accs_array"].values]

        # Highlight target task
        if task_idx == target_task_id:
            ax.plot(
                eval_ids, task_accs,
                color="#27AE60", linestyle="-", linewidth=3,
                marker="s", markersize=7, markerfacecolor="#27AE60",
                label=f"Task {task_idx} (Target)", zorder=5
            )
        else:
            ax.plot(
                eval_ids, task_accs,
                color=colors[task_idx], linestyle="-", linewidth=1.8,
                marker="o", markersize=4, alpha=0.7,
                label=f"Task {task_idx}"
            )

    # Auxiliary elements
    for eid in eval_ids:
        ax.axvline(x=eid, color="#D3D3D3", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", color="#F0F0F0", alpha=0.8)

    # Axis and title
    ax.set_xlabel("Training Sequence (Evaluation ID)", fontweight="bold")
    ax.set_ylabel("Local Accuracy (%)", fontweight="bold")
    ax.set_title(f"Local Accuracy Comparison Across All Tasks (Total: {total_tasks})", fontweight="bold", fontsize=12)

    # Legend
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5),
        frameon=True, framealpha=0.9, ncol=2
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"📸 Plot 3 saved: {save_path}")


# --------------------------
# 5. Main execution flow
# --------------------------
if __name__ == "__main__":
    # Update this path to your CSV file location
    CSV_FILE_PATH = "C:/Users/40518/code/LAE/libml/utils/single_task_evaluation_data.csv"

    # Step 1: Load and preprocess data (fixed parsing)
    data_df = load_and_preprocess_data(CSV_FILE_PATH)

    # Step 2: Generate all three plots
    plot_single_task_trend(data_df)
    plot_all_tasks_global(data_df)
    plot_all_tasks_local(data_df)

    print("\n🎉 All three plots generated successfully!")