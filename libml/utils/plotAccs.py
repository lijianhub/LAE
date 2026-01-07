import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_task_accuracy_trend(
        csv_file_path: str,
        output_image_path: str = "final_accuracy_plot.png",
        y_axis_label: str = "Accuracy (%)",
        x_axis_label: str = "Task Sequence",
        y_axis_range: tuple = (60, 100),
        figure_size: tuple = (12, 6)
):
    """
    Matches reference plot logic:
    - X-axis = Task sequence (Task 0-Task 9)
    - Y-axis = Accuracy percentage
    - One line per log file (distinguished by unique colors)
    - Skip empty values, no 0-value display
    """
    # 1. Load data
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded CSV: {csv_file_path} (Number of log files: {len(df)}, Number of tasks: {len(df.columns) - 1})")
    except Exception as e:
        print(f"❌ Load failed: {str(e)}")
        return

    # Extract task columns (Global_Task_0 to Global_Task_9)
    task_columns = [col for col in df.columns if col.startswith(("Global_Task_", "Local_Task_"))]
    if not task_columns:
        print(f"❌ No task columns found (Global_Task_*/Local_Task_*)")
        return
    if "log_filename" not in df.columns:
        print(f"❌ Missing 'log_filename' column")
        return

    # 2. Prepare color palette (unique color per log file)
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    exclude_colors = ['white', 'whitesmoke', 'lightgray', 'gray']
    colors = [c for c in colors if c not in exclude_colors]

    # 3. Prepare plot data
    x_ticks = [f"Task {i}" for i in range(len(task_columns))]  # X-axis: Task 0-Task 9
    legend_labels = [f"Task {i}" for i in range(len(task_columns))]  # Log files as legend labels

    # Process data for each log file (skip empty values)
    plot_data = []
    for _, row in df.iterrows():
        # Convert to numeric values, empty values → NaN (not displayed)
        file_vals = pd.to_numeric(row[task_columns], errors='coerce')
        plot_data.append(file_vals.values)

    # 4. Plot (X-axis = tasks, Y-axis = accuracy, one line per log file)
    plt.figure(figsize=figure_size)
    for idx, (data, label) in enumerate(zip(plot_data, legend_labels)):
        # Keep only non-empty values
        valid_mask = ~np.isnan(data)
        valid_x = [x_ticks[i] for i in range(len(data)) if valid_mask[i]]
        valid_y = data[valid_mask]

        if len(valid_y) > 0:
            plt.plot(
                valid_x, valid_y,
                linestyle="-",
                color=colors[idx % len(colors)],
                marker="o",
                linewidth=1.5,
                markersize=6,
                label=label
            )

    # 5. Plot formatting (match reference plot)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.ylim(y_axis_range)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Add vertical dashed lines between tasks (match reference plot)
    for i in range(len(x_ticks)):
        plt.axvline(x=i, color="lightgray", linestyle="--", linewidth=0.5)
    plt.grid(True, linestyle="--", alpha=0.2)

    # Legend (placed outside plot on the right)
    plt.legend(
        title="Global Task Acc",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        frameon=True
    )

    # Adjust layout to prevent legend cutoff
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space for legend

    # 6. Save and display
    try:
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to: {output_image_path}")
    except Exception as e:
        print(f"❌ Save failed: {str(e)}")
    plt.show()


def main():
    # Configuration (modify only this section)
    CSV_PATH = "global_per_task_accs.csv"  # Your CSV file path
    OUTPUT_PATH = "global_per_task_accs.png"  # Output image path
    Y_AXIS_RANGE = (60, 100)  # Accuracy range (adjust based on your data)

    plot_task_accuracy_trend(
        csv_file_path=CSV_PATH,
        output_image_path=OUTPUT_PATH,
        y_axis_range=Y_AXIS_RANGE,
        figure_size=(14, 7)
    )


if __name__ == "__main__":
    main()