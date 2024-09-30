import os
import re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt

def get_run_metrics(run_path):
    """Extract metrics from the given run path."""
    metrics_path = os.path.join(run_path, 'metrics')
    metrics = {}
    
    if os.path.exists(metrics_path):
        for metric_file in os.listdir(metrics_path):
            with open(os.path.join(metrics_path, metric_file), 'r') as f:
                values = [float(line.split()[1]) for line in f.readlines()]
                metrics[metric_file] = values

    return metrics

def get_best_and_least_loss(metrics):
    """Get the best accuracy and least loss from the metrics."""
    best_accuracy = max(metrics.get('val_accuracy', [0]))
    least_loss = min(metrics.get('val_loss', [float('inf')]))
    return best_accuracy, least_loss

def get_run_name(run_path):
    """Retrieve the run name from the given run path."""
    tags_path = os.path.join(run_path, 'tags')
    run_name_file = os.path.join(tags_path, 'mlflow.runName')
    
    if os.path.exists(run_name_file):
        with open(run_name_file, 'r') as f:
            run_name = f.read().strip()
        return run_name
    return None

def parse_run_name(run_name):
    """Parse the run name to extract model details."""
    # Use regular expression to extract the parts of the run name
    match = re.match(r'([A-Za-z0-9_]+)_([A-Za-z\-]+)_([0-9]+)_len([0-9]+)S', run_name)
    if match:
        model_name = match.group(1)
        transformation_type = match.group(2)
        num_features = int(match.group(3))
        length_in_seconds = int(match.group(4))
        return model_name, transformation_type, num_features, length_in_seconds
    else:
        print(f"Run name format is incorrect: {run_name}")
        raise ValueError(f"Run name format is incorrect: {run_name}")

def get_model_parameters(model_name):
    """Get the number of parameters for the given model."""
    model_sizes = {
        "U_Net": 34526084,
        "AttentionUNet": 34877486,
        "R2U_Net": 39091460,
        "R2AttU_Net": 39442992
    }
    return model_sizes.get(model_name, 0)

def highlight_cells(ws, df, max_acc, min_loss):
    """Highlight cells in the Excel sheet based on conditions."""
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=7, max_col=7):
        for cell in row:
            if cell.value == max_acc:
                cell.fill = yellow_fill

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=8, max_col=8):
        for cell in row:
            if cell.value == min_loss:
                cell.fill = green_fill

def plot_model_performance(df):
    """Plot the model performance with different shapes for models and colors for transformations."""
    shapes = {
        "U_Net": "o",
        "AttentionUNet": "s",
        "R2U_Net": "D",
        "R2AttU_Net": "^"
    }
    
    colors = {
        "MFCC": "b",
        "LFCC": "g",
        "delta": "r",
        "delta-delta": "c",
        "lfcc-delta": "m",
        "lfcc-delta-delta": "y"
    }
    
    plt.figure(figsize=(10, 6))
    
    for _, row in df.iterrows():
        model_name = row['Model Name']
        transformation_type = row['Transformation Type']
        accuracy = row['Best Accuracy']
        loss = row['Least Loss']
        
        shape = shapes.get(model_name, "o")
        color = colors.get(transformation_type, "k")
        
        plt.scatter(loss, accuracy, marker=shape, color=color, s=100, label=f"{model_name}-{transformation_type}")
    
    # Create a legend with unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.xlabel('Least Loss')
    plt.ylabel('Best Accuracy')
    plt.title('Model Performance')
    plt.grid(True)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()


def main():
    mlflow_dir = "/Users/zainhazzouri/projects/Master-thesis-experiments/mlflow/"
    runs = []

    for root, dirs, files in os.walk(mlflow_dir):
        for dir_name in dirs:
            run_path = os.path.join(root, dir_name)
            if 'metrics' in os.listdir(run_path):
                run_name = get_run_name(run_path)
                if run_name:
                    metrics = get_run_metrics(run_path)
                    best_accuracy, least_loss = get_best_and_least_loss(metrics)
                    if best_accuracy > 0 and least_loss < float('inf'):
                        model_name, transformation_type, num_features, length_in_seconds = parse_run_name(run_name)
                        num_parameters = get_model_parameters(model_name)
                        runs.append((run_name, model_name, transformation_type, num_features, length_in_seconds, num_parameters, best_accuracy, least_loss))

    # Create a DataFrame for better visualization
    df = pd.DataFrame(runs, columns=['Run Name', 'Model Name', 'Transformation Type', 'Number of Features', 'Length of Chunks (s)', 'Number of Parameters', 'Best Accuracy', 'Least Loss'])
    
    # Save the DataFrame to an Excel file
    excel_file = 'run_metrics.xlsx'
    df.to_excel(excel_file, index=False)

    # Load the workbook and select the active worksheet
    wb = load_workbook(excel_file)
    ws = wb.active

    # Apply highlighting
    max_acc = df['Best Accuracy'].max()
    min_loss = df['Least Loss'].min()
    highlight_cells(ws, df, max_acc, min_loss)

    # Save the workbook
    wb.save(excel_file)
    print(f"Metrics saved to {excel_file}")

    # Plot the model performance
    plot_model_performance(df)

if __name__ == "__main__":
    main()
