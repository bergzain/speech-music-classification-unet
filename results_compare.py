import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def get_run_metrics(run_path):
    metrics_path = os.path.join(run_path, 'metrics')
    metrics = {}
    
    if os.path.exists(metrics_path):
        for metric_file in os.listdir(metrics_path):
            with open(os.path.join(metrics_path, metric_file), 'r') as f:
                values = [float(line.split()[1]) for line in f.readlines()]
                metrics[metric_file] = values

    return metrics

def get_best_and_least_loss(metrics):
    best_accuracy = max(metrics.get('val_accuracy', [0]))
    least_loss = min(metrics.get('val_loss', [float('inf')]))
    return best_accuracy, least_loss

def get_run_name(run_path):
    tags_path = os.path.join(run_path, 'tags')
    run_name_file = os.path.join(tags_path, 'mlflow.runName')
    
    if os.path.exists(run_name_file):
        with open(run_name_file, 'r') as f:
            run_name = f.read().strip()
        return run_name
    return None

def main():
    # mlflow_dir = 'mlflow' # 
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
                    runs.append((run_name, best_accuracy, least_loss))

    # Create a DataFrame for better visualization
    df = pd.DataFrame(runs, columns=['Run Name', 'Best Accuracy', 'Least Loss'])
    
    # Save the DataFrame to an Excel file
    excel_file = 'run_metrics.xlsx'
    df.to_excel(excel_file, index=False)

    # Load the workbook and select the active worksheet
    wb = load_workbook(excel_file)
    ws = wb.active

    # Apply highlighting
    max_acc = df['Best Accuracy'].max()
    min_loss = df['Least Loss'].min()

    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=2, max_col=2):
        for cell in row:
            if cell.value == max_acc:
                cell.fill = yellow_fill

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=3, max_col=3):
        for cell in row:
            if cell.value == min_loss:
                cell.fill = green_fill

    # Save the workbook
    wb.save(excel_file)
    print(f"Metrics saved to {excel_file}")

if __name__ == "__main__":
    main()

