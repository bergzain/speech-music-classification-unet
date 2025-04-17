import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.
    
    Parameters:
    - file_path (Path): Path to the CSV file.
    
    Returns:
    - DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV file loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for analysis.
    
    Parameters:
    - df (DataFrame): Raw data from CSV.
    
    Returns:
    - DataFrame: Processed data.
    """
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['Length of Chunks (s)'] = df['length_in_seconds']
    df['Number of Parameters'] = df['model_type'].map({
        "U_Net": 34526084,
        "AttentionUNet": 34877486,
        "R2U_Net": 39091460,
        "R2AttU_Net": 39442992
    }).fillna(0).astype(int)
    df['Best Accuracy'] = df['best_val_accuracy']
    df['Least Loss'] = df['val_loss']
    df = df.sort_values(by='Start Time')
    return df





def plot_ball_chart(df: pd.DataFrame, output_dir: Path):
    """Plot a bubble chart for model performance."""
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Filter the DataFrame for runs with 5 seconds chunk length
    df = df[(df['length_in_seconds'] == 5) & (df['n_mfcc'] == 32)]

    transformation_types = df['feature_type'].unique()
    colors = sns.color_palette("hsv", len(transformation_types))
    color_dict = dict(zip(transformation_types, colors))

    model_names = df['model_type'].unique()
    markers = ['o', 's', 'D', '^']
    marker_dict = dict(zip(model_names, markers))

    # Create model labels with parameter counts
    model_params = {
        "U_Net": "34.5M",
        "AttentionUNet": "34.9M",
        "R2U_Net": "39.1M",
        "R2AttU_Net": "39.4M"
    }
    model_labels = [f"{model} ({model_params[model]})" for model in model_names]

    min_size = 50
    max_size = 300
    sizes = df['Number of Parameters']
    sizes_scaled = (sizes - sizes.min()) / (sizes.max() - sizes.min())
    sizes_scaled = sizes_scaled * (max_size - min_size) + min_size

    for idx, row in df.iterrows():
        plt.scatter(
            row['Least Loss'], row['Best Accuracy'], s=sizes_scaled[idx],
            c=[color_dict[row['feature_type']]], marker=marker_dict[row['model_type']],
            edgecolors='k', alpha=0.7
        )

    legend1 = [Line2D([0], [0], marker=marker_dict[model], label=f"{model} ({model_params[model]})", linestyle='None') 
               for model in model_names]
    legend2 = [Patch(color=color_dict[trans], label=trans) for trans in transformation_types]
    plt.legend(handles=legend1 + legend2, title='Legend', loc='lower right')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.title('Model Performance (chunk length = 5 secs, number of Feature = 32)')
    plt.tight_layout()
    plt.savefig(output_dir / 'bubble_chart_5_secs.png')
    plt.close()

def correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Perform correlation analysis and save the heatmap.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the heatmap.
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = df[['Best Accuracy', 'Least Loss', 'Number of Parameters', 'total_training_time']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

def accuracy_vs_training_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot Accuracy vs. Training Time.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='total_training_time', y='Best Accuracy', hue='feature_type', style='model_type', s=100)
    plt.title('Best Accuracy vs. Total Training Time')
    plt.xlabel('Total Training Time (s)')
    plt.ylabel('Best Accuracy')
    plt.legend(title='Feature & Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_training_time.png')
    plt.close()

def distribution_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generate distribution plots for key metrics.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plots.
    """
    metrics = ['Best Accuracy', 'Least Loss']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[metric], kde=True, bins=20)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric.lower()}_distribution.png')
        plt.close()

def feature_vs_accuracy(df: pd.DataFrame, output_dir: Path):
    """
    Plot how features impact Best Accuracy.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='feature_type', y='Best Accuracy', hue='model_type')
    plt.title('Impact of Features on Best Accuracy')
    plt.xlabel('Feature Type')
    plt.ylabel('Best Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_vs_accuracy.png')
    plt.close()



def boxplot_accuracy_vs_feature(df: pd.DataFrame, output_dir: Path):
    """
    Generate a boxplot for Best Accuracy grouped by Feature Types.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='feature_type', y='Best Accuracy', palette='Set2')
    plt.title('Best Accuracy by Feature Type')
    plt.xlabel('Feature Type')
    plt.ylabel('Best Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_feature_boxplot.png')
    plt.close()

def boxplot_loss_vs_model(df: pd.DataFrame, output_dir: Path):
    """
    Generate a boxplot for Loss grouped by Model Types.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='model_type', y='Least Loss', palette='coolwarm')
    plt.title('Least Loss by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('Least Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_vs_model_boxplot.png')
    plt.close()

def boxplot_training_time_vs_feature(df: pd.DataFrame, output_dir: Path):
    """
    Generate a boxplot for Total Training Time grouped by Feature Types.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='feature_type', y='total_training_time', palette='Spectral')
    plt.title('Training Time by Feature Type')
    plt.xlabel('Feature Type')
    plt.ylabel('Total Training Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_vs_feature_boxplot.png')
    plt.close()

def extended_analysis_with_boxplots(df: pd.DataFrame, output_dir: Path):
    """
    Perform extended analysis and generate additional visualizations including boxplots.
    
    Parameters:
    - df (DataFrame): Processed data.
    - output_dir (Path): Directory to save analysis results.
    """
    correlation_analysis(df, output_dir)
    accuracy_vs_training_time(df, output_dir)
    distribution_plots(df, output_dir)
    feature_vs_accuracy(df, output_dir)

    # Add boxplots
    boxplot_accuracy_vs_feature(df, output_dir)
    boxplot_loss_vs_model(df, output_dir)
    boxplot_training_time_vs_feature(df, output_dir)
    
def effect_of_features(df: pd.DataFrame, output_dir: Path):
    """
    Plot the effect of features on Best Accuracy and Least Loss.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    # Effect on Best Accuracy
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='feature_type', y='Best Accuracy', palette='Set3')
    plt.title('Effect of Features on Accuracy')
    plt.xlabel('Feature Type')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_of_features_accuracy.png')
    plt.close()

    # Effect on Least Loss
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='feature_type', y='Least Loss', palette='coolwarm')
    plt.title('Effect of Features on Least Loss')
    plt.xlabel('Feature Type')
    plt.ylabel('Least Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_of_features_loss.png')
    plt.close()

def effect_of_chunk_length_unet(df: pd.DataFrame, output_dir: Path):
    """
    Analyze the effect of chunk length on U-Net model performance.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    unet_df = df[df['model_type'] == 'U_Net']
    if unet_df.empty:
        logging.info("No data available for U-Net model.")
        return

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=unet_df, x='Length of Chunks (s)', y='Best Accuracy', color='gray')
    plt.title('Effect of Chunk Length on Accuracy (U-Net)')
    plt.xlabel('Chunk Length (s)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_of_chunk_length_unet_accuracy.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=unet_df, x='Length of Chunks (s)', y='Least Loss', color='black')
    plt.title('Effect of Chunk Length on Least Loss (U-Net)')
    plt.xlabel('Chunk Length (s)')
    plt.ylabel('Least Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_of_chunk_length_unet_loss.png')
    plt.close()

def best_performing_models(df: pd.DataFrame, output_dir: Path):
    """
    Plot the best performing models by accuracy with value labels.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    top_models = df.nlargest(5, 'Best Accuracy')

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=top_models, x='Name', y='Best Accuracy', hue='model_type', palette='Set2')
    
    # # Add value labels on top of each bar
    # for i in ax.containers:
    #     ax.bar_label(i, fmt='%.2f', padding=3)
    
    plt.title('Top 5 Best Performing Models by Accuracy')
    plt.xlabel('Run ID')
    plt.ylabel('Best Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    # Move legend to upper left corner
    plt.legend(title='Model Type', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_performing_models.png')
    plt.close()

def extended_analysis_with_detailed_plots(df: pd.DataFrame, output_dir: Path):
    """
    Perform extended analysis and generate detailed visualizations for features, chunk length, and top models.
    
    Parameters:
    - df (DataFrame): Processed data.
    - output_dir (Path): Directory to save analysis results.
    """
    effect_of_features(df, output_dir)
    effect_of_chunk_length_unet(df, output_dir)
    best_performing_models(df, output_dir)

def plot_model_complexity_vs_accuracy(df: pd.DataFrame, output_dir: Path):
    """
    Plot the impact of model complexity (number of parameters) on Best Accuracy.
    
    Parameters:
    - df (DataFrame): Data for analysis.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Number of Parameters', y='Best Accuracy', hue='model_type', style='feature_type', s=100)
    plt.title('Impact of Model Complexity on Best Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Best Accuracy')
    plt.legend(title='Model & Feature Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_complexity_vs_accuracy.png')
    plt.close()

def analyze_all_metrics(df: pd.DataFrame, output_dir: Path):
    """
    Analyze all performance metrics across different models and configurations.
    Show up to 20 top performers for each metric.
    
    Parameters:
    - df (DataFrame): Data for analysis
    - output_dir (Path): Directory to save the analysis results
    """
    metrics = {
        'Speech Precision': 'val_speech_precision',
        'Speech Recall': 'val_speech_recall',
        'Speech F1': 'val_speech_f1',
        'Music Precision': 'val_music_precision',
        'Music Recall': 'val_music_recall',
        'Music F1': 'val_music_f1',
        'MCC': 'val_mcc'
    }
    
    # Find top performers for each metric (up to 20)
    print("\n=== Top Performers for Each Metric ===")
    for metric_name, metric_col in metrics.items():
        # top_models = df.nlargest(20, metric_col).drop_duplicates(subset=[metric_col])
        top_models = df.nlargest(20, metric_col)

        print(f"\nTop Models for {metric_name}:")
        print(f"{'Model Type':<15} {'Feature Type':<12} {'Length (s)':<10} {metric_name:<10}")
        print("-" * 50)
        for idx, row in top_models.iterrows():
            print(f"{row['model_type']:<15} {row['feature_type']:<12} {row['length_in_seconds']:<10.1f} {row[metric_col]:.4f}")

    # Find models that perform well across all metrics
    df['avg_performance'] = df[list(metrics.values())].mean(axis=1)
    df['min_performance'] = df[list(metrics.values())].min(axis=1)
    
    print("\n=== Best Overall Performers (Balanced across all metrics) ===")
    top_balanced = df.nlargest(20, 'avg_performance')
    print("\nTop Balanced Performers:")
    print(f"{'Model Type':<15} {'Feature Type':<12} {'Length (s)':<10} {'Avg Score':<10} {'Min Score':<10}")
    print("-" * 65)
    for idx, row in top_balanced.iterrows():
        print(f"{row['model_type']:<15} {row['feature_type']:<12} "
              f"{row['length_in_seconds']:<10.1f} {row['avg_performance']:.4f} "
              f"{row['min_performance']:.4f}")

    print("\n=== Specialized Models ===")
    
    # Best for Speech Detection
    df['speech_score'] = df['val_speech_precision'] * df['val_speech_recall']
    top_speech = df.nlargest(10, 'speech_score')
    print("\nBest for Speech Detection:")
    print(f"{'Model Type':<15} {'Feature Type':<12} {'Length (s)':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 75)
    for idx, row in top_speech.iterrows():
        print(f"{row['model_type']:<15} {row['feature_type']:<12} "
              f"{row['length_in_seconds']:<10.1f} {row['val_speech_precision']:.4f} "
              f"{row['val_speech_recall']:.4f} {row['val_speech_f1']:.4f}")

    # Best for Music Detection
    df['music_score'] = df['val_music_precision'] * df['val_music_recall']
    top_music = df.nlargest(10, 'music_score')
    print("\nBest for Music Detection:")
    print(f"{'Model Type':<15} {'Feature Type':<12} {'Length (s)':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 75)
    for idx, row in top_music.iterrows():
        print(f"{row['model_type']:<15} {row['feature_type']:<12} "
              f"{row['length_in_seconds']:<10.1f} {row['val_music_precision']:.4f} "
              f"{row['val_music_recall']:.4f} {row['val_music_f1']:.4f}")

    # Analyze best chunk lengths
    print("\n=== Best Performing Chunk Lengths ===")
    for metric_name, metric_col in metrics.items():
        best_lengths = df.groupby('length_in_seconds')[metric_col].max().sort_values(ascending=False)
        print(f"\nTop Chunk Lengths for {metric_name}:")
        print(f"{'Length (s)':<10} {'Score':<10}")
        print("-" * 25)
        for length, score in best_lengths.items():
            print(f"{length:<10.1f} {score:.4f}")

    # Quick Processing Models (good performance with short chunks)
    short_chunks = df[df['length_in_seconds'] <= 2].copy()
    short_chunks['quick_score'] = short_chunks[list(metrics.values())].mean(axis=1)
    top_quick = short_chunks.nlargest(10, 'quick_score')
    print("\nBest Quick Processing Models (≤ 2s chunks):")
    print(f"{'Model Type':<15} {'Feature Type':<12} {'Length (s)':<10} {'Score':<10}")
    print("-" * 50)
    for idx, row in top_quick.iterrows():
        print(f"{row['model_type']:<15} {row['feature_type']:<12} "
              f"{row['length_in_seconds']:<10.1f} {row['quick_score']:.4f}")


def main():
    """Main function to process the CSV file and generate reports."""
    setup_logging()

    csv_file = Path('/Users/zainhazzouri/projects/Bachelor_Thesis/results/runs.csv'      )
    output_dir = Path("/Users/zainhazzouri/projects/Bachelor_Thesis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    df = load_csv(csv_file)
    df = preprocess_data(df)


    plot_ball_chart(df, output_dir)
    extended_analysis_with_boxplots(df, output_dir)
    extended_analysis_with_detailed_plots(df, output_dir)
    plot_model_complexity_vs_accuracy(df, output_dir)
    analyze_all_metrics(df, output_dir)



    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()