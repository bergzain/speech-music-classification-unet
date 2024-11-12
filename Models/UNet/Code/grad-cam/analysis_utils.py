# analysis_utils.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import json
from cam_utils import VanillaGradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM

def analyze_with_multiple_cams(model: torch.nn.Module, input_tensor: torch.Tensor, 
                               target_layer: torch.nn.Module, class_idx: int, 
                               device: str) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Analyze a sample with multiple CAM techniques."""
    cam_techniques = {
        'GradCAM': VanillaGradCAM(model, target_layer),
        'GradCAM++': GradCAMPlusPlus(model, target_layer),
        'LayerCAM': LayerCAM(model, target_layer),
        'ScoreCAM': ScoreCAM(model, target_layer)
    }
    results = {}
    for name, cam in cam_techniques.items():
        print(f"Generating {name} heatmap...")
        heatmap, output = cam.get_heatmap(input_tensor.to(device), class_idx)
        confidence = F.softmax(output, dim=1)[0, class_idx].item()
        results[name] = (heatmap, confidence)
    return results

def create_comparative_analysis(analysis_results: Dict[str, List[Dict]], save_dir: str) -> None:
    """Create comparative visualizations across samples and classes."""
    # Prepare data for plotting
    class_data = {
        class_name: {
            'confidences': [],
            'feature_stats': []
        } for class_name in analysis_results.keys()
    }
    for class_name, results in analysis_results.items():
        for result in results:
            class_data[class_name]['confidences'].append(
                list(result['confidences'].values())
            )
            class_data[class_name]['feature_stats'].append(
                list(result['feature_stats'].values())
            )
    # Confidence Distribution Plot
    plt.figure(figsize=(12, 6))
    positions = np.arange(len(analysis_results.keys()))
    width = 0.2
    for i, technique in enumerate(['GradCAM', 'GradCAM++', 'LayerCAM','ScoreCAM']):
        confidences = [np.mean([sample[i] for sample in class_data[cls]['confidences']])
                      for cls in analysis_results.keys()]
        errors = [np.std([sample[i] for sample in class_data[cls]['confidences']])
                 for cls in analysis_results.keys()]
        plt.bar(positions + i*width, confidences, width,
                label=technique, yerr=errors, capsize=5)
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.title('CAM Technique Confidence Comparison')
    plt.xticks(positions + width, analysis_results.keys())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_comparison.png'))
    plt.close()
    # Create correlation matrices
    for class_name in analysis_results.keys():
        create_correlation_matrix(class_name, analysis_results[class_name], save_dir)

def create_correlation_matrix(class_name: str, results: List[Dict], save_dir: str) -> None:
    """Create and save correlation matrix for features and confidences."""
    data = []
    columns = (
        ['GradCAM', 'GradCAM++', 'ScoreCAM'] +
        ['ZCR', 'RMS', 'Spectral_Centroid']
    )
    for result in results:
        row = (
            list(result['confidences'].values()) +
            list(result['feature_stats'].values())
        )
        data.append(row)
    correlation_matrix = np.corrcoef(np.array(data).T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
               xticklabels=columns,
               yticklabels=columns,
               annot=True,
               cmap='coolwarm',
               center=0)
    plt.title(f'{class_name} - Feature and Confidence Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{class_name.lower()}_correlations.png'))
    plt.close()

def generate_summary_report(analysis_results: Dict[str, List[Dict]], 
                            save_dir: str, config: Dict[str, Any]) -> None:
    """Generate comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(save_dir, 'analysis_summary.txt')
    with open(report_path, 'w') as f:
        f.write("Audio Classification XAI Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Analysis performed at: {timestamp}\n")
        f.write(f"Model type: {config['model_type']}\n")
        f.write(f"Feature type: {config['feature_params']['type']}\n\n")
        for class_name, results in analysis_results.items():
            f.write(f"\n{class_name} Analysis:\n")
            f.write("-" * (len(class_name) + 10) + "\n")
            for i, result in enumerate(results):
                f.write(f"\nSample {i+1}:\n")
                f.write("CAM Confidences:\n")
                for technique, conf in result['confidences'].items():
                    f.write(f"  {technique}: {conf:.4f}\n")
                f.write("\nFeature Statistics:\n")
                for feat, value in result['feature_stats'].items():
                    f.write(f"  {feat}: {value:.4f}\n")
            f.write("\n" + "="*50 + "\n")