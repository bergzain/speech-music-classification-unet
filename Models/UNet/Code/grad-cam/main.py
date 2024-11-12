# main.py

import sys
import os
import numpy as np

# Adjust sys.path to include parent directory and current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.insert(0, current_dir)   # For importing modules within grad-cam
sys.path.insert(0, parent_dir)    # For importing datapreprocessing and models

# Import from grad-cam modules
from grad_cam_utils import set_random_seeds, get_device, parse_model_filename
from model_utils import load_model, get_target_layer
from data_utils import find_samples_with_seed
from visualization_utils import (
    visualize_sample_with_features,
    analyze_audio_features,
    plot_multiple_cams_with_features,
    create_attention_maps_gif
)
from analysis_utils import (
    analyze_with_multiple_cams,
    create_comparative_analysis,
    generate_summary_report
)

# Import from parent directory
from datapreprocessing import AudioProcessor
from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet

import torch
import logging
import json
import traceback
from datetime import datetime
import soundfile as sf
from torch.utils.data import DataLoader



def main():
    """Main function for comprehensive XAI analysis with consistent signal processing."""
    # Configuration
    model_path = '/Users/zainhazzouri/projects/Master-thesis-experiments/results/U_Net_LFCC_32_len5.0S/U_Net_LFCC_32_len5.0S.pth'
    model_type, feature_type, n_mfcc, length_in_seconds = parse_model_filename(model_path)
    config = {
        'model_path': model_path,
        'test_data_path': "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/",
        'save_dir': '/Users/zainhazzouri/projects/Master-thesis-experiments/results/U_Net_LFCC_32_len5.0S/cam',
        'model_type': model_type,
        'random_seed': 42,
        'feature_params': {
            'type': feature_type,
            'n_mfcc': n_mfcc,
            'length_in_seconds': length_in_seconds,
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048
        },
        'batch_size': 16,
        'samples_per_class': 2
    }
    try:
        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config['save_dir'], f"analysis_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(save_dir, 'analysis_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Analysis started")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")
        # Setup environment
        device = get_device()
        set_random_seeds(config['random_seed'])
        logging.info(f"Using device: {device}")
        # Load model and dataset using extracted parameters
        model = load_model(config['model_path'], config['model_type'], device)
        target_layer = get_target_layer(model, config['model_type'])
        # Parameters for feature extraction
        feature_params = config['feature_params']
        sr = feature_params['sample_rate']
        n_mfcc = feature_params['n_mfcc']
        length_in_seconds = feature_params['length_in_seconds']
        n_fft = feature_params['n_fft']
        hop_length = feature_params['hop_length']
        win_length = feature_params['win_length']
        dataset = AudioProcessor(
            audio_dir=config['test_data_path'],
            n_mfcc=n_mfcc,
            length_in_seconds=length_in_seconds,
            type_of_transformation=feature_params['type']
        )
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            worker_init_fn=lambda worker_id: torch.manual_seed(config['random_seed'] + worker_id)
        )
        # Find samples
        class_names = ['Music', 'Speech']
        samples_found = find_samples_with_seed(
            model=model,
            loader=loader,
            device=device,
            samples_per_class=config['samples_per_class'],
            seed=config['random_seed']
        )
        # Analysis results storage
        analysis_results = {class_name: [] for class_name in class_names}
        # Process each sample
        for class_idx, samples in samples_found.items():
            for i, (input_tensor, label) in enumerate(samples):
                sample_results = {}
                try:
                    # Get audio file path
                    matching_files = [
                        (idx, filepath) for idx, (filepath, file_label) 
                        in enumerate(dataset.audio_files_and_labels) 
                        if file_label == class_idx
                    ]
                    if not matching_files:
                        raise ValueError(f"No files found for class {class_idx}")
                    # Use the first matching file for this class
                    file_idx, file_path = matching_files[i]
                    print(f"Processing {class_names[class_idx]} sample {i + 1}")
                    print(f"Using file: {file_path}")
                    # Generate comprehensive visualizations
                    y, sr = visualize_sample_with_features(
                        file_path=file_path,
                        save_dir=save_dir,
                        class_name=class_names[class_idx],
                        sample_num=i+1,
                        feature_type=feature_params['type'],
                        n_mfcc=n_mfcc,
                        sr=sr,
                        length_in_seconds=length_in_seconds,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length
                    )
                    # Analyze audio features
                    feature_results = analyze_audio_features(
                        y=y,
                        sr=sr,
                        output_dir=save_dir,
                        class_name=class_names[class_idx],
                        sample_num=i+1,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length
                    )
                    # Generate CAM visualizations
                    cam_results = analyze_with_multiple_cams(
                        model=model,
                        input_tensor=input_tensor.unsqueeze(0),
                        target_layer=target_layer,
                        class_idx=label,
                        device=device
                    )
                    # Save results
                    audio_save_path = os.path.join(
                        save_dir,
                        f"{class_names[class_idx].lower()}_sample_{i+1}.wav"
                    )
                    sf.write(audio_save_path, y, sr, subtype='PCM_16')
                    # Store results
                    sample_results = {
                        'sample_idx': i+1,
                        'audio_path': audio_save_path,
                        'confidences': {name: conf for name, (_, conf) in cam_results.items()},
                        'feature_stats': {
                            'zcr_mean': float(np.mean(feature_results['zcr'])),
                            'rms_mean': float(np.mean(feature_results['rms'])),
                            'spectral_centroid_mean': float(np.mean(feature_results['spectral_centroids']))
                        },
                        'cam_results': cam_results
                    }
                    # Generate combined visualization
                    plot_multiple_cams_with_features(
                        input_tensor=input_tensor,
                        audio_features=feature_results,
                        results=cam_results,
                        class_name=class_names[class_idx],
                        save_path=os.path.join(
                            save_dir,
                            f'combined_analysis_{class_names[class_idx].lower()}_sample_{i+1}.png'
                        ),
                        feature_type=feature_params['type'],
                        n_mfcc=feature_params['n_mfcc'],
                        length_in_seconds=feature_params['length_in_seconds'],
                        sr=feature_params['sample_rate'],
                        hop_length=feature_params['hop_length']
                    )
                except Exception as e:
                    logging.error(f"Error processing {class_names[class_idx]} sample {i+1}: {str(e)}")
                    logging.error(traceback.format_exc())
                    sample_results['error'] = str(e)
                analysis_results[class_names[class_idx]].append(sample_results)
        # Generate final analyses
        create_comparative_analysis(analysis_results, save_dir)
        generate_summary_report(analysis_results, save_dir, config)
        # Save final metadata
        metadata = {
            'timestamp': timestamp,
            'config': config,
            'analysis_stats': {
                'total_samples': sum(len(samples) for samples in analysis_results.values()),
                'successful_analyses': sum(
                    1 for samples in analysis_results.values()
                    for sample in samples
                    if 'error' not in sample
                )
            }
        }
        with open(os.path.join(save_dir, 'analysis_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        # Create completion marker
        with open(os.path.join(save_dir, 'analysis_complete.txt'), 'w') as f:
            f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {save_dir}")
        print("\nGenerated files:")
        for root, dirs, files in os.walk(save_dir):
            level = root.replace(save_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            subindent = ' ' * 4 * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{subindent}{f}")
    except Exception as e:
        logging.error(f"Fatal error during analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error during execution: {e}")
        traceback.print_exc()