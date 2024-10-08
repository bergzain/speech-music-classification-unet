import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM,
    ScoreCAM, LayerCAM, FullGrad, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import math
from pathlib import Path
import librosa
import librosa.display
import random
from scipy.ndimage import zoom
import shutil  # For deleting directories

from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet
from datapreprocessing import AudioProcessor
from torch.utils.data import Dataset  # Ensure Dataset is imported

def get_device():
    """Determine the available device (GPU, MPS, or CPU) for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    return device

def set_seed(seed):
    """Set the random seed for reproducibility."""
    print(f"[INFO] Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_target_layer(model, model_name):
    """Retrieve the target layer for CAM based on the model architecture."""
    if model_name == 'U_Net':
        target_layers = [model.Up_conv2.conv2]
    elif model_name == 'R2U_Net':
        target_layers = [model.Up_RRCNN2.RCNN[1].conv[0]]
    elif model_name == 'R2AttU_Net':
        target_layers = [model.Up_RRCNN2.RCNN[1].conv[0]]
    elif model_name == 'AttentionUNet':
        target_layers = [model.Up_conv2.conv2]
    else:
        raise ValueError(f"[ERROR] Unknown model name: {model_name}")
    return target_layers

def get_cam_methods():
    """Return a dictionary of available CAM methods."""
    methods = {
        'GradCAM': GradCAM,
        'HiResCAM': HiResCAM,
        'GradCAMElementWise': GradCAMElementWise,
        'GradCAMPlusPlus': GradCAMPlusPlus,
        'XGradCAM': XGradCAM,
        'AblationCAM': AblationCAM,
        'ScoreCAM': ScoreCAM,
        'LayerCAM': LayerCAM,
        'FullGrad': FullGrad,
        'EigenCAM': EigenCAM,
        'EigenGradCAM': EigenGradCAM
    }
    print(f"[DEBUG] Available CAM methods: {list(methods.keys())}")
    return methods

def parse_run_name(run_name):
    """
    Parse the run name to extract model_name, type_of_transformation, n_mfcc, and length_in_seconds.

    Expected run_name format examples:
    - R2AttU_Net_lfcc-delta_32_len5S
    - U_Net_delta_32_len30S
    - AttentionUNet_MFCC_32_len5S
    """
    print(f"[INFO] Parsing run name: {run_name}")
    try:
        parts = run_name.split('_')
        if len(parts) < 4:
            raise ValueError(f"[ERROR] Run name '{run_name}' does not have enough parts to parse.")

        # Identify model_name by matching known models
        known_models = ['U_Net', 'R2U_Net', 'R2AttU_Net', 'AttentionUNet']
        model_name = None
        for model in known_models:
            if run_name.startswith(model):
                model_name = model
                break
        if model_name is None:
            raise ValueError(f"[ERROR] Model name not found in run name '{run_name}'.")

        # Remove model_name from parts
        remaining = run_name[len(model_name) + 1:]  # +1 to remove the underscore
        remaining_parts = remaining.split('_')
        if len(remaining_parts) != 3:
            raise ValueError(f"[ERROR] Run name '{run_name}' does not conform to expected format after model name.")

        type_of_transformation = remaining_parts[0]
        n_mfcc = int(remaining_parts[1])
        len_part = remaining_parts[2]
        if not len_part.startswith('len') or not len_part.endswith('S'):
            raise ValueError(f"[ERROR] Length part '{len_part}' is not in expected format 'lenXS'.")
        length_in_seconds = int(len_part[3:-1])  # Remove 'len' prefix and 'S' suffix

        print(f"[INFO] Parsed Parameters - Model: {model_name}, Transformation: {type_of_transformation}, "
              f"n_mfcc: {n_mfcc}, Length: {length_in_seconds} seconds")

        return model_name, type_of_transformation, n_mfcc, length_in_seconds
    except Exception as e:
        print(f"[ERROR] Failed to parse run name '{run_name}': {e}")
        raise e

def load_tags(tags_path):
    """Load tags from the MLflow run."""
    tags = {}
    try:
        for tag_file in os.listdir(tags_path):
            with open(os.path.join(tags_path, tag_file), 'r') as f:
                tags[tag_file] = f.read().strip()
        print(f"[INFO] Loaded tags from {tags_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load tags from {tags_path}: {e}")
    return tags

def load_metrics(metrics_path):
    """Load metrics from the MLflow run with robust parsing."""
    metrics = {}
    try:
        if not os.path.exists(metrics_path):
            print(f"[WARNING] Metrics path does not exist: {metrics_path}")
            return metrics
        
        for metric_file in os.listdir(metrics_path):
            metric_name = metric_file
            metric_file_path = os.path.join(metrics_path, metric_file)
            print(f"[DEBUG] Loading metric file: {metric_file_path}")
            
            with open(metric_file_path, 'r') as f:
                lines = f.readlines()
                
                if not lines:
                    print(f"[WARNING] Metric file '{metric_file}' is empty.")
                    continue  # Skip empty metric files
                
                last_line = lines[-1].strip()
                print(f"[DEBUG] Last line in '{metric_file}': '{last_line}'")
                
                # Initialize value as None
                value = None
                
                # Strategy 1: Split by space
                parts = last_line.split(' ')
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        print(f"[DEBUG] Parsed '{metric_name}' using space separator: {value}")
                    except ValueError:
                        print(f"[WARNING] Unable to parse value using space separator in '{metric_file}'. Trying next strategy.")
                
                # Strategy 2: Split by comma
                if value is None and ',' in last_line:
                    parts = last_line.split(',')
                    if len(parts) >= 2:
                        try:
                            value = float(parts[1])
                            print(f"[DEBUG] Parsed '{metric_name}' using comma separator: {value}")
                        except ValueError:
                            print(f"[WARNING] Unable to parse value using comma separator in '{metric_file}'. Trying next strategy.")
                
                # Strategy 3: Single value per line
                if value is None:
                    try:
                        value = float(last_line)
                        print(f"[DEBUG] Parsed '{metric_name}' as single float value: {value}")
                    except ValueError:
                        print(f"[WARNING] Unable to parse value as single float in '{metric_file}'. Trying next strategy.")
                
                # Strategy 4: Extract float from last part
                if value is None:
                    parts = last_line.split()
                    for part in reversed(parts):
                        try:
                            value = float(part)
                            print(f"[DEBUG] Parsed '{metric_name}' by extracting float from '{part}': {value}")
                            break
                        except ValueError:
                            continue
                    if value is None:
                        print(f"[WARNING] Could not parse any float value in '{metric_file}'. Skipping this metric.")
                
                if value is not None:
                    metrics[metric_name] = value
                else:
                    print(f"[WARNING] Failed to parse '{metric_name}' from '{metric_file}'.")
        
        print(f"[INFO] Loaded metrics from {metrics_path}: {metrics}")
    
    except Exception as e:
        print(f"[ERROR] Failed to load metrics from {metrics_path}: {e}")
    
    return metrics

def plot_cam_side_by_side_with_librosa(original, cam_images, titles, save_path, sample_type, run_name, transformation_type, target_sample_rate=44100):
    """
    Plot the original MFCC (or other transformation) alongside CAM overlays using librosa for audio visualization.

    Parameters:
    - original (np.ndarray): Original MFCC (or other transformation) array.
    - cam_images (list of np.ndarray): List of CAM arrays.
    - titles (list of str): Titles for each CAM method.
    - save_path (str): Directory to save the plot.
    - sample_type (str): Type of the sample ('music' or 'speech').
    - run_name (str): Name of the run for naming individual CAM images.
    - transformation_type (str): Type of transformation applied (e.g., 'MFCC', 'LFCC', etc.).
    - target_sample_rate (int): Sampling rate for librosa display.
    """
    try:
        print(f"[INFO] Plotting CAMs for sample type: {sample_type} with transformation: {transformation_type}")
        num_cam_techniques = len(cam_images)
        
        # Define separate directories for individual and combined images
        individual_save_dir = os.path.join(save_path, 'individual')
        combined_save_dir = os.path.join(save_path, 'combined')
        os.makedirs(individual_save_dir, exist_ok=True)
        os.makedirs(combined_save_dir, exist_ok=True)

        # Initialize the combined figure
        fig_combined, axes_combined = plt.subplots(1, num_cam_techniques + 1, figsize=((num_cam_techniques + 1) * 5, 5))

        # Reshape the original data
        original = original.squeeze()

        # Plot original transformation using librosa on the far left
        img = librosa.display.specshow(original, sr=target_sample_rate, x_axis='time', ax=axes_combined[0], cmap='magma')
        fig_combined.colorbar(img, ax=axes_combined[0], format="%+2.f dB")
        axes_combined[0].set_title(f"Original {transformation_type} ({sample_type})")

        # Iterate over each CAM method and save individually
        for i, (cam_image, title) in enumerate(zip(cam_images, titles), start=1):
            # Normalize CAM image
            cam_image = (cam_image - np.min(cam_image)) / (np.max(cam_image) - np.min(cam_image))
            cam_image = np.clip(cam_image, 0, 1)
            # Resize the CAM image to match the original transformation dimensions using zoom
            zoom_factors = (original.shape[0] / cam_image.shape[0], original.shape[1] / cam_image.shape[1])
            cam_image_resized = zoom(cam_image, zoom_factors)

            # Plot transformation with CAM overlay on the combined figure
            axes_combined[i].imshow(original, cmap='magma', aspect='auto', origin='lower')
            axes_combined[i].imshow(cam_image_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
            # To ensure the colorbar corresponds to the transformation
            img_cam = axes_combined[i].images[0]
            fig_combined.colorbar(img_cam, ax=axes_combined[i], format="%+2.f dB")
            axes_combined[i].set_title(f"{title} CAM ({sample_type})")

            # Save individual CAM image in 'individual' subdirectory
            individual_cam_filename = f"{run_name}_{sample_type}_{title}.png"
            individual_cam_path = os.path.join(individual_save_dir, individual_cam_filename)
            fig_individual, ax_individual = plt.subplots(figsize=(5, 5))
            ax_individual.imshow(original, cmap='magma', aspect='auto', origin='lower')
            ax_individual.imshow(cam_image_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
            ax_individual.set_title(f"{title} CAM ({sample_type})")
            # Add colorbar
            img_individual = ax_individual.imshow(cam_image_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
            fig_individual.colorbar(img_individual, ax=ax_individual, format="%+2.f dB")
            fig_individual.tight_layout()
            fig_individual.savefig(individual_cam_path)
            plt.close(fig_individual)
            print(f"[INFO] Saved individual CAM image: {individual_cam_path}")

        # Save the combined CAM image in 'combined' subdirectory
        fig_combined.tight_layout()
        combined_save_filename = f"All_CAMs_{run_name}_{sample_type}.png"
        combined_save_path = os.path.join(combined_save_dir, combined_save_filename)
        fig_combined.savefig(combined_save_path)
        plt.close(fig_combined)
        print(f"[INFO] Saved combined CAM visualization for {sample_type} at {combined_save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to plot CAMs for {sample_type}: {e}")

def delete_gradcam_outputs(experiments_path):
    """Delete all gradcam_outputs directories within the experiments path."""
    print(f"[INFO] Deleting all 'gradcam_outputs' directories in {experiments_path}...")
    deleted_count = 0
    try:
        for root, dirs, files in os.walk(experiments_path):
            for dir_name in dirs:
                if dir_name == 'gradcam_outputs':
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    print(f"[INFO] Deleted directory: {dir_path}")
        print(f"[INFO] Deleted {deleted_count} 'gradcam_outputs' directories.")
    except Exception as e:
        print(f"[ERROR] Failed to delete 'gradcam_outputs' directories: {e}")

def main():
    parser = argparse.ArgumentParser(description='XAI with pytorch_grad_cam for U-Net models')
    parser.add_argument('--mlflow_dir', type=str, required=True, help='Path to the MLflow directory')
    parser.add_argument('--audio_dir', type=str, default='', help='Path to the test data directory')
    parser.add_argument('--save_global_path', type=str, default='./cam_outputs/',
                        help='Global path to save the CAM images if artifacts path is not writable')
    parser.add_argument('--path_type', type=str, default='cluster', choices=['cluster', 'local'],
                        help='Path type: cluster or local')
    parser.add_argument('--rerun', action='store_true',
                        help='If set, rerun Grad-CAM even if outputs already exist')
    parser.add_argument('--delete_cam', action='store_true',
                        help='If set, delete all grad-cam output folders from the experiments before running')
    parser.add_argument('--seed', type=int, default=16, help='Random seed for reproducibility')
    args = parser.parse_args()

    print("[INFO] Starting XAI script...")

    # Set the random seed for reproducibility
    set_seed(args.seed)

    device = get_device()

    # Set paths based on the path type
    if args.path_type == 'cluster':
        main_path = "/home/zhazzouri/speech-music-classification-unet/"
        data_main_path = "/netscratch/zhazzouri/dataset/"
        experiments_path = "/netscratch/zhazzouri/experiments/"  # Path to the folder where the mlflow experiments are stored
    else:
        main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
        data_main_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
        experiments_path = "/Users/zainhazzouri/projects/Master-thesis-experiments/"  # Path to the folder where the mlflow experiments are stored

    # Handle deletion of existing Grad-CAM outputs if --delete_cam is set
    if args.delete_cam:
        delete_gradcam_outputs(experiments_path)

    # If audio_dir is not provided, use default path
    if args.audio_dir == '':
        args.audio_dir = os.path.join(data_main_path, 'test/')
        print(f"[INFO] No audio_dir provided. Using default path: {args.audio_dir}")
        if not os.path.exists(args.audio_dir):
            print(f"[ERROR] Test data not found at {args.audio_dir}. Please provide the correct path using --audio_dir.")
            return

    mlflow_dir = args.mlflow_dir
    print(f"[INFO] MLflow directory: {mlflow_dir}")

    # Iterate over MLflow runs (two levels deep)
    for root, dirs, files in os.walk(mlflow_dir):
        for dir_name in dirs:
            first_level_run_path = os.path.join(root, dir_name)
            try:
                second_level_dirs = os.listdir(first_level_run_path)
                print(f"[DEBUG] Found second-level directories: {second_level_dirs} in {first_level_run_path}")
            except Exception as e:
                print(f"[ERROR] Unable to list directories in {first_level_run_path}: {e}")
                continue

            for second_dir in second_level_dirs:
                run_path = os.path.join(first_level_run_path, second_dir)
                artifacts_path = os.path.join(run_path, 'artifacts')
                params_path = os.path.join(run_path, 'params')  # Not used since parameters are parsed from run_name
                tags_path = os.path.join(run_path, 'tags')
                metrics_path = os.path.join(run_path, 'metrics')

                print(f"\n[INFO] Processing run directory: {run_path}")

                if not os.path.exists(artifacts_path) or not os.path.exists(tags_path):
                    print(f"[WARNING] Necessary directories missing in run {second_dir}. Skipping...")
                    continue  # Skip if necessary directories do not exist

                # Check if model exists
                model_artifact_path = os.path.join(artifacts_path, 'model')
                if not os.path.exists(model_artifact_path):
                    print(f"[WARNING] Model artifact not found for run {second_dir}. Skipping...")
                    continue  # Skip runs without model artifacts

                # Load tags to get run_name
                try:
                    tags = load_tags(tags_path)
                    run_name = tags.get('mlflow.runName', second_dir)
                    print(f"[INFO] Run name: {run_name}")
                except Exception as e:
                    print(f"[ERROR] Failed to load tags for run {second_dir}: {e}")
                    continue

                # Parse run_name to extract parameters
                try:
                    model_name, type_of_transformation, n_mfcc, length_in_seconds = parse_run_name(run_name)
                except Exception as e:
                    print(f"[ERROR] Skipping run {run_name} due to run name parsing error.")
                    continue

                # Define the Grad-CAM save directory
                save_dir = os.path.join(artifacts_path, 'gradcam_outputs')
                os.makedirs(save_dir, exist_ok=True)

                # Check if Grad-CAM outputs already exist unless rerun is requested
                if not args.rerun:
                    combined_save_filename_music = f'{run_name}_All_CAMs_music.png'
                    combined_save_filename_speech = f'{run_name}_All_CAMs_speech.png'
                    combined_save_path_music = os.path.join(save_dir, 'combined', combined_save_filename_music)
                    combined_save_path_speech = os.path.join(save_dir, 'combined', combined_save_filename_speech)
                    if os.path.exists(combined_save_path_music) and os.path.exists(combined_save_path_speech):
                        print(f"[INFO] Grad-CAM outputs already exist for run {run_name}. Skipping...")
                        continue  # Skip this run

                # Load metrics if available
                try:
                    metrics = load_metrics(metrics_path)
                    print(f"[INFO] Loaded metrics for run {run_name}: {metrics}")
                except Exception as e:
                    print(f"[ERROR] Failed to load metrics for run {run_name}: {e}. Skipping...")
                    continue

                val_accuracy = metrics.get('val_accuracy', None)
                val_loss = metrics.get('val_loss', None)

                if val_accuracy is not None and val_loss is not None:
                    if val_accuracy == 0 or math.isinf(val_loss):
                        print(f"[WARNING] Validation Accuracy is 0 or Validation Loss is infinite for run {run_name}. Skipping...")
                        continue
                else:
                    print(f"[WARNING] Validation metrics not found for run {run_name}. Proceeding without metrics check.")
                    # Decide whether to skip or proceed
                    # For example, to skip runs without metrics:
                    # continue

                # Update dataset transforms based on parameters
                try:
                    print(f"[INFO] Initializing AudioProcessor with n_mfcc={n_mfcc}, length_in_seconds={length_in_seconds}, transformation={type_of_transformation}")
                    test_dataset = AudioProcessor(audio_dir=args.audio_dir, n_mfcc=n_mfcc,
                                                  length_in_seconds=length_in_seconds,
                                                  type_of_transformation=type_of_transformation)
                except Exception as e:
                    print(f"[ERROR] Failed to initialize AudioProcessor for run {run_name}: {e}. Skipping...")
                    continue

                # Select samples from different classes
                try:
                    music_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 0]
                    speech_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 1]

                    if len(music_indices) == 0 or len(speech_indices) == 0:
                        print(f"[WARNING] Run {run_name} does not have both music and speech samples. Skipping...")
                        continue

                    # Select one random sample from each class
                    music_idx = random.choice(music_indices)
                    speech_idx = random.choice(speech_indices)

                    samples = {
                        'music': test_dataset[music_idx][0].unsqueeze(0).to(device),
                        'speech': test_dataset[speech_idx][0].unsqueeze(0).to(device)
                    }

                    print(f"[INFO] Selected sample indices - Music: {music_idx}, Speech: {speech_idx}")
                except Exception as e:
                    print(f"[ERROR] Failed to select samples for run {run_name}: {e}. Skipping...")
                    continue

                # Load the model
                try:
                    models_dict = {
                        "U_Net": U_Net(),
                        "R2U_Net": R2U_Net(),
                        "R2AttU_Net": R2AttU_Net(),
                        "AttentionUNet": AttentionUNet()
                    }
                    if model_name not in models_dict:
                        print(f"[ERROR] Unknown model name '{model_name}' for run {run_name}. Skipping...")
                        continue

                    print(f"[INFO] Loading model '{model_name}' for run {run_name}...")
                    import mlflow.pytorch
                    model = mlflow.pytorch.load_model(model_artifact_path, map_location=device)
                    model.to(device)
                    model.eval()
                    print(f"[INFO] Model '{model_name}' loaded and set to evaluation mode.")
                except Exception as e:
                    print(f"[ERROR] Failed to load model for run {run_name}: {e}. Skipping...")
                    continue

                # Get the target layer
                try:
                    target_layers = get_target_layer(model, model_name)
                    print(f"[INFO] Retrieved target layers for model '{model_name}'.")
                except Exception as e:
                    print(f"[ERROR] Failed to get target layer for run {run_name}: {e}. Skipping...")
                    continue

                # Get CAM methods
                cam_methods_dict = get_cam_methods()
                print(f"[INFO] Retrieved {len(cam_methods_dict)} CAM methods.")

                # Prepare to plot for each sample type
                for sample_type, input_tensor in samples.items():
                    print(f"\n[INFO] Processing sample type: {sample_type}")
                    try:
                        # Compute the output and get the predicted class
                        output = model(input_tensor)
                        _, predicted_class = output.max(dim=1)
                        target_category = predicted_class.item()
                        print(f"[INFO] Predicted class for {sample_type} sample: {target_category}")

                        # Define the targets for CAM
                        targets = [ClassifierOutputTarget(target_category)]

                        # Get CAM methods
                        cam_images = []
                        cam_titles = []

                        for method_name, cam_method_class in cam_methods_dict.items():
                            print(f"[INFO] Processing CAM method: {method_name} for {sample_type} sample...")
                            try:
                                # Initialize CAM
                                cam = cam_method_class(model=model, target_layers=target_layers)
                                # Generate CAM
                                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                                grayscale_cam = grayscale_cam[0]  # Get the CAM for the first (and only) sample
                                cam_images.append(grayscale_cam)
                                cam_titles.append(method_name)
                                print(f"[INFO] Successfully processed {method_name} for {sample_type} sample.")
                            except Exception as e:
                                print(f"[ERROR] Failed to process {method_name} for run {run_name}, sample type: {sample_type}: {e}")
                                # Append a placeholder image
                                cam_images.append(np.zeros((input_tensor.size(-2), input_tensor.size(-1))))
                                cam_titles.append(f"{method_name} (Error)")

                        # Normalize the input image for visualization
                        input_image = input_tensor.cpu().numpy()[0, 0, :, :]  # Assuming single channel

                        # Plot and save the CAMs with side-by-side comparison using librosa
                        plot_cam_side_by_side_with_librosa(
                            original=input_image,
                            cam_images=cam_images,
                            titles=cam_titles,
                            save_path=save_dir,
                            sample_type=sample_type,
                            run_name=run_name,
                            transformation_type=type_of_transformation  # Pass the transformation type
                        )

                    except Exception as e:
                        print(f"[ERROR] Failed to process sample type {sample_type} for run {run_name}: {e}")
                        continue

                print(f"[INFO] Completed Grad-CAM visualization for run {run_name}")

if __name__ == '__main__':
    main()