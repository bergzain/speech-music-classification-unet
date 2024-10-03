# xai.py
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
import json
import math
from pathlib import Path

from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet
from datapreprocessing import AudioProcessor


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device.type}")
    return device


def get_target_layer(model, model_name):
    if model_name == 'U_Net':
        target_layers = [model.Up_conv2.conv2]
    elif model_name == 'R2U_Net':
        target_layers = [model.Up_RRCNN2.RCNN[1].conv[0]]
    elif model_name == 'R2AttU_Net':
        target_layers = [model.Up_RRCNN2.RCNN[1].conv[0]]
    elif model_name == 'AttentionUNet':
        target_layers = [model.Up_conv2.conv2]
    else:
        raise ValueError(f"Unknown model name {model_name}")
    return target_layers


def get_cam_methods():
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
    return methods


def load_params(params_path):
    params = {}
    for param_file in os.listdir(params_path):
        with open(os.path.join(params_path, param_file), 'r') as f:
            params[param_file] = f.read().strip()
    return params


def load_tags(tags_path):
    tags = {}
    for tag_file in os.listdir(tags_path):
        with open(os.path.join(tags_path, tag_file), 'r') as f:
            tags[tag_file] = f.read().strip()
    return tags


def load_metrics(metrics_path):
    metrics = {}
    for metric_file in os.listdir(metrics_path):
        metric_name = metric_file
        with open(os.path.join(metrics_path, metric_file), 'r') as f:
            lines = f.readlines()
            if lines:
                # Each line is in the format: timestamp value step
                # We take the last line (latest value)
                last_line = lines[-1]
                try:
                    timestamp, value, step = last_line.strip().split(' ')
                    value = float(value)
                    metrics[metric_name] = value
                except ValueError:
                    pass  # Skip if line format is unexpected
    return metrics


def main():
    parser = argparse.ArgumentParser(description='XAI with pytorch_grad_cam for U-Net models')
    parser.add_argument('--mlflow_dir', type=str, required=True, help='Path to the MLflow directory')
    parser.add_argument('--audio_dir', type=str, default='', help='Path to the test data directory')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index in the test dataset')
    parser.add_argument('--save_global_path', type=str, default='./cam_outputs/',
                        help='Global path to save the CAM images if artifacts path is not writable')
    parser.add_argument('--path_type', type=str, default='cluster', choices=['cluster', 'local'],
                        help='Path type: cluster or local')
    args = parser.parse_args()

    device = get_device()

    # Set paths based on the path type
    if args.path_type == 'cluster':
        main_path = "/home/zhazzouri/speech-music-classification-unet/"
        data_main_path = "/netscratch/zhazzouri/dataset/"
        experiments_path = "/netscratch/zhazzouri/experiments/" # path to the folder where the mlflow experiments are stored
    else:
        main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
        data_main_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
        experiment_path = "/Users/zainhazzouri/projects/Master-thesis-experiments/" # path to the folder where the mlflow experiments are stored
    

    # If audio_dir is not provided, use default path
    if args.audio_dir == '':
        args.audio_dir = os.path.join(data_main_path, 'test/')
        if not os.path.exists(args.audio_dir):
            raise FileNotFoundError(
                f"Test data not found at {args.audio_dir}. Please provide the correct path using --audio_dir.")

    mlflow_dir = args.mlflow_dir

    # Iterate over MLflow runs
    for root, dirs, files in os.walk(mlflow_dir):
        for dir_name in dirs:
            run_path = os.path.join(root, dir_name)
            artifacts_path = os.path.join(run_path, 'artifacts')
            params_path = os.path.join(run_path, 'params')
            tags_path = os.path.join(run_path, 'tags')
            metrics_path = os.path.join(run_path, 'metrics')

            if not os.path.exists(artifacts_path) or not os.path.exists(params_path) or not os.path.exists(tags_path) or not os.path.exists(metrics_path):
                continue  # Skip if necessary directories do not exist

            # Check if model exists
            model_artifact_path = os.path.join(artifacts_path, 'model')
            if not os.path.exists(model_artifact_path):
                continue  # Skip runs without model artifacts

            # Load parameters, tags, and metrics
            params = load_params(params_path)
            tags = load_tags(tags_path)
            metrics = load_metrics(metrics_path)
            run_name = tags.get('mlflow.runName', dir_name)

            # **Check if Best Accuracy == 0 or Loss == inf**
            best_accuracy = metrics.get('best_accuracy', None)
            loss = metrics.get('loss', None)  # Assuming 'loss' is the metric name

            if best_accuracy is None or loss is None:
                print(f"Metrics not found for run {run_name}, skipping...")
                continue

            if best_accuracy == 0 or math.isinf(loss):
                print(f"Best Accuracy is 0 or Loss is infinite for run {run_name}, skipping...")
                continue

            # **Check if Grad-CAM outputs already exist**
            save_dir = os.path.join(artifacts_path, 'gradcam_outputs')
            combined_save_filename = f'{run_name}_All_CAMs_sample_{args.sample_idx}.png'
            combined_save_path = os.path.join(save_dir, combined_save_filename)
            if os.path.exists(combined_save_path):
                print(f"Grad-CAM outputs already exist for run {run_name}, skipping...")
                continue  # Skip this run

            # Extract necessary parameters
            model_name = params.get('model_name')
            if model_name is None:
                print(f"model_name not found for run {run_name}, skipping...")
                continue

            type_of_transformation = params.get('type_of_transformation', 'MFCC')
            n_mfcc = int(params.get('n_mfcc', '13'))
            length_in_seconds = int(params.get('length_in_seconds', '5'))

            # Update dataset transforms based on parameters
            test_dataset = AudioProcessor(audio_dir=args.audio_dir, n_mfcc=n_mfcc,
                                          length_in_seconds=length_in_seconds,
                                          type_of_transformation=type_of_transformation)
            input_tensor, label = test_dataset[args.sample_idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

            # Load the model
            models_dict = {
                "U_Net": U_Net(),
                "R2U_Net": R2U_Net(),
                "R2AttU_Net": R2AttU_Net(),
                "AttentionUNet": AttentionUNet()
            }
            if model_name not in models_dict:
                print(f"Unknown model name {model_name} for run {run_name}, skipping...")
                continue

            model = models_dict[model_name].to(device)

            # Load model state
            try:
                # Assuming the model is saved using mlflow.pytorch
                import mlflow.pytorch
                model = mlflow.pytorch.load_model(model_artifact_path).to(device)
            except Exception as e:
                print(f"Failed to load model for run {run_name}: {e}")
                continue

            model.eval()

            # Get the target layer
            try:
                target_layers = get_target_layer(model, model_name)
            except Exception as e:
                print(f"Failed to get target layer for run {run_name}: {e}")
                continue

            # Get CAM methods
            cam_methods_dict = get_cam_methods()

            # Compute the output and get the predicted class
            output = model(input_tensor)
            _, predicted_class = output.max(dim=1)
            target_category = predicted_class.item()

            # Define the targets for CAM
            targets = [ClassifierOutputTarget(target_category)]

            # Prepare to plot
            num_methods = len(cam_methods_dict)
            num_cols = 2  # Adjust as needed
            num_rows = (num_methods + 1) // num_cols + 1  # Additional row for the original image

            plt.figure(figsize=(10, num_rows * 4))

            # Normalize the input image for visualization
            input_image = input_tensor.cpu().numpy()[0, 0, :, :]  # Assuming single channel
            input_image_norm = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image_rgb = np.stack([input_image_norm]*3, axis=-1)

            # Plot the original image
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(input_image_norm, cmap='viridis')
            plt.title(f'Original Input\nRun: {run_name}')
            plt.axis('off')

            # Iterate over CAM methods
            for idx, (method_name, cam_method_class) in enumerate(cam_methods_dict.items(), start=2):
                print(f"Processing {method_name} for run {run_name}...")
                try:
                    cam = cam_method_class(model=model, target_layers=target_layers, use_cuda=(device.type == 'cuda'))
                    grayscale_cam = cam(input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0]  # Get the CAM for the first (and only) sample

                    # Overlay the CAM on the input image
                    cam_image = show_cam_on_image(input_image_rgb, grayscale_cam, use_rgb=True)

                    # Plot the CAM
                    plt.subplot(num_rows, num_cols, idx)
                    plt.imshow(cam_image)
                    plt.title(f'{method_name}\nRun: {run_name}')
                    plt.axis('off')

                    # Save individual CAM images if needed
                    os.makedirs(save_dir, exist_ok=True)
                    save_filename = f'{model_name}_{method_name}_sample_{args.sample_idx}.png'
                    plt.imsave(os.path.join(save_dir, save_filename), cam_image)
                except Exception as e:
                    print(f"Failed to process {method_name} for run {run_name}: {e}")
                    plt.subplot(num_rows, num_cols, idx)
                    plt.text(0.5, 0.5, f"Error\n{method_name}\nRun: {run_name}", horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')

            plt.tight_layout()

            # Save the combined figure
            combined_save_filename = f'{run_name}_All_CAMs_sample_{args.sample_idx}.png'
            combined_save_path = os.path.join(save_dir, combined_save_filename)

            try:
                plt.savefig(combined_save_path, dpi=300)
                print(f"Saved Grad-CAM visualization for run {run_name} at {combined_save_path}")
            except Exception as e:
                print(f"Failed to save Grad-CAM visualization for run {run_name}: {e}")
                # Save to global path if artifacts path is not writable
                os.makedirs(args.save_global_path, exist_ok=True)
                combined_save_path = os.path.join(args.save_global_path, combined_save_filename)
                plt.savefig(combined_save_path, dpi=300)
                print(f"Saved Grad-CAM visualization for run {run_name} at {combined_save_path}")

            plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    main()