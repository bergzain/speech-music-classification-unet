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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XAI with pytorch_grad_cam for U-Net models')
    parser.add_argument('--model', type=str, default='U_Net',
                        choices=['U_Net', 'R2U_Net', 'R2AttU_Net', 'AttentionUNet'], help='Model type')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index in the test dataset')
    parser.add_argument('--save_path', type=str, default='./cam_outputs/',
                        help='Path to save the CAM images')
    parser.add_argument('--type_of_transformation', type=str, default='MFCC',
                        choices=['MFCC', 'LFCC', 'delta', 'delta-delta', 'lfcc-delta', 'lfcc-delta-delta'],
                        help='Type of transformation')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs to extract')
    parser.add_argument('--length_in_seconds', type=int, default=5, help='Length of audio clips in seconds')
    parser.add_argument('--audio_dir', type=str, default='', help='Path to the test data directory')
    parser.add_argument('--model_weights', type=str, default='', help='Path to the saved model weights')
    parser.add_argument('--path_type', type=str, default='cluster', choices=['cluster', 'local'],
                        help='Path type: cluster or local')

    args = parser.parse_args()

    # Set paths based on the path type
    if args.path_type == 'cluster':
        main_path = "/home/zhazzouri/speech-music-classification-unet/"
        data_main_path = "/netscratch/zhazzouri/dataset/"
        experiments_path = "/netscratch/zhazzouri/experiments/"  # Path to the folder where the mlflow experiments are stored
    else:
        main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
        data_main_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
        experiments_path = "/Users/zainhazzouri/projects/Master-thesis-experiments/"  # Path to the folder where the mlflow experiments are stored

    # Set device
    device = get_device()

    # If model_weights path is not provided, construct it based on the model and parameters
    if args.model_weights == '':
        experiment_name = f"{args.model}_{args.type_of_transformation}_{args.n_mfcc}_len{args.length_in_seconds}S"
        model_weights_path = os.path.join(experiments_path, 'results', experiment_name,
                                          f"{experiment_name}.pth")
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_weights_path}. Please provide the correct path using --model_weights.")
    else:
        model_weights_path = args.model_weights
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_weights_path}. Please provide the correct path using --model_weights.")

    # Load the model
    models_dict = {
        "U_Net": U_Net(),
        "R2U_Net": R2U_Net(),
        "R2AttU_Net": R2AttU_Net(),
        "AttentionUNet": AttentionUNet()
    }
    model_name = args.model
    model = models_dict[model_name].to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # If audio_dir is not provided, use default path
    if args.audio_dir == '':
        args.audio_dir = os.path.join(data_main_path, 'test/')
        if not os.path.exists(args.audio_dir):
            raise FileNotFoundError(
                f"Test data not found at {args.audio_dir}. Please provide the correct path using --audio_dir.")

    # Load the dataset
    test_dataset = AudioProcessor(audio_dir=args.audio_dir, n_mfcc=args.n_mfcc,
                                  length_in_seconds=args.length_in_seconds,
                                  type_of_transformation=args.type_of_transformation)

    # Get the sample
    input_tensor, label = test_dataset[args.sample_idx]
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Get the target layer
    target_layers = get_target_layer(model, model_name)

    # Get all CAM methods
    cam_methods_dict = get_cam_methods()

    # Compute the output and get the predicted class
    output = model(input_tensor)
    _, predicted_class = output.max(dim=1)
    target_category = predicted_class.item()

    # Define the targets for CAM
    targets = [ClassifierOutputTarget(target_category)]

    # Prepare to plot
    num_methods = len(cam_methods_dict)
    num_cols = 4  # Number of columns in the plot grid
    num_rows = (num_methods + 1) // num_cols + 1  # Additional row for the original image

    plt.figure(figsize=(15, num_rows * 4))

    # Normalize the input image for visualization
    input_image = input_tensor.cpu().numpy()[0, 0, :, :]  # Assuming single channel
    input_image_norm = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    input_image_rgb = np.stack([input_image_norm]*3, axis=-1)

    # Plot the original image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(input_image_norm, cmap='viridis')
    plt.title('Original Input')
    plt.axis('off')

    # Iterate over CAM methods
    for idx, (method_name, cam_method_class) in enumerate(cam_methods_dict.items(), start=2):
        print(f"Processing {method_name}...")
        try:
            cam = cam_method_class(model=model, target_layers=target_layers, use_cuda=(device.type == 'cuda'))
            grayscale_cam = cam(input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]  # Get the CAM for the first (and only) sample

            # Overlay the CAM on the input image
            cam_image = show_cam_on_image(input_image_rgb, grayscale_cam, use_rgb=True)

            # Plot the CAM
            plt.subplot(num_rows, num_cols, idx)
            plt.imshow(cam_image)
            plt.title(method_name)
            plt.axis('off')

            # Save individual CAM images if needed
            os.makedirs(args.save_path, exist_ok=True)
            save_filename = f'{model_name}_{method_name}_sample_{args.sample_idx}.png'
            plt.imsave(os.path.join(args.save_path, save_filename), cam_image)
        except Exception as e:
            print(f"Failed to process {method_name}: {e}")
            plt.subplot(num_rows, num_cols, idx)
            plt.text(0.5, 0.5, f"Error\n{method_name}", horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

    plt.tight_layout()
    # Save the combined figure
    combined_save_filename = f'{model_name}_All_CAMs_sample_{args.sample_idx}.png'
    plt.savefig(os.path.join(args.save_path, combined_save_filename), dpi=300)
    # plt.show()