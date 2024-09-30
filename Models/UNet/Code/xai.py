#%%
import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import random
try:
    from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, LayerCAM, FullGrad, EigenCAM, EigenGradCAM
except AttributeError as e:
    print(f"AttributeError: {e}")
except ImportError as e:
    print(f"ImportError: {e}")
    
from scipy.ndimage import zoom

from datapreprocessing import AudioProcessor
from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet


# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device}")


# Parse arguments
parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for different UNet models.')
parser.add_argument('--model', type=str, default='U_Net', choices=['U_Net', 'R2U_Net', 'R2AttU_Net', 'AttentionUNet'], help='Model type to use')
parser.add_argument('--type_of_transformation', default='MFCC', type=str, required=True, choices=['MFCC', 'LFCC', 'delta', 'delta-delta', 'lfcc-delta', 'lfcc-delta-delta'], help='Type of transformation')
parser.add_argument('--n_mfcc', type=int, default=32, help='Number of MFCCs to extract')
parser.add_argument('--length_in_seconds', type=int, default=5, help='Length of audio clips in seconds')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')
parser.add_argument('--path_type', type=str, default='cluster', choices=['cluster', 'local'], help='Path type: cluster or local')
parser.add_argument('--random_seed', type=int, default=23, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

target_sample_rate = 44100


# Set paths based on the path type
if args.path_type == 'cluster':
    main_path = "/home/zhazzouri/speech-music-classification-unet/"
    data_main_path = "/netscratch/zhazzouri/dataset/"
    experiments_path = "/netscratch/zhazzouri/experiments/" # path to the folder where the mlflow experiments are stored
else:
    main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
    data_main_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
    experiments_path = "/Users/zainhazzouri/projects/Master-thesis-experiments/" # path to the folder where the mlflow experiments are stored

experiment_name = f"{args.model}_{args.type_of_transformation}_{args.n_mfcc}_len{args.length_in_seconds}S"

save_path = os.path.join(experiments_path, "results", experiment_name) # main_path/results/experiment_name_folder/
os.makedirs(save_path, exist_ok=True)




path_to_test = os.path.join(data_main_path, "test/")

# Create dataset and DataLoader
val_dataset = AudioProcessor(audio_dir=path_to_test, n_mfcc=args.n_mfcc, length_in_seconds=args.length_in_seconds, type_of_transformation=args.type_of_transformation)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Load model
models = {
    "U_Net": U_Net(),
    "R2U_Net": R2U_Net(),
    "R2AttU_Net": R2AttU_Net(),
    "AttentionUNet": AttentionUNet()
}
model_name = args.model
model = models[model_name].to(device)

# Load the best model
model.load_state_dict(torch.load(f'{save_path}/{args.model}_{args.type_of_transformation}_{args.n_mfcc}_len{args.length_in_seconds}S.pth', map_location=device), strict=False)

# Generate the Grad-CAM visualization
cam_techniques = [
    "GradCAM",
    "HiResCAM",
    "GradCAMElementWise",
    "GradCAMPlusPlus",
    "XGradCAM",
    "AblationCAM",
    "ScoreCAM",
    "LayerCAM",
    "FullGrad",
    "EigenCAM",
    "EigenGradCAM"
]

# Grad-CAM Application Function with Normalization and Smoothing
def apply_cam_technique(cam_technique, model, target_layers, input_tensor):
    cam_dict = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "GradCAMElementWise": GradCAMElementWise,
        "GradCAMPlusPlus": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "AblationCAM": AblationCAM,
        "ScoreCAM": ScoreCAM,
        "LayerCAM": LayerCAM,
        "FullGrad": FullGrad,
        "EigenCAM": EigenCAM,
        "EigenGradCAM": EigenGradCAM,
    }
    cam = cam_dict[cam_technique](model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam

# Enhanced Plotting Function with Librosa and Side-by-Side Comparison
def plot_cam_side_by_side_with_librosa(original, cam_images, titles, save_path, sample_type):
    num_cam_techniques = len(cam_images)
    fig, axes = plt.subplots(1, num_cam_techniques + 1, figsize=((num_cam_techniques + 1) * 5, 5))
    
    # Reshape the original data
    original = original.squeeze()
    
    # Plot original MFCC using librosa on the far left
    img = librosa.display.specshow(original, sr=target_sample_rate, x_axis='time', ax=axes[0], cmap='magma')
    fig.colorbar(img, ax=axes[0], format="%+2.f dB")
    axes[0].set_title(f"Original MFCC ({sample_type})")
    
    for i, (cam_image, title) in enumerate(zip(cam_images, titles), start=1):
        # Normalize CAM image
        cam_image = (cam_image - np.min(cam_image)) / (np.max(cam_image) - np.min(cam_image))
        cam_image = np.clip(cam_image, 0, 1)
        # Resize the CAM image to match the original MFCC dimensions using zoom
        zoom_factors = (original.shape[0] / cam_image.shape[0], original.shape[1] / cam_image.shape[1])
        cam_image_resized = zoom(cam_image, zoom_factors)
        
        # Plot MFCC with GradCAM overlay
        img = axes[i].imshow(original, cmap='magma', aspect='auto', origin='lower')
        axes[i].imshow(cam_image_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
        fig.colorbar(img, ax=axes[i], format="%+2.f dB")
        axes[i].set_title(f"{title} CAM ({sample_type})")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_{sample_type}_cam.png")
    plt.show()

# Main processing function
model.eval()
target_layers = [model.Conv_1x1] # the final layer

# Process and visualize a random music sample and a random speech sample
music_sample = None
speech_sample = None

music_indices = [i for i, (_, label) in enumerate(val_loader.dataset) if label == 0]
speech_indices = [i for i, (_, label) in enumerate(val_loader.dataset) if label == 1]

if len(music_indices) > 0 and len(speech_indices) > 0:
    music_idx = random.choice(music_indices)
    speech_idx = random.choice(speech_indices)
    
    music_sample = val_loader.dataset[music_idx][0]
    speech_sample = val_loader.dataset[speech_idx][0]
else:
    raise ValueError("Could not find both music and speech samples in the dataset.")

music_tensor = music_sample.unsqueeze(0).to(device)
speech_tensor = speech_sample.unsqueeze(0).to(device)

music_cam_images = []
music_titles = []
speech_cam_images = []
speech_titles = []

for cam_technique in cam_techniques:
    music_grayscale_cam = apply_cam_technique(cam_technique, model, target_layers, music_tensor)
    speech_grayscale_cam = apply_cam_technique(cam_technique, model, target_layers, speech_tensor)
    
    music_cam_images.append(music_grayscale_cam)
    music_titles.append(cam_technique)
    speech_cam_images.append(speech_grayscale_cam)
    speech_titles.append(cam_technique)

plot_cam_side_by_side_with_librosa(music_tensor[0].cpu().numpy(), music_cam_images, music_titles, save_path, "music")
plot_cam_side_by_side_with_librosa(speech_tensor[0].cpu().numpy(), speech_cam_images, speech_titles, save_path, "speech")
# %%
