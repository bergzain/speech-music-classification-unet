#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import librosa
import librosa.display
import random
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import zoom

from cnn_model import AttentionUNet
from datapreprocessing import AudioProcessor
#%%

# set Mlflow tracking uri and  experiment name
mlflow.set_tracking_uri("/Users/zainhazzouri/projects/Bachelor_Thesis/mlflow")
experiment_name = "AttentionUNet_MFCCs"
mlflow.set_experiment(experiment_name)
#%%
# Training parameters
target_sample_rate = 44100  # Define your target sample rate

batch_size = 8
learning_rate = 1e-3 # 1e-4= 0.0001
num_epochs = 100
patience = 10 # for early stopping
save_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/results/Attention_UNet/MFCCs"

#%%
# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():  # if you have apple silicon mac
    device = "mps"  # if it doesn't work try device = torch.device('mps')
else:
    device = "cpu"
print(f"Using {device}")

#%%
path_to_train = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/train/"
path_to_test = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/"

train_dataset = AudioProcessor(audio_dir=path_to_train)
val_dataset = AudioProcessor(audio_dir=path_to_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Initialize model, loss, and optimizer
model_name = "AttentionUNet"
model = AttentionUNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the best model
model.load_state_dict(torch.load(f'{save_path}/best_model.pth'), strict=False)


# Generate the Grad-CAM visualization
cam_techniques = [
        "GradCAM",
        "HiResCAM",
        "GradCAMElementWise",
        "GradCAMPlusPlus",
        "XGradCAM",
        "AblationCAM",
        "ScoreCAM",
        # "EigenCAM",
        # "EigenGradCAM",
        "LayerCAM",
        "FullGrad"
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
        # "EigenCAM": EigenCAM,
        # "EigenGradCAM": EigenGradCAM,
        "LayerCAM": LayerCAM,
        "FullGrad": FullGrad
    }
    cam = cam_dict[cam_technique](model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


# Enhanced Plotting Function with Librosa and Side-by-Side Comparison
def plot_cam_side_by_side_with_librosa(original, cam_image, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalize CAM image
    cam_image = (cam_image - np.min(cam_image)) / (np.max(cam_image) - np.min(cam_image))
    
    # Squeeze the original if needed
    if original.shape[0] == 1:
        original = original.squeeze(0)
    
    # Resize the CAM image to match the original MFCC dimensions using zoom
    zoom_factors = (original.shape[0] / cam_image.shape[0], original.shape[1] / cam_image.shape[1])
    cam_image_resized = zoom(cam_image, zoom_factors)
    
    # Plot original MFCC using librosa
    img = librosa.display.specshow(original, sr=target_sample_rate, x_axis='time', ax=axes[0], cmap='magma')
    fig.colorbar(img, ax=axes[0], format="%+2.f dB")
    axes[0].set_title("Original MFCC")
    
    # Plot MFCC with GradCAM overlay
    img = axes[1].imshow(original, cmap='magma', aspect='auto', origin='lower')
    axes[1].imshow(cam_image_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
    fig.colorbar(img, ax=axes[1], format="%+2.f dB")
    axes[1].set_title(f"{title} CAM")
    
    plt.suptitle(title)
    plt.savefig(f"{save_path}/{title}_cam.png")
    plt.show()


# Main processing function
model.eval()
# 
target_layers = [model.Conv5] # The last convolutional block before upsampling 
# target_layers = [model.Up_conv5] # The first upsampling layer 



# Process and visualize a random sample
random_idx = random.randint(0, len(val_loader.dataset) - 1)
random_sample, _ = val_loader.dataset[random_idx]
input_tensor = random_sample.unsqueeze(0).to(device)  # Convert to batch format


for cam_technique in cam_techniques:
    grayscale_cam = apply_cam_technique(cam_technique, model, target_layers, input_tensor)
    plot_cam_side_by_side_with_librosa(input_tensor[0].cpu().numpy(), grayscale_cam, cam_technique, save_path)# %%

# %%
