#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch

from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image





from cnn_model import U_Net
from datapreprocessing import AudioProcessor
#%%
# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("/Users/zainhazzouri/projects/Bachelor_Thesis/mlflow")
experiment_name = "UNet_MFCCs"
mlflow.set_experiment(experiment_name)
run_name = experiment_name + " 150 ms seconds" + " learning decay"


# Training parameters
batch_size = 16 
learning_rate = 1e-3
num_epochs = 100
patience = 10
save_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/results/UNet/MFCCs"
#%%
# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
# device = "cpu"

print(f"Using {device}")
#%%
path_to_train = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/train/"
path_to_test = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/"

train_dataset = AudioProcessor(audio_dir=path_to_train)
val_dataset = AudioProcessor(audio_dir=path_to_test)
#%%
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%%

model_name = "U_Net"
model = U_Net(device=device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5, verbose=True)
#%%

def apply_cam_technique(cam_technique, model, target_layers, input_tensor, device, save_path):
    if cam_technique == "GradCAM":
        cam_extractor = GradCAM(model=model, target_layers=target_layers)
    elif cam_technique == "HiResCAM":
        cam_extractor = HiResCAM(model=model, target_layers=target_layers)
    elif cam_technique == "GradCAMElementWise":
        cam_extractor = GradCAMElementWise(model=model, target_layers=target_layers)
    elif cam_technique == "GradCAMPlusPlus":
        cam_extractor = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_technique == "XGradCAM":
        cam_extractor = XGradCAM(model=model, target_layers=target_layers)
    elif cam_technique == "AblationCAM":
        cam_extractor = AblationCAM(model=model, target_layers=target_layers)
    elif cam_technique == "ScoreCAM":
        cam_extractor = ScoreCAM(model=model, target_layers=target_layers)
    elif cam_technique == "EigenCAM":
        cam_extractor = EigenCAM(model=model, target_layers=target_layers)
    elif cam_technique == "EigenGradCAM":
        cam_extractor = EigenGradCAM(model=model, target_layers=target_layers)
    elif cam_technique == "LayerCAM":
        cam_extractor = LayerCAM(model=model, target_layers=target_layers)
    elif cam_technique == "FullGrad":
        cam_extractor = FullGrad(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unknown CAM technique: {cam_technique}")

    input_tensor = input_tensor.requires_grad_(True)  # Ensure the input tensor requires gradients
    cam_output = cam_extractor(input_tensor=input_tensor)  # Generate the Grad-CAM output

    input_image = input_tensor[0].permute(1, 2, 0).cpu().detach().numpy()
    input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    cam_output_image = cam_output[0]  # No need for .cpu().detach().numpy() as it's already a numpy array

    visualized_img = show_cam_on_image(input_image, cam_output_image, use_rgb=True)

    # Save the visualized image
    plt.imsave(f"{save_path}/{cam_technique}_visualization.png", visualized_img)
    plt.imshow(visualized_img)
    plt.title(f"{cam_technique} Visualization")
    plt.show()


#%%
# Load the best model
# model.init_fc_layer()  # Initialize the fully connected layer correctly.

model.load_state_dict(torch.load(f'{save_path}/best_model.pth'),strict=False)
#%%
# Create an instance of the Grad-CAM technique you want to use
target_layers = [model.Conv_1x1]  # Adjust based on the layer you want to target
cam_extractor = GradCAM(model=model, target_layers=target_layers)

#%%
# Prepare the input data
input_tensor = next(iter(val_loader))[0].to(device)  # Get a batch of input data
#%%
# Generate the Grad-CAM visualization
input_tensor = input_tensor.requires_grad_(True)  # Ensure the input tensor requires gradients
cam_output = cam_extractor(input_tensor=input_tensor)  # Generate the Grad-CAM output
# # %%
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

for cam_technique in cam_techniques:
    apply_cam_technique(cam_technique, model, target_layers, input_tensor, device, save_path)
# %%
