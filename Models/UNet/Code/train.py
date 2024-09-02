#%%
import os
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
import argparse

from models import U_Net,R2U_Net,R2AttU_Net,AttentionUNet
from datapreprocessing import AudioProcessor
#%%

#%%
# Parse arguments
parser = argparse.ArgumentParser(description='Train different versions of UNet models.')
parser.add_argument('--model', type=str, default='U_Net', choices=['U_Net', 'R2U_Net', 'R2AttU_Net', 'AttentionUNet'], help='Model type to train')
parser.add_argument('--type_of_transformation',default='MFCC',type=str, required=True, choices=['MFCC', 'LFCC', 'delta', 'delta-delta', 'lfcc-delta', 'lfcc-delta-delta'], help='Type of transformation')
parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs to extract')
parser.add_argument('--length_in_seconds', type=int, default=5, help='Length of audio clips in seconds')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--path_type', type=str, default='cluster', choices=['cluster', 'local'], help='Path type: cluster or local')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')  # Added argument for num_workers

args = parser.parse_args()

# Set paths based on the path type
if args.path_type == 'cluster':
    main_path = "/home/zhazzouri/speech-music-classification-unet/"
    data_main_path = "/netscratch/zhazzouri/dataset/"
    experiments_path = "/netscratch/zhazzouri/experiments/" # path to the folder where the mlflow experiments are stored
else:
    main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
    data_main_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
    experiment_path = main_path # path to the folder where the mlflow experiments are stored
    
#%%
# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri(experiments_path+ "/mlflow")
experiment_name = f"{args.model}_{args.type_of_transformation}_{args.n_mfcc}_len{args.length_in_seconds}S"

mlflow.set_experiment(experiment_name)
run_name = experiment_name 

# Set save path and create directory if it doesn't exist
save_path = os.path.join(experiments_path, "results", experiment_name) # main_path/results/experiment_name_folder/
os.makedirs(save_path, exist_ok=True)

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
path_to_train = data_main_path + "train/"
path_to_test =  data_main_path + "test/"

#%%
# Create datasets
train_dataset = AudioProcessor(audio_dir=path_to_train, n_mfcc=args.n_mfcc, length_in_seconds=args.length_in_seconds, type_of_transformation=args.type_of_transformation)
val_dataset = AudioProcessor(audio_dir=path_to_test, n_mfcc=args.n_mfcc, length_in_seconds=args.length_in_seconds, type_of_transformation=args.type_of_transformation)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

#%%
models = {
    "U_Net": U_Net(),
    "R2U_Net": R2U_Net(),
    "R2AttU_Net": R2AttU_Net(),
    "AttentionUNet": AttentionUNet()
}
model_name = args.model
model = models[model_name].to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Initialize the ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5, verbose=True)
#%%

def one_hot_encode(labels, num_classes, device):
    return torch.eye(num_classes, device=device)[labels]

def evaluate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # targets = one_hot_encode(targets, num_classes=2, device=device).to(device)

            outputs = model(inputs)


            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1) # 
            # _, targets = torch.max(targets, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            


    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None)

    mcc = matthews_corrcoef(all_targets, all_predictions)

    return avg_loss, accuracy, precision, recall, f1_score, mcc
#%%

with mlflow.start_run(run_name=run_name):
    # Log hyperparameters
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("num_epochs", args.num_epochs)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("n_mfcc", args.n_mfcc)
    mlflow.log_param("length_in_seconds", args.length_in_seconds)

    
    

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_mccs = []
    best_epoch = 0

    best_val_Accuracy = float('-inf') 
    no_improv_counter = 0

    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch+1}/{args.num_epochs}")

        model.train()
        running_loss = 0.0
        correct = 0 
        total = 0

        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
            inputs = inputs.to(device)
            # targets = one_hot_encode(targets, num_classes=2, device=device).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # _, targets = torch.max(targets, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_loss, val_accuracy, val_precision, val_recall, val_f1_score, val_mcc = evaluate(val_loader, model, criterion, device)
        # Update the learning rate scheduler "learnung rate decay "
        scheduler.step(val_loss)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1_score)
        val_mccs.append(val_mcc)

        print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
        
        # Log the updated learning rate
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        
        # Log metrics for each epoch
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_music_precision", val_precision[0], step=epoch)
        mlflow.log_metric("val_speech_precision", val_precision[1], step=epoch)
        mlflow.log_metric("val_music_recall", val_recall[0], step=epoch)
        mlflow.log_metric("val_speech_recall", val_recall[1], step=epoch)
        mlflow.log_metric("val_music_f1_score", val_f1_score[0], step=epoch)
        mlflow.log_metric("val_speech_f1_score", val_f1_score[1], step=epoch)
        mlflow.log_metric("val_mcc", val_mcc, step=epoch)

        if val_accuracy > best_val_Accuracy:
            best_val_Accuracy = val_accuracy
            no_improv_counter = 0
            best_epoch = epoch 
            torch.save(model.state_dict(), f"{save_path}/{experiment_name}.pth")
        else:
            no_improv_counter += 1

        if no_improv_counter >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping..")
            break

    print("Training finished.")

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return x

    metrics_df = pd.DataFrame({
        'val_losses': [to_numpy(x) for x in val_losses],
        'val_accuracies': [to_numpy(x) for x in val_accuracies],
        'music_val_precisions': [to_numpy(x[0]) for x in val_precisions],
        'speech_val_precisions': [to_numpy(x[1]) for x in val_precisions],
        'music_val_recalls': [to_numpy(x[0]) for x in val_recalls],
        'speech_val_recalls': [to_numpy(x[1]) for x in val_recalls],
        'music_val_f1_scores': [to_numpy(x[0]) for x in val_f1_scores],
        'speech_val_f1_scores': [to_numpy(x[1]) for x in val_f1_scores],
        'val_mccs': [to_numpy(x) for x in val_mccs],
        'best_epoch': to_numpy(best_epoch)
    })
    metrics_df.to_csv(f"{save_path}/{model_name}metrics.csv", index=False)

    # model.load_state_dict(torch.load(f'{save_path}/best_model.pth'))
    checkpoint = torch.load(f'{save_path}/{experiment_name}.pth')
    # model_dict = model.state_dict()
    # checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    # model_dict.update(checkpoint)
    # model.load_state_dict(model_dict)

    # Log the best model artifact
    mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)

    # Log other artifacts
    mlflow.log_artifact(f"{save_path}/{model_name}metrics.csv")

    # Plotting loss
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Validation')
    ax.set_title(f'{model_name} Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_loss.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_loss.png")

    # Plotting accuracy
    fig, ax = plt.subplots()
    ax.plot(train_accuracies, label='Train')
    ax.plot(val_accuracies, label='Validation')
    ax.set_title(f'{model_name} Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_accuracy.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_accuracy.png")

    music_precisions = [x[0] for x in val_precisions]
    speech_precisions = [x[1] for x in val_precisions]
    music_recalls = [x[0] for x in val_recalls]
    speech_recalls = [x[1] for x in val_recalls]
    music_f1_scores = [x[0] for x in val_f1_scores]
    speech_f1_scores = [x[1] for x in val_f1_scores]

    # Plotting Precision for each class
    fig, ax = plt.subplots()
    ax.plot(music_precisions, label='Music Precision')
    ax.plot(speech_precisions, label='Speech Precision')
    ax.set_title(f'{model_name} Precision by Class')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_precision_by_class.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_precision_by_class.png")

    # Plotting Recall for each class
    fig, ax = plt.subplots()
    ax.plot(music_recalls, label='Music Recall')
    ax.plot(speech_recalls, label='Speech Recall')
    ax.set_title(f'{model_name} Recall by Class')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_recall_by_class.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_recall_by_class.png")

    # Plotting F1-Score for each class
    fig, ax = plt.subplots()
    ax.plot(music_f1_scores, label='Music F1-Score')
    ax.plot(speech_f1_scores, label='Speech F1-Score')
    ax.set_title(f'{model_name} F1-Score by Class')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_f1_score_by_class.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_f1_score_by_class.png")

    # Plotting MCC
    fig, ax = plt.subplots()
    ax.plot(val_mccs, label='Validation')
    ax.set_title(f'{model_name}_MCC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MCC')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_mcc.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_mcc.png")

# %%
