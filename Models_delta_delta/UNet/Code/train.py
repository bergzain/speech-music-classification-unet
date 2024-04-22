import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch

from cnn_model import U_Net
from datapreprocessing import AudioProcessor

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("/Users/zainhazzouri/projects/Bachelor_Thesis/mlflow")
experiment_name = "UNet_Delta_Delta"
mlflow.set_experiment(experiment_name)
run_name = experiment_name + "1" 

# Training parameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 100
patience = 10
save_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/results/UNet/MFCCs_delta_delta"

# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device}")

path_to_train = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/train/"
path_to_test = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/"

train_dataset = AudioProcessor(audio_dir=path_to_train)
val_dataset = AudioProcessor(audio_dir=path_to_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_name = "U_Net"
model = U_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_sdr(target, prediction):
    target = target.float()
    prediction = prediction.float()

    target_energy = torch.sum(target**2)
    error_signal = target - prediction
    error_energy = torch.sum(error_signal**2)

    sdr = 10 * torch.log10(target_energy / error_energy)
    return sdr

def one_hot_encode(labels, num_classes, device):
    return torch.eye(num_classes, device=device)[labels]

def evaluate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_sdrs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = one_hot_encode(targets, num_classes=4, device=device).to(device)

            outputs = model(inputs)

            all_sdrs.append(calculate_sdr(targets, outputs))

            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_sdr = sum(all_sdrs) / len(all_sdrs)

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None)

    mcc = matthews_corrcoef(all_targets, all_predictions)

    return avg_loss, accuracy, precision, recall, f1_score, mcc, avg_sdr

with mlflow.start_run(run_name=run_name):
    # Log hyperparameters
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_mccs = []
    val_sdrs = []
    best_epoch = 0

    best_val_Accuracy = float('-inf') 
    no_improv_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
            inputs = inputs.to(device)
            targets = one_hot_encode(targets, num_classes=4, device=device).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        val_loss, val_accuracy, val_precision, val_recall, val_f1_score, val_mcc, val_sdr = evaluate(val_loader, model, criterion, device)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1_score)
        val_mccs.append(val_mcc)
        val_sdrs.append(val_sdr)

        print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

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
        mlflow.log_metric("val_sdr", val_sdr.item(), step=epoch)

        if val_accuracy > best_val_Accuracy:
            best_val_Accuracy = val_accuracy
            no_improv_counter = 0
            best_epoch = epoch 
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        else:
            no_improv_counter += 1

        if no_improv_counter >= patience:
            print(f"No improvement for {patience} epochs, stopping..")
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
        'val_sdrs': [to_numpy(x) for x in val_sdrs],
        'best_epoch': to_numpy(best_epoch)
    })
    metrics_df.to_csv(f"{save_path}/{model_name}metrics.csv", index=False)

    model.load_state_dict(torch.load(f'{save_path}/best_model.pth'))

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

    # Plotting SDR
    if isinstance(val_sdrs[0], torch.Tensor):
        val_sdrs_np = [sdr.cpu().numpy() for sdr in val_sdrs]
    else:
        val_sdrs_np = val_sdrs

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(val_sdrs_np, label='Validation')
    ax.set_title(f'{model_name} Signal-to-Distortion Ratio (SDR)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SDR')
    ax.legend()
    fig.savefig(f"{save_path}/{model_name}_sdr.png")
    mlflow.log_artifact(f"{save_path}/{model_name}_sdr.png")