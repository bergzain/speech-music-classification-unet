#%% 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef,confusion_matrix
import torchaudio
import numpy as np
import pandas as pd




from cnn_model import R2AttU_Net

from datapreprocessing import AudioProcessor
#%%
# Training parameters
batch_size = 8
learning_rate = 1e-3 # 1e-4= 0.0001
num_epochs = 200
patience = 10 # for early stopping
save_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/results/AttentionR2UNet/MFCCs"

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
model_name = "R2AttU_Net"
model = R2AttU_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
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

#%%
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




            # Forward pass
            outputs = model(inputs)

            # Calculate SDR and append to list
            all_sdrs.append(calculate_sdr(targets, outputs))

            # Calculate loss
            loss = criterion(outputs, targets)

            # Update loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Store targets and predictions for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_sdr = sum(all_sdrs) / len(all_sdrs)


    # Calculate average loss and accuracy
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    # Calculate precision, recall, F1-score, and MCC
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_targets, all_predictions)

    return avg_loss, accuracy, precision, recall, f1_score, mcc, avg_sdr

    # Evaluate the model
    val_loss, val_accuracy, val_precision, val_recall, val_f1_score, val_mcc, avg_sdr = evaluate(val_loader, model, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-score: {val_f1_score:.4f}")
    print(f"Validation MCC: {val_mcc:.4f}")
    print(f"Validation SDR: {avg_sdr:.4f}")
    
    
#%%
# Initialize lists for storing loss and accuracy
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

# training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
        inputs = inputs.to(device)

        # targets = targets.to(device)
        targets = one_hot_encode(targets, num_classes=4, device=device).to(device)



        # Zero the parameter gradients
        optimizer.zero_grad() # zero the gradient buffers

        # Forward pass
        outputs = model(inputs)


        # print(f'outputs shape: {outputs.shape}')
        # print(f'targets shape: {targets.shape}')



        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update loss
        running_loss += loss.item()

        # Update total and correct
        _, predicted = torch.max(outputs.data, 1)
        _, targets = torch.max(targets, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Validate and store the validation loss and accuracy

    # Validate and store the validation metrics
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
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-score: {val_f1_score:.4f}")
    print(f"Validation MCC: {val_mcc:.4f}")
    
    
    ## early stopping
    # Compare with best validation loss
    if val_accuracy > best_val_Accuracy:
        best_val_Accuracy = val_accuracy
        no_improv_counter = 0
        best_epoch = epoch 
        # Save the best model using save_path variable
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")

    else:
        no_improv_counter += 1

    # If no improvement for 10 epochs, stop training
    if no_improv_counter >= patience:
        print(f"No improvement for {patience} epochs, stopping..")
        break

print("Training finished.")

#%%
def to_numpy(x):
    if isinstance(x, torch.Tensor): # if x is a torch tensor
        return x.cpu().numpy()
    else:
        return x

# Create a DataFrame from your metrics
metrics_df = pd.DataFrame({
    'val_losses': [to_numpy(x) for x in val_losses],
    'val_accuracies': [to_numpy(x) for x in val_accuracies],
    'val_precisions': [to_numpy(x) for x in val_precisions],
    'val_recalls': [to_numpy(x) for x in val_recalls],
    'val_f1_scores': [to_numpy(x) for x in val_f1_scores],
    'val_mccs': [to_numpy(x) for x in val_mccs],
    'val_sdrs': [to_numpy(x) for x in val_sdrs],
    'best_epoch': to_numpy(best_epoch)
})
# Save the DataFrame to a CSV file
metrics_df.to_csv(f"{save_path}/{model_name}_metrics.csv", index=False)

# After training, load the best model for further use
model.load_state_dict(torch.load(f'{save_path}/best_model.pth'))

#%%
# Plotting
fig, ax = plt.subplots()

# Loss
ax.plot(train_losses, label='Train')
ax.plot(val_losses, label='Validation')
ax.set_title(f'{model_name} Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_loss.png")

# Accuracy
fig, ax = plt.subplots()

ax.plot(train_accuracies, label='Train')
ax.plot(val_accuracies, label='Validation')
ax.set_title(f'{model_name} Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_accuracy.png")
# plt.show()


#%%


# Precision
fig, ax = plt.subplots()
ax.plot(val_precisions, label='Validation')
ax.set_title(f'{model_name} Precision')
ax.set_xlabel('Epoch')
ax.set_ylabel('Precision')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_precision.png")

# Recall
fig, ax = plt.subplots()
ax.plot(val_recalls, label='Validation')
ax.set_title(f'{model_name}_Recall')
ax.set_xlabel('Epoch')
ax.set_ylabel('Recall')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_recall.png")

# F1-score
fig, ax = plt.subplots()
ax.plot(val_f1_scores, label='Validation')
ax.set_title(f'{model_name} F1-Score')
ax.set_xlabel('Epoch')
ax.set_ylabel('F1-Score')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_f1_score.png")

# MCC
fig, ax = plt.subplots()
ax.plot(val_mccs, label='Validation')
ax.set_title(f'{model_name}_MCC')
ax.set_xlabel('Epoch')
ax.set_ylabel('MCC')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_mcc.png")




#%%
# Convert list of tensors to list of numpy arrays (if necessary)
if isinstance(val_sdrs[0], torch.Tensor):
    val_sdrs_np = [sdr.cpu().numpy() for sdr in val_sdrs]
else:
    val_sdrs_np = val_sdrs

# Plotting SDR
fig, ax = plt.subplots(figsize=(6, 4))

# SDR
ax.plot(val_sdrs_np, label='Validation')
ax.set_title(f'{model_name} Signal-to-Distortion Ratio (SDR)')
ax.set_xlabel('Epoch')
ax.set_ylabel('SDR')
ax.legend()
fig.savefig(f"{save_path}/{model_name}_sdr.png")

# plt.show()

#%%
def plot_confusion_matrix(y_true, y_pred, labels, ax=None, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

def get_weights_gradients(model):
    weights = []
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights.append(param.data.cpu().numpy())
            gradients.append(param.grad.data.cpu().numpy())
    return weights, gradients

def plot_histograms(weights, gradients, figsize=(10, 4)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for w in weights:
        ax1.hist(w.flatten(), bins=100, alpha=0.5)
    ax1.set_title("Weights Histogram")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")

    for g in gradients:
        ax2.hist(g.flatten(), bins=100, alpha=0.5)
    ax2.set_title("Gradients Histogram")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")

    plt.show()

# Get true labels and predictions on the validation set
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Confusion Matrix")
fig.savefig(f'{save_path}/{model_name}_confusion_matrix.png')


# Extract weights and gradients and plot histograms
weights, gradients = get_weights_gradients(model)
plot_histograms(weights, gradients)
fig.savefig(f'{save_path}/{model_name}_weights_gradients.png')

# %%
