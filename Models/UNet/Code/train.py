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
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import argparse
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import json

from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet
from datapreprocessing import AudioProcessor
from mlflow_utils import MLflowExperimentManager
from mlflow_config import MLflowConfig
#%%
class TrainingConfig:
    def __init__(self, args: argparse.Namespace):
        self.model_type = args.model
        self.feature_type = args.type_of_transformation
        self.n_mfcc = args.n_mfcc
        self.length_in_seconds = args.length_in_seconds
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.patience = args.patience
        self.num_workers = args.num_workers
        
        # Set paths based on args.path_type
        if args.path_type == 'cluster':
            self.main_path = "/home/zhazzouri/speech-music-classification-unet/"
            self.data_path = "/netscratch/zhazzouri/dataset/"
            self.experiments_path = "/netscratch/zhazzouri/experiments/"
        else:
            self.main_path = "/Users/zainhazzouri/projects/Bachelor_Thesis/"
            self.data_path = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/"
            self.experiments_path = "/Users/zainhazzouri/projects/Master-thesis-experiments/"
        
        self.experiment_name = f"{self.model_type}_{self.feature_type}_{self.n_mfcc}_len{self.length_in_seconds}S"
        self.save_path = os.path.join(self.experiments_path, "results", self.experiment_name)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self.get_device()
        self.mlflow_manager = MLflowExperimentManager(config.experiments_path + "/mlflow")
        
        # Initialize models, datasets, and optimizers
        self.setup_training_components()
        
    def get_device() -> str:
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_built():
            return "mps"
        return "cpu"
        
    def setup_training_components(self):
        # Initialize model
        models = {
            "U_Net": U_Net,
            "R2U_Net": R2U_Net,
            "R2AttU_Net": R2AttU_Net,
            "AttentionUNet": AttentionUNet
        }
        self.model = models[self.config.model_type]().to(self.device)
        
        # Setup datasets
        self.train_dataset = AudioProcessor(
            audio_dir=os.path.join(self.config.data_path, "train"),
            n_mfcc=self.config.n_mfcc,
            length_in_seconds=self.config.length_in_seconds,
            type_of_transformation=self.config.feature_type
        )
        
        self.val_dataset = AudioProcessor(
            audio_dir=os.path.join(self.config.data_path, "test"),
            n_mfcc=self.config.n_mfcc,
            length_in_seconds=self.config.length_in_seconds,
            type_of_transformation=self.config.feature_type
        )
        
        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average=None
        )
        mcc = matthews_corrcoef(all_targets, all_predictions)
        accuracy = 100 * np.mean(np.array(all_targets) == np.array(all_predictions))
        avg_loss = running_loss / len(loader)
        
        return avg_loss, accuracy, precision, recall, f1_score, mcc
    
    def log_epoch_metrics(self, metrics: Dict[str, float], epoch: int, prefix: str = ""):
        for name, value in metrics.items():
            mlflow.log_metric(f"{prefix}{name}", value, step=epoch)
    
    def save_training_plots(self, train_metrics: Dict[str, list], val_metrics: Dict[str, list]):
        # Plot and save training curves
        for metric in ['loss', 'accuracy']:
            plt.figure(figsize=(10, 6))
            plt.plot(train_metrics[metric], label=f'Train {metric}')
            plt.plot(val_metrics[metric], label=f'Validation {metric}')
            plt.title(f'{self.config.model_type} {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(self.config.save_path, f'{metric}_curve.png')
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
    
    def train(self):
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.experiment_name) as run:
            # Log parameters
            mlflow.log_params({
                "model_type": self.config.model_type,
                "feature_type": self.config.feature_type,
                "n_mfcc": self.config.n_mfcc,
                "length_in_seconds": self.config.length_in_seconds,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "patience": self.config.patience,
                "device": self.device,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__
            })
            
            # Training setup
            best_val_accuracy = float('-inf')
            no_improve_count = 0
            train_metrics = {'loss': [], 'accuracy': []}
            val_metrics = {'loss': [], 'accuracy': []}
            training_start_time = time.time()
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Training step
                for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                # Calculate training metrics
                train_loss = running_loss / len(self.train_loader)
                train_accuracy = 100 * correct / total
                train_metrics['loss'].append(train_loss)
                train_metrics['accuracy'].append(train_accuracy)
                
                # Validation step
                val_loss, val_accuracy, val_precision, val_recall, val_f1, val_mcc = self.evaluate(self.val_loader)
                val_metrics['loss'].append(val_loss)
                val_metrics['accuracy'].append(val_accuracy)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Log metrics
                epoch_metrics = {
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_music_precision': val_precision[0],
                    'val_speech_precision': val_precision[1],
                    'val_music_recall': val_recall[0],
                    'val_speech_recall': val_recall[1],
                    'val_music_f1': val_f1[0],
                    'val_speech_f1': val_f1[1],
                    'val_mcc': val_mcc,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': time.time() - epoch_start_time
                }
                self.log_epoch_metrics(epoch_metrics, epoch)
                
                # Model checkpoint
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    no_improve_count = 0
                    
                    # Save model
                    model_path = os.path.join(self.config.save_path, f"{self.config.experiment_name}.pth")
                    torch.save(self.model.state_dict(), model_path)
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(
                        self.model,
                        "model",
                        registered_model_name=self.config.model_type
                    )
                else:
                    no_improve_count += 1
                
                # Early stopping
                if no_improve_count >= self.config.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Log progress
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
            
            # End of training
            training_time = time.time() - training_start_time
            mlflow.log_metric("total_training_time", training_time)
            
            # Save and log final plots
            self.save_training_plots(train_metrics, val_metrics)
            
            # Log final model performance
            mlflow.log_metric("best_val_accuracy", best_val_accuracy)
            
            # Save experiment metadata
            metadata = {
                "best_val_accuracy": best_val_accuracy,
                "total_epochs": epoch + 1,
                "training_time": training_time,
                "early_stopped": no_improve_count >= self.config.patience
            }
            
            with open(os.path.join(self.config.save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            
            mlflow.log_artifact(os.path.join(self.config.save_path, "metadata.json"))

def main():
    parser = argparse.ArgumentParser(description='Train different versions of UNet models.')
    parser.add_argument('--model', type=str, default='U_Net',
                      choices=['U_Net', 'R2U_Net', 'R2AttU_Net', 'AttentionUNet'])
    parser.add_argument('--type_of_transformation', type=str, required=True,
                      choices=['MFCC', 'LFCC', 'delta', 'delta-delta', 'lfcc-delta', 'lfcc-delta-delta'])
    parser.add_argument('--n_mfcc', type=int, default=32)
    parser.add_argument('--length_in_seconds', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--path_type', type=str, default='cluster',
                      choices=['cluster', 'local'])
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(args)
    
    # Initialize and run trainer
    trainer = Trainer(config)
    # print using device 
    print("using device:", trainer.device)
    trainer.train()

if __name__ == "__main__":
    main()