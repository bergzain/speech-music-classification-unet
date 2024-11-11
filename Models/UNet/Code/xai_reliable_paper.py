import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torchaudio
import json
import logging
import traceback
from datetime import datetime
from torch.utils.data import DataLoader
from datapreprocessing import AudioProcessor
from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet
import imageio
from typing import Dict, List, Tuple, Any, Optional
import warnings
import re
warnings.filterwarnings('ignore')




def parse_model_filename(model_path):
    filename = os.path.splitext(os.path.basename(model_path))[0]  # Remove extension if any
    pattern = r'^(.*?)_([^_]+(?:-[^_]+)*)_(\d+)_len([\d\.]+)S$'
    match = re.match(pattern, filename)
    if match:
        model_type = match.group(1)
        feature_type = match.group(2)
        n_mfcc = int(match.group(3))
        length_in_seconds = float(match.group(4))
        return model_type, feature_type, n_mfcc, length_in_seconds
    else:
        raise ValueError(f"Model filename '{filename}' does not match the expected pattern.")
    
class BaseCAM:
    """Base class for all CAM techniques"""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        def save_gradient(grad):
            self.gradients = grad.detach()
        
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(lambda m, grad_in, grad_out: save_gradient(grad_out[0]))

class VanillaGradCAM(BaseCAM):
    """Original Grad-CAM implementation"""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        with torch.no_grad():
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:], 
                                  mode='bilinear', align_corners=False)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, output

class GradCAMPlusPlus(BaseCAM):
    """Grad-CAM++ implementation"""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        with torch.no_grad():
            alpha_num = self.gradients.pow(2)
            alpha_denom = alpha_num.mul(2) + \
                         self.activations.mul(self.gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
            alpha = alpha_num.div(alpha_denom + 1e-7)
            
            weights = (alpha * F.relu(self.gradients)).sum(dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:],
                                  mode='bilinear', align_corners=False)
            heatmap = F.relu(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, output

class ScoreCAM(BaseCAM):
    """Score-CAM implementation"""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        with torch.no_grad():
            b, c, h, w = self.activations.shape
            
            acts = self.activations
            acts = F.interpolate(acts, size=input_tensor.shape[2:],
                               mode='bilinear', align_corners=False)
            
            scores = torch.zeros((b, c), device=input_tensor.device)
            
            for i in range(c):
                act_mask = acts[:, i:i+1, :, :]
                act_mask = (act_mask - act_mask.min()) / (act_mask.max() - act_mask.min() + 1e-8)
                masked_input = input_tensor * act_mask
                score = self.model(masked_input)
                scores[:, i] = score[:, class_idx]
            
            scores = F.softmax(scores, dim=1)
            cam = torch.sum(scores.view(b, c, 1, 1) * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, output
    
class LayerCAM(BaseCAM):
    """LayerCAM implementation
    
    LayerCAM computes the weighted combination of forward activation maps 
    for a specific class. Unlike GradCAM, it preserves fine-grained details
    by using positive gradients at each activation position.
    """
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        with torch.no_grad():
            # Get positive gradients only
            positive_gradients = F.relu(self.gradients)
            
            # Element-wise multiplication of feature maps with positive gradients
            weighted_activations = self.activations * positive_gradients
            
            # Global average pooling over spatial dimensions
            weights = torch.mean(positive_gradients, dim=(2, 3), keepdim=True)
            
            # Weight the activation maps and sum
            heatmap = torch.sum(weighted_activations * weights, dim=1, keepdim=True)
            
            # Apply ReLU to highlight positive contributions
            heatmap = F.relu(heatmap)
            
            # Interpolate to input size
            heatmap = F.interpolate(
                heatmap,
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Normalize to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, output


    
def compute_feature(y: np.ndarray, sr: int, feature_type: str = 'LFCC', n_mfcc: int = 32,
                    n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048,
                    length_in_seconds: float = 5.0) -> Tuple[np.ndarray, str]:
    """Compute audio features with specified parameters and ensure consistency in feature dimensions."""
    # Ensure the audio signal y is the correct length
    desired_length = int(sr * length_in_seconds)
    if len(y) > desired_length:
        y = y[:desired_length]
    else:
        y = np.pad(y, (0, max(0, desired_length - len(y))), 'constant')

    # Calculate expected number of frames (time steps)
    expected_time_steps = int(np.ceil((desired_length - win_length) / hop_length)) + 1

    # Compute features based on the specified type
    if feature_type == 'MFCC':
        feature = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        title = 'MFCC'
    elif feature_type == 'LFCC':
        waveform = torch.FloatTensor(y)
        transform = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_lfcc=n_mfcc,
            speckwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length
            }
        )
        feature = transform(waveform).numpy()
        title = 'LFCC'
    elif feature_type == 'delta':
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        feature = librosa.feature.delta(mfcc)
        title = 'Delta MFCC'
    elif feature_type == 'delta-delta':
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        feature = librosa.feature.delta(mfcc, order=2)
        title = 'Delta-Delta MFCC'
    elif feature_type == 'lfcc-delta':
        waveform = torch.FloatTensor(y)
        transform = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_lfcc=n_mfcc,
            speckwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length
            }
        )
        lfcc = transform(waveform)
        feature = librosa.feature.delta(lfcc.numpy())
        title = 'Delta LFCC'
    elif feature_type == 'lfcc-delta-delta':
        waveform = torch.FloatTensor(y)
        transform = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_lfcc=n_mfcc,
            speckwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length
            }
        )
        lfcc = transform(waveform)
        feature = librosa.feature.delta(lfcc.numpy(), order=2)
        title = 'Delta-Delta LFCC'
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    # Ensure the feature has the expected number of time steps
    feature = pad_or_trim_feature(feature, expected_time_steps)

    return feature, title

def pad_or_trim_feature(feature: np.ndarray, expected_time_steps: int) -> np.ndarray:
    """Pad or trim the feature to have the expected number of time steps."""
    current_time_steps = feature.shape[1]
    if current_time_steps > expected_time_steps:
        feature = feature[:, :expected_time_steps]
    elif current_time_steps < expected_time_steps:
        padding = expected_time_steps - current_time_steps
        feature = np.pad(feature, ((0, 0), (0, padding)), mode='constant')
    return feature

def analyze_audio_features(y: np.ndarray, sr: int, output_dir: str, 
                           class_name: str, sample_num: int,
                           n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048) -> Dict[str, np.ndarray]:
    """Analyze audio features and generate visualizations"""
    features = {
        'zcr': librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=win_length),
        'rms': librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length),
        'spectral_centroids': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(features['zcr'][0])
    ax1.set_title('Zero Crossing Rate')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('ZCR')
    
    ax2.plot(features['rms'][0])
    ax2.set_title('RMS Energy')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Energy')
    
    ax3.plot(features['spectral_centroids'][0])
    ax3.set_title('Spectral Centroid')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Frequency (Hz)')
    
    ax4.plot(features['spectral_rolloff'][0])
    ax4.set_title('Spectral Rolloff')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_sample_{sample_num}_features.png'))
    plt.close()
    
    return features

def visualize_sample_with_features(file_path: str, save_dir: str, class_name: str, 
                                   sample_num: int, feature_type: str = 'LFCC', 
                                   n_mfcc: int = 32, sr: int = 44100,
                                   length_in_seconds: float = 5.0,
                                   n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048) -> Tuple[np.ndarray, int]:
    """Create comprehensive visualization of audio sample with features"""
    # Load audio
    y, _ = librosa.load(file_path, sr=sr)
    
    # Trim or pad the audio to the desired length
    desired_length = int(sr * length_in_seconds)
    if len(y) > desired_length:
        y = y[:desired_length]
    else:
        y = np.pad(y, (0, max(0, desired_length - len(y))), 'constant')
    
    # Proceed with visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Waveform
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Feature representation
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    feature, title = compute_feature(y, sr, feature_type, n_mfcc, n_fft, hop_length, win_length)
    img = librosa.display.specshow(feature, x_axis='time', y_axis='linear', sr=sr, hop_length=hop_length, ax=ax2)
    ax2.set_title(f'{title} Features')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Coefficient')
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # Mel Spectrogram
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length, ax=ax3)
    ax3.set_title('Mel Spectrogram')
    plt.colorbar(img, ax=ax3, format='%+2.0f dB')
    
    # Additional features
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=win_length)
    ax4.plot(zcr[0])
    ax4.set_title('Zero Crossing Rate')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('ZCR')
    
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    ax5.plot(spectral_centroids[0])
    ax5.set_title('Spectral Centroid')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{class_name}_sample_{sample_num}_full_analysis.png'))
    plt.close()
    
    return y, sr

def plot_multiple_cams_with_features(input_tensor: torch.Tensor, 
                                   audio_features: Dict[str, np.ndarray],
                                   results: Dict[str, Tuple[torch.Tensor, float]],
                                   class_name: str, save_path: str):
    """Create comprehensive visualization combining CAM results with audio features"""
    num_techniques = len(results)
    fig = plt.figure(figsize=(20, 5*(num_techniques + 1)))
    
    # Plot audio features first
    ax_feat = plt.subplot2grid((num_techniques + 1, 3), (0, 0), colspan=3)
    input_data = input_tensor.squeeze().cpu().numpy()
    im1 = ax_feat.imshow(input_data, aspect='auto', origin='lower')
    ax_feat.set_title(f'Original {class_name} Features')
    plt.colorbar(im1, ax=ax_feat)
    
    # Plot CAM results
    for idx, (technique_name, (heatmap, confidence)) in enumerate(results.items()):
        row = idx + 1
        heatmap_data = heatmap.squeeze().cpu().numpy()
        
        # Original
        ax1 = plt.subplot2grid((num_techniques + 1, 3), (row, 0))
        im1 = ax1.imshow(input_data, aspect='auto', origin='lower')
        ax1.set_title(f'{technique_name} Input')
        plt.colorbar(im1, ax=ax1)
        
        # Heatmap
        ax2 = plt.subplot2grid((num_techniques + 1, 3), (row, 1))
        im2 = ax2.imshow(heatmap_data, cmap='jet', aspect='auto', origin='lower')
        ax2.set_title(f'{technique_name} Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        # Overlay
        ax3 = plt.subplot2grid((num_techniques + 1, 3), (row, 2))
        im3 = ax3.imshow(input_data, aspect='auto', origin='lower')
        ax3.imshow(heatmap_data, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
        ax3.set_title(f'{technique_name} Overlay (Conf: {confidence:.2f})')
        plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_device() -> str:
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def load_model(model_path: str, model_type: str, device: str) -> torch.nn.Module:
    """Load model with proper device handling"""
    model_classes = {
        'U_Net': lambda: U_Net(img_ch=1, output_ch=2),
        'R2U_Net': lambda: R2U_Net(img_ch=1, output_ch=2),
        'R2AttU_Net': lambda: R2AttU_Net(img_ch=1, output_ch=2),
        'AttentionUNet': lambda: AttentionUNet(mfcc_dim=32, output_ch=2)
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_classes[model_type]()
    
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    
    # Initialize fc layer
    dummy_input = torch.randn(1, 1, 32, 112).to(device)
    _ = model(dummy_input)
    
    if 'fc.weight' in state_dict:
        model.fc.weight.data = state_dict['fc.weight'].to(device)
        model.fc.bias.data = state_dict['fc.bias'].to(device)
    
    return model.eval()

def get_target_layer(model: torch.nn.Module, model_type: str) -> torch.nn.Module:
    """Get the target layer for CAM visualization"""
    target_layers = {
        'U_Net': lambda m: m.Up_conv2,
        'AttentionUNet': lambda m: m.Up_conv2,
        'R2U_Net': lambda m: m.Up_RRCNN2,
        'R2AttU_Net': lambda m: m.Up_RRCNN2
    }
    return target_layers[model_type](model)

def find_samples_with_seed(model: torch.nn.Module, loader: DataLoader, 
                         device: str, samples_per_class: int = 2, 
                         seed: int = 15) -> Dict[int, List[Tuple[torch.Tensor, int]]]:
    """Find samples with fixed random seed for reproducibility"""
    set_random_seeds(seed)
    
    class_samples = {0: [], 1: []}
    print("Collecting candidate samples...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            for sample_idx in range(len(labels)):
                label = labels[sample_idx].item()
                if preds[sample_idx] == label:
                    class_samples[label].append((
                        inputs[sample_idx].cpu(),
                        label,
                        batch_idx,
                        sample_idx
                    ))
    
    selected_samples = {0: [], 1: []}
    for class_idx in class_samples:
        if len(class_samples[class_idx]) >= samples_per_class:
            selected_indices = random.sample(range(len(class_samples[class_idx])), samples_per_class)
            selected_samples[class_idx] = [
                (class_samples[class_idx][i][0], class_samples[class_idx][i][1])
                for i in selected_indices
            ]
            print(f"Selected samples for class {class_idx} from batches: "
                  f"{[class_samples[class_idx][i][2] for i in selected_indices]}, "
                  f"indices: {[class_samples[class_idx][i][3] for i in selected_indices]}")
        else:
            print(f"Warning: Not enough samples for class {class_idx}. "
                  f"Found {len(class_samples[class_idx])}, needed {samples_per_class}")
    
    return selected_samples

def analyze_with_multiple_cams(model: torch.nn.Module, input_tensor: torch.Tensor, 
                             target_layer: torch.nn.Module, class_idx: int, 
                             device: str) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Analyze a sample with multiple CAM techniques"""
    cam_techniques = {
        'GradCAM': VanillaGradCAM(model, target_layer),
        'GradCAM++': GradCAMPlusPlus(model, target_layer),
        'LayerCAM': LayerCAM(model, target_layer),
        'ScoreCAM': ScoreCAM(model, target_layer)
    }
    
    results = {}
    for name, cam in cam_techniques.items():
        print(f"Generating {name} heatmap...")
        heatmap, output = cam.get_heatmap(input_tensor.to(device), class_idx)
        confidence = F.softmax(output, dim=1)[0, class_idx].item()
        results[name] = (heatmap, confidence)
    
    return results

def create_comparative_analysis(analysis_results: Dict[str, List[Dict]], save_dir: str) -> None:
    """Create comparative visualizations across samples and classes"""
    
    # Prepare data for plotting
    class_data = {
        class_name: {
            'confidences': [],
            'feature_stats': []
        } for class_name in analysis_results.keys()
    }
    
    for class_name, results in analysis_results.items():
        for result in results:
            class_data[class_name]['confidences'].append(
                list(result['confidences'].values())
            )
            class_data[class_name]['feature_stats'].append(
                list(result['feature_stats'].values())
            )
    
    # Confidence Distribution Plot
    plt.figure(figsize=(12, 6))
    positions = np.arange(len(analysis_results.keys()))
    width = 0.2
    
    for i, technique in enumerate(['GradCAM', 'GradCAM++', 'LayerCAM','ScoreCAM']):
        confidences = [np.mean([sample[i] for sample in class_data[cls]['confidences']])
                      for cls in analysis_results.keys()]
        errors = [np.std([sample[i] for sample in class_data[cls]['confidences']])
                 for cls in analysis_results.keys()]
        
        plt.bar(positions + i*width, confidences, width,
                label=technique, yerr=errors, capsize=5)
    
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.title('CAM Technique Confidence Comparison')
    plt.xticks(positions + width, analysis_results.keys())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_comparison.png'))
    plt.close()
    
    # Create correlation matrices
    for class_name in analysis_results.keys():
        create_correlation_matrix(class_name, analysis_results[class_name], save_dir)

def create_correlation_matrix(class_name: str, results: List[Dict], save_dir: str) -> None:
    """Create and save correlation matrix for features and confidences"""
    data = []
    columns = (
        ['GradCAM', 'GradCAM++', 'ScoreCAM'] +
        ['ZCR', 'RMS', 'Spectral_Centroid']
    )
    
    for result in results:
        row = (
            list(result['confidences'].values()) +
            list(result['feature_stats'].values())
        )
        data.append(row)
    
    correlation_matrix = np.corrcoef(np.array(data).T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
               xticklabels=columns,
               yticklabels=columns,
               annot=True,
               cmap='coolwarm',
               center=0)
    plt.title(f'{class_name} - Feature and Confidence Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{class_name.lower()}_correlations.png'))
    plt.close()

def generate_summary_report(analysis_results: Dict[str, List[Dict]], 
                          save_dir: str, config: Dict[str, Any]) -> None:
    """Generate comprehensive summary report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(save_dir, 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("Audio Classification XAI Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Analysis performed at: {timestamp}\n")
        f.write(f"Model type: {config['model_type']}\n")
        f.write(f"Feature type: {config['feature_params']['type']}\n\n")
        
        for class_name, results in analysis_results.items():
            f.write(f"\n{class_name} Analysis:\n")
            f.write("-" * (len(class_name) + 10) + "\n")
            
            for i, result in enumerate(results):
                f.write(f"\nSample {i+1}:\n")
                f.write("CAM Confidences:\n")
                for technique, conf in result['confidences'].items():
                    f.write(f"  {technique}: {conf:.4f}\n")
                
                f.write("\nFeature Statistics:\n")
                for feat, value in result['feature_stats'].items():
                    f.write(f"  {feat}: {value:.4f}\n")
            
            f.write("\n" + "="*50 + "\n")

def main():
    """Main function for comprehensive XAI analysis with consistent signal processing"""
    # Configuration
    model_path =  '/Users/zainhazzouri/projects/Master-thesis-experiments/results/U_Net_LFCC_32_len5.0S/U_Net_LFCC_32_len5.0S.pth'

    
        # Extract parameters from the model filename
    model_type, feature_type, n_mfcc, length_in_seconds = parse_model_filename(model_path)

    # Update the config dictionary
    config = {
        'model_path': model_path,
        'test_data_path': "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/",
        'save_dir': '/Users/zainhazzouri/projects/Master-thesis-experiments/results/U_Net_LFCC_32_len5.0S/cam',
        'model_type': model_type,
        'random_seed': 42,
        'feature_params': {
            'type': feature_type,
            'n_mfcc': n_mfcc,
            'length_in_seconds': length_in_seconds,
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048
        },
        'batch_size': 16,
        'samples_per_class': 2
    }
    

    try:
        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config['save_dir'], f"analysis_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(save_dir, 'analysis_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Analysis started")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Setup environment
        device = get_device()
        set_random_seeds(config['random_seed'])
        logging.info(f"Using device: {device}")

        # Load model and dataset using extracted parameters
        model = load_model(config['model_path'], config['model_type'], device)
        target_layer = get_target_layer(model, config['model_type'])

        # Parameters for feature extraction
        feature_params = config['feature_params']
        sr = feature_params['sample_rate']
        n_mfcc = feature_params['n_mfcc']
        length_in_seconds = feature_params['length_in_seconds']
        n_fft = feature_params['n_fft']
        hop_length = feature_params['hop_length']
        win_length = feature_params['win_length']

        dataset = AudioProcessor(
            audio_dir=config['test_data_path'],
            n_mfcc=n_mfcc,
            length_in_seconds=length_in_seconds,
            type_of_transformation=feature_params['type']
        )



        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(config['random_seed'] + worker_id)
        )

        # Find samples
        class_names = ['Music', 'Speech']
        samples_found = find_samples_with_seed(
            model=model,
            loader=loader,
            device=device,
            samples_per_class=config['samples_per_class'],
            seed=config['random_seed']
        )

        # Analysis results storage
        analysis_results = {class_name: [] for class_name in class_names}

        # Process each sample
        for class_idx, samples in samples_found.items():
            for i, (input_tensor, label) in enumerate(samples):
                sample_results = {}

                try:
                    # Get audio file path
                    file_path = dataset.audio_files_and_labels[i][0]

                    # Generate comprehensive visualizations
                    y, sr = visualize_sample_with_features(
                        file_path=file_path,
                        save_dir=save_dir,
                        class_name=class_names[class_idx],
                        sample_num=i+1,
                        feature_type=feature_params['type'],
                        n_mfcc=n_mfcc,
                        sr=sr,
                        length_in_seconds=length_in_seconds,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length
                    )

                    # Analyze audio features
                    feature_results = analyze_audio_features(
                        y=y,
                        sr=sr,
                        output_dir=save_dir,
                        class_name=class_names[class_idx],
                        sample_num=i+1,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length
                    )

                    # Generate CAM visualizations
                    cam_results = analyze_with_multiple_cams(
                        model=model,
                        input_tensor=input_tensor.unsqueeze(0),
                        target_layer=target_layer,
                        class_idx=label,
                        device=device
                    )

                    # Save results
                    audio_save_path = os.path.join(
                        save_dir,
                        f"{class_names[class_idx].lower()}_sample_{i+1}.wav"
                    )
                    sf.write(audio_save_path, y, sr, subtype='PCM_16')

                    # Store results
                    sample_results = {
                        'sample_idx': i+1,
                        'audio_path': audio_save_path,
                        'confidences': {name: conf for name, (_, conf) in cam_results.items()},
                        'feature_stats': {
                            'zcr_mean': float(np.mean(feature_results['zcr'])),
                            'rms_mean': float(np.mean(feature_results['rms'])),
                            'spectral_centroid_mean': float(np.mean(feature_results['spectral_centroids']))
                        },
                        'cam_results': cam_results
                    }

                    # Generate combined visualization
                    plot_multiple_cams_with_features(
                        input_tensor=input_tensor,
                        audio_features=feature_results,
                        results=cam_results,
                        class_name=class_names[class_idx],
                        save_path=os.path.join(
                            save_dir,
                            f'combined_analysis_{class_names[class_idx].lower()}_sample_{i+1}.png'
                        )
                    )

                except Exception as e:
                    logging.error(f"Error processing {class_names[class_idx]} sample {i+1}: {str(e)}")
                    logging.error(traceback.format_exc())
                    sample_results['error'] = str(e)

                analysis_results[class_names[class_idx]].append(sample_results)

        # Generate final analyses
        create_comparative_analysis(analysis_results, save_dir)
        generate_summary_report(analysis_results, save_dir, config)

        # Save final metadata
        metadata = {
            'timestamp': timestamp,
            'config': config,
            'analysis_stats': {
                'total_samples': sum(len(samples) for samples in analysis_results.values()),
                'successful_analyses': sum(
                    1 for samples in analysis_results.values()
                    for sample in samples
                    if 'error' not in sample
                )
            }
        }

        with open(os.path.join(save_dir, 'analysis_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create completion marker
        with open(os.path.join(save_dir, 'analysis_complete.txt'), 'w') as f:
            f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {save_dir}")
        print("\nGenerated files:")
        for root, dirs, files in os.walk(save_dir):
            level = root.replace(save_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            subindent = ' ' * 4 * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{subindent}{f}")

    except Exception as e:
        logging.error(f"Fatal error during analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def create_attention_maps_gif(results_dir: str) -> None:
    """Create animated GIFs of attention maps over time"""
    for class_name in ['music', 'speech']:
        for technique in ['GradCAM', 'GradCAM++', 'ScoreCAM']:
            frames = []
            results_path = os.path.join(results_dir, f'combined_analysis_{class_name}_sample_1.png')
            
            if os.path.exists(results_path):
                img = plt.imread(results_path)
                
                # Process frames
                num_frames = img.shape[1] // 4
                for i in range(num_frames):
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(img[:, i*4:(i+1)*4], aspect='auto')
                    plt.title(f'{class_name.capitalize()} {technique} - Frame {i+1}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    
                    # Convert plot to image
                    fig = plt.gcf()
                    plt.close()
                    fig.canvas.draw()
                    frame = np.array(fig.canvas.renderer.buffer_rgba())
                    frames.append(frame)
                
                # Save as GIF
                gif_path = os.path.join(results_dir, f'{class_name}_{technique}_attention.gif')
                imageio.mimsave(gif_path, frames, duration=0.2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error during execution: {e}")
        traceback.print_exc()