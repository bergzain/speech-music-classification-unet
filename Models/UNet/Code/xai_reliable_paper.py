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
import scipy
from scipy.fftpack import dct  # Import dct for LFCC computation
import torchaudio

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

def plot_multiple_cams_with_features(input_tensor: torch.Tensor,
                                     results: Dict[str, Tuple[torch.Tensor, float]],
                                     class_name: str, save_path: str,
                                     feature_type: str,
                                     n_mfcc: int,
                                     length_in_seconds: float,
                                     sr: int,
                                     hop_length: int):
    """
    Create comprehensive visualization combining CAM results with audio features.
    """
    try:
        print("Starting visualization...")
        input_data = input_tensor.squeeze().cpu().numpy()
        n_coefficients, n_frames = input_data.shape
        print(f"input_data.shape: {input_data.shape}")
        print(f"Number of coefficients (features): {n_coefficients}")

        # Compute time axis manually
        times = np.linspace(0, length_in_seconds, n_frames)

        # Calculate total rows
        num_techniques = len(results)
        total_rows = num_techniques + 1

        # Create the figure
        fig = plt.figure(figsize=(20, 5 * total_rows))

        # Plot CAM results
        for idx, (technique_name, (heatmap, confidence)) in enumerate(results.items()):
            print(f"Processing {technique_name}...")
            row = idx + 1
            heatmap_data = heatmap.squeeze().cpu().numpy()
            print(f"heatmap_data.shape before resizing: {heatmap_data.shape}")

            # Resize heatmap to match input data dimensions if necessary
            if heatmap_data.shape != input_data.shape:
                heatmap_resized = scipy.ndimage.zoom(
                    heatmap_data,
                    (
                        n_coefficients / heatmap_data.shape[0],
                        n_frames / heatmap_data.shape[1]
                    ),
                    order=1
                )
                print(f"heatmap_resized.shape after resizing: {heatmap_resized.shape}")
            else:
                heatmap_resized = heatmap_data

            # Original features subplot
            ax1 = plt.subplot2grid((total_rows, 3), (row, 0))
            img1 = librosa.display.specshow(
                input_data,
                x_axis='time',
                y_axis=None,  # Disable automatic scaling
                sr=sr,
                hop_length=hop_length,
                ax=ax1
            )
            ax1.set_ylabel(f'{feature_type} Coefficient Index')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylim(0, n_coefficients)
            ax1.set_yticks(np.linspace(0, n_coefficients, 5))  # Adjust as needed
            plt.colorbar(img1, ax=ax1)
            ax1.set_title(f'{technique_name} Input')

            # Heatmap subplot
            ax2 = plt.subplot2grid((total_rows, 3), (row, 1))
            img2 = ax2.imshow(
                heatmap_resized,
                aspect='auto',
                origin='lower',
                cmap='jet'
            )
            ax2.set_ylabel(f'{feature_type} Coefficient Index')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylim(0, n_coefficients)
            ax2.set_yticks(np.linspace(0, n_coefficients, 5))  # Adjust as needed
            plt.colorbar(img2, ax=ax2)
            ax2.set_title(f'{technique_name} Heatmap')

            # Overlay subplot
            ax3 = plt.subplot2grid((total_rows, 3), (row, 2))
            img3 = librosa.display.specshow(
                input_data,
                x_axis='time',
                y_axis=None,  # Disable automatic scaling
                sr=sr,
                hop_length=hop_length,
                ax=ax3
            )
            
            ax3.imshow(
                heatmap_resized,
                cmap='jet',
                alpha=0.5,
                aspect='auto',
                origin='lower'
            )
            ax3.set_ylabel(f'{feature_type} Coefficient Index')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylim(0, n_coefficients)
            ax3.set_yticks(np.linspace(0, n_coefficients, 5))  # Adjust as needed
            plt.colorbar(img3, ax=ax3)
            ax3.set_title(f'{technique_name} Overlay (Conf: {confidence:.2f})')

        print("Finalizing plot...")
        plt.tight_layout()
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot completed successfully!")

    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        traceback.print_exc()
        raise




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


def save_processed_audio(file_path: str, save_path: str, target_sample_rate: int, chunk_length_seconds: float, chunk_index: int = 0) -> bool:
    """
    Load, process, and save a specific chunk of an audio file.
    
    Args:
        file_path (str): Path to the input audio file
        save_path (str): Path where the processed audio will be saved
        target_sample_rate (int): Target sample rate for the audio
        chunk_length_seconds (float): Length of each chunk in seconds
        chunk_index (int): Index of the chunk to save (default: 0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate chunk parameters
        chunk_length_samples = int(chunk_length_seconds * target_sample_rate)
        start_sample = chunk_index * chunk_length_samples
        end_sample = start_sample + chunk_length_samples
        
        # Extract the chunk
        if end_sample <= waveform.shape[1]:
            chunk = waveform[:, start_sample:end_sample]
        else:
            # If chunk would extend beyond the audio, pad with zeros
            chunk = torch.zeros((1, chunk_length_samples), dtype=waveform.dtype)
            remaining_samples = waveform.shape[1] - start_sample
            if remaining_samples > 0:
                chunk[:, :remaining_samples] = waveform[:, start_sample:]
            
        # Save processed chunk
        torchaudio.save(save_path, chunk, target_sample_rate)
        return True
        
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return False



    
def main():
    """Main function for comprehensive XAI analysis with consistent signal processing"""
    # Configuration
    model_path =  '/Users/zainhazzouri/projects/Master-thesis-experiments/results/AttentionUNet_LFCC_32_len5.0S/AttentionUNet_LFCC_32_len5.0S.pth'

    
    # Extract parameters from the model filename
    model_type, feature_type, n_mfcc, length_in_seconds = parse_model_filename(model_path)

    # Update the config dictionary
    config = {
        'model_path': model_path,
        'test_data_path': "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/",
        'save_dir': '/Users/zainhazzouri/projects/Master-thesis-experiments/results/AttentionUNet_LFCC_32_len5.0S/cam',
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
                    
                    matching_files = [
                        (idx, filepath) for idx, (filepath, file_label) 
                        in enumerate(dataset.audio_files_and_labels) 
                        if file_label == class_idx
                    ]
                    
                    if not matching_files:
                        raise ValueError(f"No files found for class {class_idx}")
                        
                    # Use the first matching file for this class
                    file_idx, file_path = matching_files[i]
                    
                    print(f"Processing {class_names[class_idx]} sample {i + 1}")
                    print(f"Using file: {file_path}")



                    

                    # Generate CAM visualizations
                    cam_results = analyze_with_multiple_cams(
                        model=model,
                        input_tensor=input_tensor.unsqueeze(0),
                        target_layer=target_layer,
                        class_idx=label,
                        device=device
                    )



                    # Save original audio file with AudioProcessor's preprocessing
                    audio_save_path = os.path.join(
                        save_dir,
                        f"{class_names[class_idx].lower()}_sample_{i+1}.wav"
                    )
                    
                    if save_processed_audio(
                        file_path=file_path, 
                        save_path=audio_save_path,
                        target_sample_rate=config['feature_params']['sample_rate'],
                        chunk_length_seconds=config['feature_params']['length_in_seconds']
                    ):
                        # Continue with analysis
                        sample_results = {
                            'sample_idx': i+1,
                            'audio_path': audio_save_path,
                            'file_path': file_path,
                            'confidences': {name: conf for name, (_, conf) in cam_results.items()},
                            'cam_results': cam_results
                        }

                    # Generate combined visualization
                    plot_multiple_cams_with_features(
                        input_tensor=input_tensor,
                        results=cam_results,
                        class_name=class_names[class_idx],
                        save_path=os.path.join(
                            save_dir,
                            f'combined_analysis_{class_names[class_idx].lower()}_sample_{i+1}.png'
                        ),
                        feature_type=feature_params['type'],
                        n_mfcc=feature_params['n_mfcc'],
                        length_in_seconds=feature_params['length_in_seconds'],
                        sr=feature_params['sample_rate'],
                        hop_length=feature_params['hop_length']
                    )

                except Exception as e:
                    logging.error(f"Error processing {class_names[class_idx]} sample {i+1}: {str(e)}")
                    logging.error(traceback.format_exc())
                    sample_results['error'] = str(e)

                analysis_results[class_names[class_idx]].append(sample_results)


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

        # save GIF 
        # results_directory = save_dir 
        # create_attention_maps_gif(results_directory, sample_nums=[1, 2])
    
    
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



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error during execution: {e}")
        traceback.print_exc()


