# visualization_utils.py
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import scipy
import traceback
import imageio
from typing import Dict, Tuple, List
from audio_utils import compute_feature, load_and_preprocess_audio, compute_times
import torch

def analyze_audio_features(y: np.ndarray, sr: int, output_dir: str,
                           class_name: str, sample_num: int,
                           n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048) -> Dict[str, np.ndarray]:
    """Analyze audio features and generate visualizations."""
    features = {
        'zcr': librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=win_length),
        'rms': librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length),
        'spectral_centroids': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    }
    times = compute_times(len(y), sr, hop_length)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    titles = ['Zero Crossing Rate', 'RMS Energy', 'Spectral Centroid', 'Spectral Rolloff']
    ylabels = ['ZCR', 'Energy', 'Frequency (Hz)', 'Frequency (Hz)']
    for idx, (key, ax) in enumerate(zip(features, axes)):
        ax.plot(times[:features[key].shape[1]], features[key][0])
        ax.set_title(titles[idx])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabels[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_sample_{sample_num}_features.png'))
    plt.close()
    return features

def visualize_sample_with_features(file_path: str, save_dir: str, class_name: str,
                                   sample_num: int, feature_type: str = 'LFCC',
                                   n_mfcc: int = 32, sr: int = 44100,
                                   length_in_seconds: float = 5.0,
                                   n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048) -> Tuple[np.ndarray, int]:
    """Create comprehensive visualization of audio sample with features."""
    y = load_and_preprocess_audio(file_path, sr, length_in_seconds)
    fig = plt.figure(figsize=(20, 15))
    # Waveform
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    # Feature representation
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    feature, title = compute_feature(y, sr, feature_type, n_mfcc, n_fft, hop_length, win_length, length_in_seconds)
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
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=ax3, format='%+2.0f dB')
    # Additional features
    times = compute_times(len(y), sr, hop_length)
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=win_length)
    ax4.plot(times[:zcr.shape[1]], zcr[0])
    ax4.set_title('Zero Crossing Rate')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ZCR')
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    ax5.plot(times[:spectral_centroids.shape[1]], spectral_centroids[0])
    ax5.set_title('Spectral Centroid')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{class_name}_sample_{sample_num}_full_analysis.png'))
    plt.close()
    return y, sr

def plot_multiple_cams_with_features(input_tensor: torch.Tensor,
                                   audio_features: Dict[str, np.ndarray],
                                   results: Dict[str, Tuple[torch.Tensor, float]],
                                   class_name: str, 
                                   save_path: str,
                                   feature_type: str,
                                   n_mfcc: int,
                                   length_in_seconds: float,
                                   sr: int,
                                   hop_length: int):
    """
    Create comprehensive visualization combining CAM results with audio features.
    
    Parameters:
        input_tensor: Input tensor of shape (batch_size, n_features, time_steps)
        audio_features: Dictionary containing audio features
        results: Dictionary containing CAM results for different techniques
        class_name: Name of the audio class
        save_path: Path to save the visualization
        feature_type: Type of audio feature (e.g., 'LFCC', 'MFCC')
        n_mfcc: Number of features/coefficients
        length_in_seconds: Duration of audio in seconds
        sr: Sampling rate
        hop_length: Number of samples between successive frames
    """
    try:
        print("Starting visualization...")
        input_data = input_tensor.squeeze().cpu().numpy()
        n_coefficients, n_frames = input_data.shape
        
        # Calculate correct time axis
        times = np.linspace(0, length_in_seconds, n_frames)
        
        # Calculate frequency axis for feature coefficients
        if feature_type in ['MFCC', 'LFCC']:
            # For MFCC/LFCC, y-axis represents coefficient index
            freq_axis = np.arange(n_coefficients)
        else:
            # For spectral features, calculate actual frequencies
            freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2 * (n_coefficients - 1))
        
        num_techniques = len(results)
        total_rows = num_techniques + 1
        fig = plt.figure(figsize=(20, 5 * total_rows))
        
        # Plot original input features
        ax_feat = plt.subplot2grid((total_rows, 3), (0, 0), colspan=3)
        img = librosa.display.specshow(
            input_data,
            x_coords=times,
            y_coords=freq_axis,
            x_axis='time',
            y_axis='linear' if feature_type not in ['MFCC', 'LFCC'] else 'frames',
            ax=ax_feat
        )
        ax_feat.set_ylabel(f'{feature_type} Coefficient' if feature_type in ['MFCC', 'LFCC'] else 'Frequency (Hz)')
        ax_feat.set_xlabel('Time (s)')
        ax_feat.set_xlim(0, length_in_seconds)
        ax_feat.set_ylim(0, n_coefficients if feature_type in ['MFCC', 'LFCC'] else freq_axis[-1])
        plt.colorbar(img, ax=ax_feat)
        ax_feat.set_title(f'Original {class_name} {feature_type} Features')
        
        # Plot CAM results
        for idx, (technique_name, (heatmap, confidence)) in enumerate(results.items()):
            print(f"Processing {technique_name}...")
            row = idx + 1
            heatmap_data = heatmap.squeeze().cpu().numpy()
            
            # Resize heatmap to match input dimensions exactly
            heatmap_resized = scipy.ndimage.zoom(
                heatmap_data,
                (n_coefficients / heatmap_data.shape[0],
                 n_frames / heatmap_data.shape[1]),
                order=1
            )
            
            # Original features subplot
            ax1 = plt.subplot2grid((total_rows, 3), (row, 0))
            img1 = librosa.display.specshow(
                input_data,
                x_coords=times,
                y_coords=freq_axis,
                x_axis='time',
                y_axis='linear' if feature_type not in ['MFCC', 'LFCC'] else 'frames',
                ax=ax1
            )
            ax1.set_ylabel(f'{feature_type} Coefficient' if feature_type in ['MFCC', 'LFCC'] else 'Frequency (Hz)')
            ax1.set_xlabel('Time (s)')
            ax1.set_xlim(0, length_in_seconds)
            ax1.set_ylim(0, n_coefficients if feature_type in ['MFCC', 'LFCC'] else freq_axis[-1])
            plt.colorbar(img1, ax=ax1)
            ax1.set_title(f'{technique_name} Input')
            
            # Heatmap subplot
            ax2 = plt.subplot2grid((total_rows, 3), (row, 1))
            img2 = librosa.display.specshow(
                heatmap_resized,
                x_coords=times,
                y_coords=freq_axis,
                x_axis='time',
                y_axis='linear' if feature_type not in ['MFCC', 'LFCC'] else 'frames',
                ax=ax2,
                cmap='jet'
            )
            ax2.set_ylabel(f'{feature_type} Coefficient' if feature_type in ['MFCC', 'LFCC'] else 'Frequency (Hz)')
            ax2.set_xlabel('Time (s)')
            ax2.set_xlim(0, length_in_seconds)
            ax2.set_ylim(0, n_coefficients if feature_type in ['MFCC', 'LFCC'] else freq_axis[-1])
            plt.colorbar(img2, ax=ax2)
            ax2.set_title(f'{technique_name} Heatmap')
            
            # Overlay subplot
            ax3 = plt.subplot2grid((total_rows, 3), (row, 2))
            img3 = librosa.display.specshow(
                input_data,
                x_coords=times,
                y_coords=freq_axis,
                x_axis='time',
                y_axis='linear' if feature_type not in ['MFCC', 'LFCC'] else 'frames',
                ax=ax3
            )
            
            # Normalize heatmap for overlay
            heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            
            # Create overlay using pcolormesh with correct dimensions
            times_mesh, freqs_mesh = np.meshgrid(times, freq_axis)
            ax3.pcolormesh(times_mesh, freqs_mesh, heatmap_norm, 
                          alpha=0.5, cmap='jet', shading='gouraud')
            
            ax3.set_ylabel(f'{feature_type} Coefficient' if feature_type in ['MFCC', 'LFCC'] else 'Frequency (Hz)')
            ax3.set_xlabel('Time (s)')
            ax3.set_xlim(0, length_in_seconds)
            ax3.set_ylim(0, n_coefficients if feature_type in ['MFCC', 'LFCC'] else freq_axis[-1])
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

def create_attention_maps_gif(results_dir: str, sample_nums: List[int] = [1, 2]) -> None:
    """Create animated GIFs of attention maps over time for each class and CAM technique."""
    print("Starting GIF creation process...")
    techniques = ['GradCAM', 'GradCAM++', 'LayerCAM', 'ScoreCAM']
    classes = ['music', 'speech']
    frame_duration = 0.5  # Duration for each frame in seconds
    for class_name in classes:
        for sample_num in sample_nums:
            print(f"\nProcessing {class_name} sample {sample_num}")
            results_path = os.path.join(
                results_dir, 
                f'combined_analysis_{class_name}_sample_{sample_num}.png'
            )
            if not os.path.exists(results_path):
                print(f"Warning: File not found - {results_path}")
                continue
            try:
                img = plt.imread(results_path)
                print(f"Loaded image shape: {img.shape}")
                for technique in techniques:
                    print(f"Creating GIF for {technique}...")
                    frames = []
                    height = img.shape[0] // (len(techniques) + 1)  # Account for original input
                    start_y = height * (techniques.index(technique) + 1)
                    heatmap_region = img[start_y:start_y + height, :]
                    num_time_steps = heatmap_region.shape[1] // 10
                    window_size = heatmap_region.shape[1] // num_time_steps
                    for i in range(num_time_steps):
                        plt.figure(figsize=(12, 6))
                        plt.subplot(2, 1, 1)
                        plt.imshow(heatmap_region[:, i*window_size:(i+1)*window_size])
                        plt.title(f'{class_name.capitalize()} {technique} - Time Step {i+1}')
                        plt.axis('off')
                        plt.subplot(2, 1, 2)
                        plt.imshow(heatmap_region[:, i*window_size:(i+1)*window_size], cmap='jet')
                        plt.title(f'Attention Heatmap - Time Step {i+1}')
                        plt.axis('off')
                        plt.tight_layout()
                        fig = plt.gcf()
                        fig.canvas.draw()
                        frame = np.array(fig.canvas.renderer.buffer_rgba())
                        frames.append(frame)
                        plt.close()
                    gif_path = os.path.join(
                        results_dir, 
                        f'{class_name}_{technique}_sample_{sample_num}_attention.gif'
                    )
                    print(f"Saving GIF to: {gif_path}")
                    imageio.mimsave(gif_path, frames, duration=frame_duration)
            except Exception as e:
                print(f"Error processing {class_name} {technique}: {str(e)}")
                traceback.print_exc()
                continue
    print("\nGIF creation process completed!")