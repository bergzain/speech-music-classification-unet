# audio_utils.py
import numpy as np
import librosa
import scipy
from typing import Tuple

def preprocess_audio(y: np.ndarray, sr: int, length_in_seconds: float) -> np.ndarray:
    """Trim or pad the audio to the desired length."""
    desired_length = int(sr * length_in_seconds)
    if len(y) > desired_length:
        y = y[:desired_length]
    else:
        y = np.pad(y, (0, max(0, desired_length - len(y))), 'constant')
    return y

def compute_expected_time_steps(sr: int, length_in_seconds: float, win_length: int, hop_length: int) -> int:
    """Compute the expected number of time steps for the given parameters."""
    desired_length = int(sr * length_in_seconds)
    expected_time_steps = int(np.ceil((desired_length - win_length) / hop_length)) + 1
    return expected_time_steps

def compute_lfcc(y, sr, n_fft, hop_length, win_length, n_lfcc):
    """Compute Linear Frequency Cepstral Coefficients (LFCC) using librosa."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))**2
    n_filters = n_lfcc
    freqs = np.linspace(0, sr/2, n_fft//2+1)
    filter_freqs = np.linspace(0, sr/2, n_filters+2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        f_left = filter_freqs[i]
        f_center = filter_freqs[i+1]
        f_right = filter_freqs[i+2]
        left_slope = (freqs - f_left) / (f_center - f_left + 1e-8)
        right_slope = (f_right - freqs) / (f_right - f_center + 1e-8)
        fb[i] = np.maximum(0, np.minimum(left_slope, right_slope))
    lfcc_spec = np.dot(fb, S)
    lfcc_spec = np.log(lfcc_spec + 1e-10)
    lfcc = scipy.fftpack.dct(lfcc_spec, axis=0, norm='ortho')[:n_lfcc]
    return lfcc

def pad_or_trim_feature(feature: np.ndarray, expected_time_steps: int) -> np.ndarray:
    """Pad or trim the feature to have the expected number of time steps."""
    current_time_steps = feature.shape[1]
    if current_time_steps > expected_time_steps:
        feature = feature[:, :expected_time_steps]
    elif current_time_steps < expected_time_steps:
        padding = expected_time_steps - current_time_steps
        feature = np.pad(feature, ((0, 0), (0, padding)), mode='constant')
    return feature

def compute_feature(y: np.ndarray, sr: int, feature_type: str = 'LFCC', n_mfcc: int = 32,
                    n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048,
                    length_in_seconds: float = 5.0) -> Tuple[np.ndarray, str]:
    """Compute audio features with specified parameters and ensure consistency in feature dimensions."""
    y = preprocess_audio(y, sr, length_in_seconds)
    expected_time_steps = compute_expected_time_steps(sr, length_in_seconds, win_length, hop_length)
    if feature_type == 'MFCC':
        feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                       hop_length=hop_length, win_length=win_length)
        title = 'MFCC'
    elif feature_type == 'LFCC':
        feature = compute_lfcc(y, sr, n_fft, hop_length, win_length, n_mfcc)
        title = 'LFCC'
    elif feature_type == 'delta':
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop_length, win_length=win_length)
        feature = librosa.feature.delta(mfcc)
        title = 'Delta MFCC'
    elif feature_type == 'delta-delta':
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop_length, win_length=win_length)
        feature = librosa.feature.delta(mfcc, order=2)
        title = 'Delta-Delta MFCC'
    elif feature_type == 'lfcc-delta':
        lfcc = compute_lfcc(y, sr, n_fft, hop_length, win_length, n_mfcc)
        feature = librosa.feature.delta(lfcc)
        title = 'Delta LFCC'
    elif feature_type == 'lfcc-delta-delta':
        lfcc = compute_lfcc(y, sr, n_fft, hop_length, win_length, n_mfcc)
        feature = librosa.feature.delta(lfcc, order=2)
        title = 'Delta-Delta LFCC'
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    feature = pad_or_trim_feature(feature, expected_time_steps)
    return feature, title

def compute_times(y_length: int, sr: int, hop_length: int) -> np.ndarray:
    """Compute the time axis based on the length of the audio and hop length."""
    frames = np.arange(y_length // hop_length + 1)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return times

def load_and_preprocess_audio(file_path: str, sr: int, length_in_seconds: float) -> np.ndarray:
    """Load and preprocess audio file."""
    y, _ = librosa.load(file_path, sr=sr)
    y = preprocess_audio(y, sr, length_in_seconds)
    return y