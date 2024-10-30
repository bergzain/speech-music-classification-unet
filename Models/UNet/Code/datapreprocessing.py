#%%
import glob
import random
import torch
import torchaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from torch.utils.data import Dataset
import os

#%% 




#%%
sample_rate = 44100 # check data_parsing_and_preprocessing.ipynb
target_sample_rate = 44100  # Define your target sample rate


class AudioProcessor(Dataset):
    def __init__(self, audio_dir: str, n_mfcc: int, length_in_seconds: float, type_of_transformation: str):
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.device = self._get_device()
        self.target_length = int(length_in_seconds * 44100)  # sample_rate = 44100
        self.type_of_transformation = type_of_transformation
        
        # Initialize counters
        self.speech_count = self.music_count = 0
        self.speech_chunk_count = self.music_chunk_count = 0
        
        # Load audio files
        self.music_waves = glob.glob(os.path.join(audio_dir, "music_wav", "*.wav"))
        self.speech_waves = glob.glob(os.path.join(audio_dir, "speech_wav", "*.wav"))
        
        # Store file paths and labels
        self.audio_files_and_labels = self._load_audio_files_and_labels()
        
        # Configure transforms based on type
        self._setup_transform()

    def _setup_transform(self):
        """Setup the appropriate audio transformation"""
        if self.type_of_transformation == 'MFCC':
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=44100,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_mels': min(2 * self.n_mfcc, 128),  # Reduced n_mels
                    'n_fft': 2048,  # Increased FFT size
                    'hop_length': 512,
                    'f_min': 0,  # Start from 0 Hz
                    'f_max': 22050  # Up to Nyquist frequency (44100/2)
                }
            )
        elif self.type_of_transformation == 'LFCC':
            self.transform = torchaudio.transforms.LFCC(
                sample_rate=44100,
                n_lfcc=self.n_mfcc,
                speckwargs={
                    'n_fft': 2048,
                    'hop_length': 512
                }
            )
        elif self.type_of_transformation in ['delta', 'delta-delta']:
            base_transform = torchaudio.transforms.MFCC(
                sample_rate=44100,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_mels': min(2 * self.n_mfcc, 128),
                    'n_fft': 2048,
                    'hop_length': 512
                }
            )
            self.transform = lambda x: self._compute_deltas(base_transform(x))
        else:
            raise ValueError(f"Unsupported transformation type: {self.type_of_transformation}")

    def _compute_deltas(self, features):
        """Compute delta or delta-delta features"""
        if self.type_of_transformation == 'delta':
            return torchaudio.functional.compute_deltas(features)
        else:  # delta-delta
            deltas = torchaudio.functional.compute_deltas(features)
            return torchaudio.functional.compute_deltas(deltas)

    def _get_device(self):
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_built():
            return "mps"
        return "cpu"


    def _load_audio_files_and_labels(self):
        """Load audio files and their corresponding labels"""
        files_and_labels = []
        for file_path in self.music_waves:
            files_and_labels.append((file_path, 0))  # 0 for music
            self.music_count += 1
        for file_path in self.speech_waves:
            files_and_labels.append((file_path, 1))  # 1 for speech
            self.speech_count += 1
        return files_and_labels

    def preprocess(self, filepath):
        """Preprocess audio file"""
        # Load audio
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if necessary
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or cut to target length
        if waveform.shape[1] < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - waveform.shape[1]))
        
        # Split into chunks
        chunks = []
        for start in range(0, waveform.shape[1], self.target_length):
            end = start + self.target_length
            if end <= waveform.shape[1]:
                chunk = waveform[:, start:end]
                features = self.transform(chunk)
                
                # Ensure consistent feature size
                target_length = 112  # Standard feature length
                if features.shape[2] > target_length:
                    features = features[:, :, :target_length]
                elif features.shape[2] < target_length:
                    features = torch.nn.functional.pad(features, (0, target_length - features.shape[2]))
                
                chunks.append(features)
        
        return torch.stack(chunks) if chunks else torch.empty(0)

    def __len__(self):
        return len(self.audio_files_and_labels)

    def __getitem__(self, idx):
        filepath, label = self.audio_files_and_labels[idx]
        features = self.preprocess(filepath)
        
        if features.shape[0] == 0:
            # Handle empty features
            features = torch.zeros((1, self.n_mfcc, 112))
        
        # Update chunk counts
        if label == 0:
            self.music_chunk_count += features.shape[0]
        else:
            self.speech_chunk_count += features.shape[0]
        
        # Return first chunk and label
        return features[0], label

    
    def _cut_if_necessary(self, signal):
        target_length = int(self.num_samples)
        split_signals = []

        # Iterate over the signal in chunks of target_length
        for start in range(0, int(signal.shape[1]), int(target_length)):
            end = start + int(target_length)
            if end <= int(signal.shape[1]):
                split_signals.append(signal[:, start:end])
            else:
                # If the last chunk is shorter than target_length, it can be discarded or padded
                # Here, we choose to pad the last chunk
                padding = int(target_length - (int(signal.shape[1]) - start))
                split_signal = torch.nn.functional.pad(signal[:, start:], (0, padding))
                split_signals.append(split_signal)
                break  # No more chunks after the last one

        return split_signals

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1] # signal.shape = [1, 1, 441000]
        target_length = self.num_samples
        if length_signal < target_length:
            print("Padding to target length...")
            num_missing_samples = target_length - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            # print(f"signal.shape after padding: {signal.shape}")
        return signal

    def _resample_if_necessary(self, signal, sr): 
        if sr != self.target_sample_rate:
            print(f"Resampling to target sample rate: {self.target_sample_rate}...")
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            # print(f"signal.shape after resampling: {signal.shape}")
        return signal

    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0] > 1:
            print("Downmixing to mono...")
            signal = torch.mean(signal, dim=0, keepdim=True)
            # print(f"signal.shape after downmixing: {signal.shape}")
        return signal

#%%
def calculate_number_of_samples():
    path_to_test = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/" 
    # # #
    val_dataset= AudioProcessor(audio_dir=path_to_test, n_mfcc=32, length_in_seconds=0.5, type_of_transformation='MFCC')
    
    val_dataset.count_chunks(val_dataset.load_audio_files_and_labels())


    print(f"Validation dataset - Speech count: {val_dataset.speech_count}, Music count: {val_dataset.music_count}")
    print(f"Validation dataset - Speech chunk count: {val_dataset.speech_chunk_count}, Music chunk count: {val_dataset.music_chunk_count}")
    print(val_dataset[0][0].shape)
    

    # path_to_train = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/train/"
    # train_dataset = AudioProcessor(audio_dir=path_to_train)
    # train_dataset.count_chunks(train_dataset.load_audio_files_and_labels())
    # print(f"Training dataset - Speech count: {train_dataset.speech_count}, Music count: {train_dataset.music_count}")
    # print(f"Training dataset - Speech chunk count: {train_dataset.speech_chunk_count}, Music chunk count: {train_dataset.music_chunk_count}")
    # print(train_dataset[0][0].shape)
    


# %%
if __name__ == "__main__":
    calculate_number_of_samples()
# %%
