


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



length_in_seconds = 5 # duration in seconds
n_mfcc = 32 # number of MFCCs features
sample_rate = 44100 # check data_parsing_and_preprocessing.ipynb
target_sample_rate = 44100  # Define your target sample rate
samples_for_ten_seconds = length_in_seconds * sample_rate
target_length = samples_for_ten_seconds  # Assuming this is the length of the audio you want to process


class AudioProcessor(Dataset):
    def __init__(self, audio_dir, n_mfcc=n_mfcc,num_samples=samples_for_ten_seconds):
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.device = self.get_device()
        self.music_waves = []
        self.speech_waves = []
        self.load_audio_files()
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.audio_files_and_labels = self.load_audio_files_and_labels()

    def load_audio_files(self):
        self.music_waves = glob.glob(os.path.join(self.audio_dir, "music_wav", "*.wav"))
        self.speech_waves = glob.glob(os.path.join(self.audio_dir, "speech_wav", "*.wav"))
        # print("Music waves:", self.music_waves)
        # print("Speech waves:", self.speech_waves)

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"

    def preprocess(self, filepath):
        waveform, _ = torchaudio.load(filepath)
        # print(f"waveform.shape: {waveform.shape}")
        waveform = self._resample_if_necessary(waveform, sample_rate)  # resample if necessary
        
        # print(f"waveform.shape after resampling: {waveform.shape}")
        waveform = self._mix_down_if_necessary(waveform)  # convert stereo to mono
        # print(f"waveform.shape after downmixing: {waveform.shape}")
        waveform = self._right_pad_if_necessary(waveform)  # pad if necessary
        # print(f"waveform.shape after padding: {waveform.shape}")
        waveforms = self._cut_if_necessary(waveform)  # cut if necessary
        # print(f"waveforms.shape after cutting: {waveforms[0].shape}")
        mfccs = []
        deltas = []
        delta_deltas = []

        for wf in waveforms:
            # Compute MFCCs for each waveform chunk
            mfcc = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate,
                                              n_mfcc=self.n_mfcc)(wf)
            # Compute delta features
            delta = torchaudio.functional.compute_deltas(mfcc)
            # Compute delta-delta features
            delta_delta = torchaudio.functional.compute_deltas(delta)
            # Reshape delta-delta to [1, 32, 1120]
            delta_delta = delta_delta[:, :, :1120] 
            if delta_delta.shape[2] < 1120:
                # If the resulting delta-delta features are too short in the time dimension, pad them
                delta_delta = torch.nn.functional.pad(delta_delta, (0, 1120 - delta_delta.shape[2]))
            delta_deltas.append(delta_delta)
        return torch.stack(delta_deltas)


    def load_audio_files_and_labels(self):
        categories = ['music_wav', 'speech_wav'] # music_wav = 0, speech_wav = 1
        files_and_labels = []
        for i, category in enumerate(categories):
            files_in_category = glob.glob(os.path.join(self.audio_dir, category, "*.wav"))
            for file_path in files_in_category:
                files_and_labels.append((file_path, i))
        return files_and_labels

    def __len__(self):
        return len(self.audio_files_and_labels)

    def __getitem__(self, idx):
        file_path, label = self.audio_files_and_labels[idx]
        mfccs = self.preprocess(file_path)  # preprocess the file to get the MFCCs

        # return 1 mfcc from chunks of mfccs and label
        for mfcc in mfccs:
            # print(f"mfcc.shape: {mfcc.shape}")
            return mfcc, label


    def _cut_if_necessary(self, signal):
        
        target_length = self.num_samples
        split_signals = []

        # Iterate over the signal in chunks of target_length
        for start in range(0, signal.shape[1], target_length):
            end = start + target_length
            if end <= signal.shape[1]:
                split_signals.append(signal[:, start:end])
            else:
                # If the last chunk is shorter than target_length, it can be discarded or padded
                # Here, we choose to pad the last chunk
                padding = target_length - (signal.shape[1] - start)
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


# path_to_test = "/Users/zainhazzouri/projects/Datapreprocessed/Bachelor_thesis_data/test/"
# #
# val_dataset = AudioProcessor(audio_dir=path_to_test)
# print(val_dataset[0][0].shape)