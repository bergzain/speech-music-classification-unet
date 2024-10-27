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
    def __init__(self, audio_dir, n_mfcc,length_in_seconds,type_of_transformation):
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.device = self.get_device()
        target_length =  length_in_seconds * sample_rate
        self.music_waves = []
        self.speech_waves = []
        self.load_audio_files()
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_length
        self.speech_count = 0
        self.music_count = 0
        self.speech_chunk_count = 0
        self.music_chunk_count = 0
        self.audio_files_and_labels = self.load_audio_files_and_labels()
        self.type_of_transformation = type_of_transformation




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
        waveform = self._resample_if_necessary(waveform, sample_rate)  # resample if necessary
        waveform = self._mix_down_if_necessary(waveform)  # convert stereo to mono
        waveform = self._right_pad_if_necessary(waveform)  # pad if necessary
        waveforms = self._cut_if_necessary(waveform)  # cut if necessary
        features = []
        for wf in waveforms:
            if self.type_of_transformation == 'MFCC':
                feature = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate, n_mfcc=self.n_mfcc)(wf)
            elif self.type_of_transformation == 'LFCC':
                feature = torchaudio.transforms.LFCC(sample_rate=self.target_sample_rate, n_lfcc=self.n_mfcc)(wf)
            elif self.type_of_transformation == 'delta':
                feature = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate, n_mfcc=self.n_mfcc)(wf)
                feature = torchaudio.functional.compute_deltas(feature)
            elif self.type_of_transformation == 'delta-delta':
                feature = torchaudio.transforms.MFCC(sample_rate=self.target_sample_rate, n_mfcc=self.n_mfcc)(wf)
                delta = torchaudio.functional.compute_deltas(feature)
                feature = torchaudio.functional.compute_deltas(delta)
            elif self.type_of_transformation == 'lfcc-delta':
                feature = torchaudio.transforms.LFCC(sample_rate=self.target_sample_rate, n_lfcc=self.n_mfcc)(wf)
                feature = torchaudio.functional.compute_deltas(feature)
            elif self.type_of_transformation == 'lfcc-delta-delta':
                feature = torchaudio.transforms.LFCC(sample_rate=self.target_sample_rate, n_lfcc=self.n_mfcc)(wf)
                delta = torchaudio.functional.compute_deltas(feature)
                feature = torchaudio.functional.compute_deltas(delta)
            else:
                raise ValueError(f"Unknown transformation type: {self.type_of_transformation}")

            temp = feature.shape[2] / 112
            temp = round(temp)
            if temp == 0:
                temp = 1
            temp = temp * 112
            if feature.shape[2] > temp:
                feature = feature[:, :, :temp]
            elif feature.shape[2] < temp:
                feature = torch.nn.functional.pad(feature, (0, temp - feature.shape[2]))

            features.append(feature)
        return torch.stack(features)



    def load_audio_files_and_labels(self):
        categories = ['music_wav', 'speech_wav'] # music_wav = 0, speech_wav = 1
        files_and_labels = []
        for i, category in enumerate(categories):
            files_in_category = glob.glob(os.path.join(self.audio_dir, category, "*.wav"))
            for file_path in files_in_category:
                files_and_labels.append((file_path, i))
        return files_and_labels

    def count_chunks(self, files_and_labels):
        """Counts the number of chunks for each category (music and speech).

        Args:
            files_and_labels: A list of tuples, where each tuple contains the file path and its corresponding label.

        Returns:
            None
        """
        for file_path, label in files_and_labels:
            if label == 0:  # music_wav
                self.music_count += 1
                # Increment music chunk count after preprocessing
                self.music_chunk_count += len(self.preprocess(file_path))
            else:  # speech_wav
                self.speech_count += 1
                # Increment speech chunk count after preprocessing
                self.speech_chunk_count += len(self.preprocess(file_path))

    def __len__(self):
        return len(self.audio_files_and_labels)

    def __getitem__(self, idx):
        file_path, label = self.audio_files_and_labels[idx]
        mfccs = self.preprocess(file_path)  # preprocess the file to get the MFCCs

        # return 1 mfcc from chunks of mfccs and label
        for mfcc in mfccs:
            # print(f"mfcc.shape: {mfcc.shape}")
            if label == 0:
                self.music_chunk_count += 1
            else:
                self.speech_chunk_count += 1
            return mfcc, label

    
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
