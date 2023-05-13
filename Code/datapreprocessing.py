#!/usr/bin/env python
# coding: utf-8

# In[52]:


SAMPLE_RATE = 22050 # sample rate of the audio file
bit_depth = 16 # bit depth of the audio file
hop_length = 512
n_mfcc = 20 # number of MFCCs features
n_fft=1024, # window size
n_mels = 256 # number of mel bands to generate
win_length = None # window length
target_sample_rate = 22050 # target sample rate
num_samples = 22050 # number of samples
target_length = target_sample_rate




# In[53]:


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



# In[57]:




# In[61]:




class AudioProcessor(Dataset):
    def __init__(self, audio_dir, n_mfcc=n_mfcc, target_sample_rate=target_sample_rate,num_samples=num_samples):
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.device = self.get_device()
        self.music_waves = []
        self.speech_waves = []
        self.mix_waves = []
        self.silence_waves = []
        self.load_audio_files()
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.audio_files_and_labels = self.load_audio_files_and_labels()

    def load_audio_files(self):
        self.music_waves = glob.glob(os.path.join(self.audio_dir, "music_wav", "*.wav"))
        self.speech_waves = glob.glob(os.path.join(self.audio_dir, "speech_wav", "*.wav"))
        self.mix_waves = glob.glob(os.path.join(self.audio_dir, "Mix_wav", "*.wav"))
        self.silence_waves = glob.glob(os.path.join(self.audio_dir, "silence_wav", "*.wav"))
        print("Music waves:", self.music_waves)
        print("Speech waves:", self.speech_waves)
        print("Mix waves:", self.mix_waves)
        print("Silence waves:", self.silence_waves)

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"

    def preprocess(self, filepath, target_length= target_length, sample_rate=SAMPLE_RATE):
        waveform, _ = torchaudio.load(filepath)
        # waveform_length = waveform.size(1)
        #
        # if waveform_length < target_length:
        #     num_padding = target_length - waveform_length
        #     padding = torch.zeros(1, num_padding)
        #     waveform = torch.cat((waveform, padding), 1)
        # elif waveform_length > target_length:
        #     waveform = waveform[:, :target_length]

        waveform = self._resample_if_necessary(waveform, sample_rate) # resample if necessary
        waveform = self._mix_down_if_necessary(waveform) # convert stereo to mono
        waveform = self._right_pad_if_necessary(waveform) # pad if necessary
        waveform = self._cut_if_necessary(waveform) # cut if necessary

        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=self.n_mfcc)(waveform)
        return mfcc

    def load_audio_files_and_labels(self):
        categories = ['music_wav', 'speech_wav', 'Mix_wav', 'silence_wav'] # music_wav = 0, speech_wav = 1, Mix_wav = 2, silence_wav = 3
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
        waveform = self.preprocess(file_path)
        return waveform, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def play_audio_samples(self):
        music_sample = random.choice(self.music_waves)
        speech_sample = random.choice(self.speech_waves)
        mix_sample = random.choice(self.mix_waves)
        silence_sample = random.choice(self.silence_waves)

        print("Music sample:")
        ipd.display(ipd.Audio(music_sample))

        print("Speech sample:")
        ipd.display(ipd.Audio(speech_sample))

        print("Mix sample:")
        ipd.display(ipd.Audio(mix_sample))
        print("Silence sample:")
        ipd.display(ipd.Audio(silence_sample))

    def librosa_spectrogram(self, filepath):
        y, sr = librosa.load(filepath)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=self.n_mfcc)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S, sr

    def plot_single_spectrogram(self, axs, idx, mfcc_value, audio_file, audio_title):
        log_S, sr = self.librosa_spectrogram(audio_file)

        axs[idx, 0].set_title(f"Torchaudio {audio_title} MFCCs")
        img0 = axs[idx, 0].imshow(mfcc_value.squeeze().cpu().numpy().T, origin='lower', aspect='auto')
        axs[idx, 0].figure.colorbar(img0, ax=axs[idx, 0])

        axs[idx, 1].set_title(f"Librosa {audio_title} MFCCs")
        img1 = librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', ax=axs[idx, 1])
        axs[idx, 1].figure.colorbar(img1, ax=axs[idx, 1], format='%+2.0f dB')


    def plot_audio_spectrograms(self):
        audio_titles = ["Music", "Speech", "Mix", "Silence"]
        audio_files = [random.choice(self.music_waves), random.choice(self.speech_waves),
                       random.choice(self.mix_waves), random.choice(self.silence_waves)]
        audio_mfccs_values = [self.preprocess(file) for file in audio_files]

        fig, axs = plt.subplots(4, 2, figsize=(12, 16))

        for i, (mfcc_value, audio_file, audio_title) in enumerate(zip(audio_mfccs_values, audio_files, audio_titles)):
            self.plot_single_spectrogram(axs, i, mfcc_value, audio_file, audio_title)

        plt.tight_layout()
        plt.show()

    def process_and_visualize(self):
        self.play_audio_samples()
        self.plot_audio_spectrograms()



# In[62]:


if __name__ == '__main__':
    AUDIO_DIR = "/Users/zainhazzouri/projects/Bachelor_Thesis/Data/train"
    audio_processor = AudioProcessor(AUDIO_DIR)
    audio_processor.process_and_visualize()


# In[ ]:




