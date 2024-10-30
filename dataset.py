import pandas as pd
import torch
from augmentation.augment import AugmentAudio
import numpy as np
import random
import librosa

class VoiceDataset():
    def __init__(self, dataset, audio_length, sampling_rate, window_len, step_size, task_type, enable_augmentation=False, rir_path=None, noise_path=None):
        self.max_frames = audio_length * 100
        self.sampling_rate = sampling_rate
        self.window_len, self.step_size = window_len, step_size
        self.task_type = task_type  # Store the task type (gender or dialect)
        self.augment = enable_augmentation
        self.augment_audio = AugmentAudio(rir_path, noise_path) if enable_augmentation else None

        df = pd.read_csv(dataset, header=None)
        self.file_list = df[0].tolist()
        self.label_list = df[1].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.get_label(self.label_list[idx])

        # Load the base audio
        audio = self.load_audio(filename)

        # Apply augmentation if enabled
        if self.augment:
            augtype = random.randint(0, 4)
            augment_methods = {
                1: self.augment_audio.add_noise,
                2: self.augment_audio.reverberate,
                3: lambda _: self.augment_audio.speed_perturb(
                    filename, sr=self.sampling_rate, 
                    max_frames=self.max_frames, 
                    step_size=self.step_size, window_len=self.window_len
                ),
                4: lambda x: self.augment_audio.add_background_noise(
                    x, sr=self.sampling_rate
                ),
            }
            if augtype > 0:
                audio = augment_methods[augtype](audio)
                
        return torch.FloatTensor(audio), label
    
    def get_label(self, original_label):
        """Determine the label based on the task type."""
        if self.task_type == "gender":
            return int(original_label // 3) 
        elif self.task_type == "dialect":
            return int(original_label % 3)  
    
    def load_audio(self, filename):
        """Load and pad/truncate audio to fit the required length."""
        max_audio = self.max_frames * self.step_size + (self.window_len - self.step_size)
        audio, _ = librosa.load(filename, sr=self.sampling_rate)

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')

        return audio[:max_audio]