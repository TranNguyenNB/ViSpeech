import glob
import random
import torch
import torchaudio
import torchaudio.functional as F
import librosa
import numpy as np

class AugmentAudio:
    def __init__(self, rir_path, noise_path):
        self.rir_files = glob.glob(f"{rir_path}/*")
        self.noise_files = glob.glob(f"{noise_path}/*")

    def add_noise(self, audio):
        # noise = np.random.normal(0, random.choice([0.0008, 0.001, 0.002, 0.0025, 0.0015]), audio.shape)
        noise = np.random.normal(0, random.choice([0.0008, 0.001, 0.002, 0.0025, 0.0015]), audio.shape)
        return np.clip(audio + noise, -1.0, 1.0)

    def reverberate(self, audio):
        audio_tensor = torch.FloatTensor(audio).reshape([1, -1])
        rir, fs = torchaudio.load(self.rir_files[0])
        rir = rir[:, int(fs * 1.01) : int(fs * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        augmented = F.fftconvolve(audio_tensor, rir)[:, :audio_tensor.shape[1]].reshape(-1)
        return augmented

    def speed_perturb(self, file, sr, max_frames, window_len, step_size):
        speed_factor = random.choice([1.2, 0.8])

        audio, _ = librosa.load(file, sr=sr)
        audio_aug = librosa.effects.time_stretch(audio, rate=speed_factor)

        audiosize = audio_aug.shape[0]

        max_audio = max_frames * step_size + (window_len - step_size)
        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio_aug = np.pad(audio_aug, (0, shortage), 'wrap')

        audio_aug = audio_aug[:max_audio]

        return audio_aug
    
    def add_background_noise(self, audio, sr):
        file = random.choice(self.noise_files)
        
        noise, _ = librosa.load(file, sr=sr)

        if audio.shape[0] < noise.shape[0]:
            start = random.randint(0, int((noise.shape[0] - audio.shape[0])//10000))
            noise = noise[start*10000:start*10000 + audio.shape[0]]
        elif audio.shape[0] > noise.shape[0]:
            shortage = audio.shape[0] - noise.shape[0]
            noise = np.pad(noise, (0, shortage), 'wrap')

        aug = audio + 0.3 * noise

        return aug
