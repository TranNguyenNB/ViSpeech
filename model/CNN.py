import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# Custom pre-emphasis filter as mentioned in the original code
class PreEmphasis_1(torch.nn.Module):
    def __init__(self, coeff=0.97):
        super().__init__()
        self.coeff = coeff
        self.flipped_filter = torch.FloatTensor([-coeff, 1.]).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # Ensure the filter is on the same device as the input
        self.flipped_filter = self.flipped_filter.to(x.device)
        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), mode='reflect')
        return torch.nn.functional.conv1d(x, self.flipped_filter).squeeze(1)

    
class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(CNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) 
        self.bn4 = nn.BatchNorm2d(512) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 124, 1024)  # Adjust input size for new layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes) 

        # Dropouts
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after first pooling
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after second pooling
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout after third pooling
        self.dropout4 = nn.Dropout(dropout_rate)  # Dropout after fourth pooling (new)
        self.dropout_fc = nn.Dropout(dropout_rate)  # Dropout before the final FC layer

    def forward(self, x):
        x = self.pool(self.bn1(nn.ReLU()(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.bn2(nn.ReLU()(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.bn3(nn.ReLU()(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool(self.bn4(nn.ReLU()(self.conv4(x))))  
        x = self.dropout4(x)

        x = self.flatten(x)
        x = self.dropout_fc(nn.ReLU()(self.fc1(x)))
        x = self.dropout_fc(nn.ReLU()(self.fc2(x)))
        x = self.dropout_fc(nn.ReLU()(self.fc3(x)))
        x = self.fc4(x)  # Final layer without dropout

        return x


class Custom_CNN(nn.Module):
    def __init__(self, n_classes, n_mels=40, dropout_rate=0.3, sampling_rate=22050, n_fft=1024, window=552, step=220):
        super(Custom_CNN, self).__init__()
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torch.nn.Sequential(
            PreEmphasis_1(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=n_fft,
                win_length=window,
                hop_length=step,
                window_fn=torch.hamming_window,
                n_mels=n_mels))

        self.bb = CNN(n_classes, dropout_rate)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfb(x) + 1e-6
            x = x.log()
            delta = torchaudio.functional.compute_deltas(x)
            delta2 = torchaudio.functional.compute_deltas(delta)
            x = torch.stack(
                (self.instancenorm(x),
                 self.instancenorm(delta),
                 self.instancenorm(delta2)), dim=1)

        x = self.bb(x)

        return F.log_softmax(x, dim=1)
