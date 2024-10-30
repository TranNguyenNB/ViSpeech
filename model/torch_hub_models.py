import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So,
        # filter is flipped.
        self.register_buffer('flipped_filter', torch.FloatTensor(
            [-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(
            input.size()) == 2, \
            'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class Hub_Model(nn.Module):
    def __init__(self, n_classes, n_mels=40, dropout_rate=0.3, sampling_rate=22050, n_fft=1024, window=552, step=220, model_type='resnet34', version='v0.9.0'):
        super(Hub_Model, self).__init__()

        self.instancenorm = nn.InstanceNorm1d(n_mels)

        self.torchfb = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=n_fft,
                win_length=window,
                hop_length=step,
                window_fn=torch.hamming_window,
                n_mels=n_mels))

        self.bb = torch.hub.load(
            f'pytorch/vision:{version}',
            model_type,
            pretrained=True,
            verbose=False)

        # Modify the first convolution layer of ResNet34
        self.bb.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        
        # Adding dropout after the ResNet backbone
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classifier with dropout before the linear layer
        self.classifier = torch.nn.Sequential(
            nn.Linear(1000, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        x = self.dropout(x)  # Apply dropout before classification
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

