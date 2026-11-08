import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(16 * 2 * 2, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x).squeeze(1)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))
