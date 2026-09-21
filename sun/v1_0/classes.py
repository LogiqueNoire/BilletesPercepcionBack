import torch
import torch.nn as nn

# NUEVA CELDA — CNN ENCODER
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# NUEVA CELDA — GRU AUTOENCODER
class GRUAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim=64,
        hidden_dim=32
    ):

        super().__init__()

        self.cnn = CNNEncoder(feature_dim)

        self.encoder_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=feature_dim,
            batch_first=True
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = []

        # CNN FRAME FEATURES
        for t in range(T):
            ft = self.cnn(x[:, t])
            features.append(ft)
        features = torch.stack(features, dim=1)

        # ENCODER
        _, h = self.encoder_gru(features)

        # REPEAT LATENT VECTOR
        repeated = h.repeat(T,1,1).permute(1,0,2)

        # DECODER
        reconstructed, _ = self.decoder_gru(repeated)

        return reconstructed, features