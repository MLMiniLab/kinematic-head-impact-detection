import torch
from torch import nn


def conv(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding="same"), nn.ReLU())


class RecursiveNet(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=199, dropout=0.25):
        super().__init__()
        self.c1 = conv(1, 32)
        self.b1 = nn.Sequential(conv(32, 32), conv(32, 32), nn.Dropout(dropout))
        self.c2 = conv(32, 64)
        self.b2 = nn.Sequential(conv(64, 64), conv(64, 64), nn.Dropout(dropout))
        self.b3 = nn.Sequential(conv(64, 64), conv(64, 64))
        # skip concatenations with x2 and x1, as in the Keras model
        self.m3 = nn.Sequential(conv(128, 64), nn.Dropout(dropout))
        self.b4 = nn.Sequential(conv(64, 32), conv(32, 32))
        self.m4 = nn.Sequential(conv(64, 32), nn.Dropout(dropout))
        self.dense = nn.Sequential(nn.Linear(32 * n_channels * n_timesteps, 256), nn.ReLU())
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x1 = self.c1(x.unsqueeze(1))
        x2 = self.c2(self.b1(x1))
        h = self.m3(torch.cat([self.b3(self.b2(x2)), x2], dim=1))
        h = self.m4(torch.cat([self.b4(h), x1], dim=1))
        return self.out(self.dense(h.flatten(1))).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self(x))
