import torch
from torch import nn

HEADS = ["gap", "gmp", "dense"]


def conv1d_block(in_ch, filters, kernel, dropout):
    return nn.Sequential(
        nn.Conv1d(in_ch, filters, kernel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(dropout),
    )


class HIKNet(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=199, filters=150, kernel=15, dropout=0.4, head="gap"):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        self.head = head
        self.features = nn.Sequential(
            conv1d_block(n_channels, filters, kernel, dropout),
            conv1d_block(filters, filters, kernel, dropout),
        )
        # late 2D conv treats the (time, filter) map as a single channel image, like the Keras reshape
        self.conv2d = nn.Conv2d(1, filters, (3, kernel), stride=(3, 1))
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            h = self._conv(torch.zeros(1, n_channels, n_timesteps))
        n_in = h[0].numel() if head == "dense" else filters
        self.out = nn.Linear(n_in, 1)

    def _conv(self, x):
        h = self.features(x)
        h = h.transpose(1, 2).unsqueeze(1)
        return torch.relu(self.conv2d(h))

    def forward(self, x):
        h = self._conv(x)
        if self.head == "gap":
            h = h.mean(dim=(2, 3))
        elif self.head == "gmp":
            h = h.amax(dim=(2, 3))
        else:
            h = h.flatten(1)
        return self.out(self.dropout(h)).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self(x))
