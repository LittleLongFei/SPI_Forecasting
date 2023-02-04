

# 2023-2-3 written by H.Zhang.

import torch
from torch import nn
from torch.nn import functional as F



# ------------------------------------------------------------------------------------- Long Short-Term Memory Network.

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=6, num_layers=3, batch_first=True)
        self.fc = nn.Linear(in_features=6, out_features=1)

    def forward(self, x):
        # -> x is input, size (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # -> x is output, size (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(-1, 1, 1)
        return x



# ------------------------------------------------------------------------------------- Artificial Neural Network.

class ANNet(nn.Module):
    def __init__(self):
        super(ANNet, self).__init__()

        self.nn = torch.nn.Sequential(
            nn.Linear(1 * 8, 128),
            torch.nn.ReLU(),
            nn.Linear(128, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 128),
            torch.nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x(batch_size, seq_len, input_size)
        x = x.view(x.shape[0], -1)
        x = self.nn(x)
        return x



