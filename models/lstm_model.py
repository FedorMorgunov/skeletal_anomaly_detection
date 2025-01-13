import torch.nn as nn

class SkeletonAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SkeletonAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # single score output

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # use last time step
        out = self.fc(last_output)  # (batch, 1)
        return out.squeeze(1)  # (batch,)