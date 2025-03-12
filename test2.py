import numpy as np


# Example usage

import torch
import torch.nn as nn
import torch.nn.functional as F

class FetalQRSWindowDetector(nn.Module):
    def __init__(self, in_channels=4):
        super(FetalQRSWindowDetector, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        # Second convolutional block
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # Third convolutional block
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        # Final layer reduces channels to 1 for per-sample predictions
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, 4, 100)
        
        Returns:
            Tensor: Per-sample probabilities of shape (batch, 100)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # -> (batch, 16, 100)
        x = F.relu(self.bn2(self.conv2(x)))  # -> (batch, 32, 100)
        x = F.relu(self.bn3(self.conv3(x)))  # -> (batch, 64, 100)
        x = self.conv_final(x)               # -> (batch, 1, 100)
        x = x.squeeze(1)                     # -> (batch, 100)
        return torch.sigmoid(x)              # probabilities between 0 and 1

from file_reading import dataset_builder,preprocess_ecg_data

# Example usage:
if __name__ == '__main__':
    window_length = 100

    # Simulate a batch of 4-channel windowed data: shape (batch, 4, 100)
    raw_ecg_data, fqrs = dataset_builder("./data", [2,9,15,28])
    ecg_data, int_labels= preprocess_ecg_data(raw_ecg_data,fqrs,window_length,50)

   
    batch_size = ecg_data.shape[0]
    # Simulate corresponding binary labels for each window
    # Here we randomly assign a QRS complex to about 50% of the windows
    bin_labels = torch.zeros(batch_size, window_length)
    for i in range(batch_size):
        if torch.rand(1) > 0.5:  # 50% chance that this window has a QRS
            qrs_idx = torch.randint(0, window_length, (1,)).item()
            # Optionally mark a few samples around the QRS index:
            start = max(0, qrs_idx - 2)
            end = min(window_length, qrs_idx + 3)
            bin_labels[i, start:end] = 1.0

    # Instantiate and test the model
    model = FetalQRSWindowDetector(in_channels=4)
    print(ecg_data.shape)

    ecg_data = torch.tensor(ecg_data, dtype=torch.float32)  # Ensure correct dtype



    output = model(ecg_data)
    print("Output shape:", output.shape)  # Expected: (batch_size, 100)

    # Example loss using binary cross-entropy
    criterion = nn.BCELoss()
    loss = criterion(output, bin_labels)
    print("Loss:", loss.item())
