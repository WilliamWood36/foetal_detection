import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from file_reading import load_challenge_data, connvert_to_difference_data
from plotting_test import basic_plot
from scipy.signal import iirnotch, filtfilt, butter
from test import FetalQRSWindowDetector, load_data, train_test_cycle
from fvcore.nn import FlopCountAnalysis

batch_size = 64
lr = 0.0005
epochs = 100
threashold = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_1, test_1 = load_data(challenge=True)
train_2, test_2 = load_data(challenge=False)

# Name mappings
dataset_labels = {
    id(train_1): "train_challenge",
    id(train_2): "train_synthetic",
    id(ConcatDataset([train_1, train_2])): "train_combined",
    id(test_1): "test_challenge",
    id(test_2): "test_synthetic",
    id(ConcatDataset([test_1, test_2])): "test_combined"
}

# Prepare result log
result_log = []

# Training and evaluation loop
for test in (test_1, test_2, ConcatDataset([test_1, test_2])):
    for train in (train_1, train_2, ConcatDataset([train_1, train_2])):
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
        model = FetalQRSWindowDetector(in_channels=4).to(device)

        results = []
        for i in range(5):  # Run once per combination
            result = train_test_cycle((train_loader, test_loader), model,
                                      batch_size=batch_size, lr=lr,
                                      epochs=epochs, threshold=threashold)
            results.append(result)

        mean_result = np.mean(results)
        train_name = dataset_labels.get(id(train), "train_unknown")
        test_name = dataset_labels.get(id(test), "test_unknown")
        result_str = f"Train: {train_name}, Test: {test_name}, F1: {mean_result:.4f}"

        print(result_str)
        result_log.append(result_str)

# Write to file
with open("results_log.txt", "w") as f:
    f.write("\n".join(result_log))

print("\n✅ All results saved to 'results_log.txt'.\n")

