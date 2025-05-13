import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from file_reading import load_challenge_data, connvert_to_difference_data
from plotting_test import base_results_plot_vari, basic_plot
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
train_loader = DataLoader(ConcatDataset([train_1,train_2]), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ConcatDataset([test_1,train_2]), batch_size=batch_size, shuffle=True)

train_losses_all = []
test_losses_all = []
f1_scores_all = []

for i in range(5):
    model = FetalQRSWindowDetector(in_channels=4).to(device)
    train_loss, test_loss, f1_score = train_test_cycle(
        (train_loader, test_loader), model,
        batch_size=batch_size, lr=lr,
        epochs=epochs, threshold=threashold,
    )
    train_losses_all.append(train_loss)
    test_losses_all.append(test_loss)
    f1_scores_all.append(f1_score)

base_results_plot_vari(epochs, train_losses_all, test_losses_all, f1_scores_all)