
import torch
from torch.utils.data import DataLoader
from file_reading import load_challenge_data, connvert_to_difference_data
from plotting_test import basic_plot
from scipy.signal import iirnotch, filtfilt, butter
from test import FetalQRSWindowDetector, load_data, train_test_cycle

batch_size = 64
lr = 0.003
epochs = 100
threashold = 0.2
# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset, test_dataset = load_data(challenge=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, optimizer
model = FetalQRSWindowDetector(in_channels=4).to(device)

train_test_cycle((train_loader,test_loader), model, batch_size=batch_size,lr=lr,epochs=epochs,threshold=threashold)

