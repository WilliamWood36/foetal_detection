import torch
import sys
import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
base_directory_path = "./data"  # Replace with the path to your base folder

# Now import the module
from datasets.multiChannel import ECGDataset
from file_reading import dataset_builder, split_array_randomly
from DL.D1_Cnn import NeuralNetwork, test_loop, train_loop


# Check if CUDA (NVIDIA GPU) is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ecg_datab is a list of tuples where each tuple is (mECG array, fECG value)
ecg_data = dataset_builder(base_directory_path,[1,5,8,15])
train_data, test_data = split_array_randomly(ecg_data)

train_dataloader = torch.utils.data.DataLoader(ECGDataset(train_data), batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(ECGDataset(test_data), batch_size=32, shuffle=True)

print("test data size: ", train_dataloader.__len__() ,"\nTrain data size: ", test_dataloader.__len__())


# Create model and move it to the correct device
model = NeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.7)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



# Print model architecture
print(model)