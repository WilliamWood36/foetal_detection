import torch
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("PyTorch Version:", torch.__version__)  # Should show +cu121, NOT +cpu
print("CUDA Available:", torch.cuda.is_available())  # Should be True
print("CUDA Version:", torch.version.cuda)  # Should show 12.x
print("GPU Name:", torch.cuda.get_device_name(0))  # Should print GTX 1660 Ti
print("cuDNN Available:", torch.backends.cudnn.is_available())  # Should be True


import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Device setup (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2000),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()  # Set model to training mode
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:.6f}  [{current}/{size}]")


# Testing loop
def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move to correct device
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100 * correct):.1f}%, Avg loss: {test_loss:.6f}\n")


# Initialize model, loss function, and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
