import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from file_reading import dataset_builder, preprocess_ecg_data

# Define Fetal QRS Detector (Ensure this matches your existing model structure)
class FetalQRSWindowDetector(nn.Module):
    def __init__(self, in_channels=4):
        super(FetalQRSWindowDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * 100, 100)  # Adjust if needed

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc(x))  # Binary classification
        return x

# Load dataset
def load_data():
    window_length = 100
    stride = 50
    # Simulate a batch of 4-channel windowed data: shape (batch, 4, 100)
    raw_ecg_data, fqrs = dataset_builder("./data", [2,9,15,28])
    ecg_data = preprocess_ecg_data(raw_ecg_data,fqrs,window_length,stride)
    pure_ecg = []
    bin_labels = []
    for i in ecg_data:
        label = np.zeros(window_length)
        if i[1] != -1:
            start = max(0, i[1] - 2)
            end = min(window_length, i[1] + 3)
            label[start:end] = 1.0

        pure_ecg.append(i[0])
        bin_labels.append(label)

    pure_ecg = torch.from_numpy(np.array(pure_ecg, dtype=np.float32))
    bin_labels = torch.from_numpy(np.array(bin_labels, dtype=np.float32))

    
    batch_size = 20998


    dataset = TensorDataset(pure_ecg, bin_labels)
    test_size = batch_size // 4  # 25% for testing
    train_size = batch_size - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for ecg_data, labels in dataloader:
        ecg_data, labels = ecg_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(ecg_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Testing function
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for ecg_data, labels in dataloader:
            ecg_data, labels = ecg_data.to(device), labels.to(device)
            outputs = model(ecg_data)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset, test_dataset = load_data()

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
model = FetalQRSWindowDetector(in_channels=4).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, accuracy = test(model, test_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f} - "
          f"Test Loss: {test_loss:.4f} - "
          f"Accuracy: {accuracy:.2f}%")
