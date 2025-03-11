import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Device setup


# CNN Model for ECG QRS Detection
class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 250, 128)  # Adjust based on input window size
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Regression output: position of QRS

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)  # Output QRS position
        return x





def preprocess_ecg_data(ecg_data, qrs_positions, window_size=2000, stride=500):
    """
    Splits ECG data into overlapping windows and assigns QRS positions.
    
    Args:
    - ecg_data: (4, 750000) -> 4-channel ECG signals
    - qrs_positions: List of true QRS positions in the full signal
    - window_size: Number of samples per window
    - stride: Step size for sliding window
    
    Returns:
    - X: Processed ECG segments
    - y: QRS positions relative to window
    """
    num_windows = (ecg_data.shape[1] - window_size) // stride + 1
    X, y = [], []
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        segment = ecg_data[:, start:end]  # Extract window

        # Find QRS complex within the window
        qrs_in_window = [q - start for q in qrs_positions if start <= q < end]

        # Label: QRS position or -1 if no QRS found
        label = qrs_in_window[0] if qrs_in_window else -1

        X.append(segment)
        y.append(label)
    
    return np.array(X), np.array(y)





import torch.utils.data as data

# Convert to PyTorch Dataset
class ECGDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Training loop
def train(model, dataloader, loss_fn, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")




def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            total_loss += loss_fn(predictions.squeeze(), y_batch).item()
    
    print(f"Test Loss: {total_loss / len(dataloader):.6f}")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = ECG_CNN().to(device)


from file_reading import dataset_builder

# Example Usage
ecg_data = dataset_builder("./data,[]")  # Simulated ECG (4 channels, 750000 samples)
qrs_positions = [5000, 12000, 18000]  # Example QRS positions
X, y = preprocess_ecg_data(ecg_data, qrs_positions)

# Create Dataset
dataset = ECGDataset(X, y)
train_loader = data.DataLoader(dataset, batch_size=64, shuffle=True)

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Regression loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, loss_fn, optimizer)

evaluate(model, train_loader)
