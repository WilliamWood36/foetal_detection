import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import time


from plotting_test import visualize_predictions
from file_reading import dataset_builder, preprocess_ecg_data

# Define Fetal QRS Detector 
class FetalQRSWindowDetector(nn.Module):
    def __init__(self, in_channels=4, hidden_width = 2):
        super(FetalQRSWindowDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(2)
        self.fc = nn.Linear(2 * 100, 100)  # Adjust if needed

    def forward(self, x):
        # Ensure input is [batch, channels, length]
        x = x.view(-1, 4, 100)  # Reshape to correct input format
        x = torch.relu(self.bn2(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc(x))  # Binary classification
        return x

# Load dataset
def load_data():
    window_length = 100
    stride = 50
    # Simulate a batch of 4-channel windowed data: shape (batch, 4, 100)
    raw_ecg_data, fqrs = dataset_builder("./data", [2,9,15,28])
    ecg_data, labels = preprocess_ecg_data(raw_ecg_data, fqrs, window_length, stride)
    print("lables :",labels.shape," ecg_data:  ", ecg_data.shape)
    pure_ecg = []
    bin_labels = []
    for index,i in enumerate(ecg_data):
        # Ensure we only take the first 4 channels if more exist
        segment = i
        label = np.zeros(window_length)
        if labels[index] != -1:
            start = max(0, labels[index] - 2)
            end = min(window_length, labels[index] + 3)
            label[start:end] = 1.0

        pure_ecg.append(segment)
        bin_labels.append(label)

    print("lables :", np.array(bin_labels).shape," ecg_data:  ", np.array(pure_ecg).shape)

    pure_ecg = torch.from_numpy(np.array(pure_ecg, dtype=np.float32)).cuda()
    bin_labels = torch.from_numpy(np.array(bin_labels, dtype=np.float32)).cuda()

    # Create the dataset with actual length
    dataset = TensorDataset(pure_ecg, bin_labels)
    
    # Calculate split based on actual dataset length
    actual_size = len(dataset)
    test_size = actual_size // 4  # 25% for testing
    train_size = actual_size - test_size
    
    print(f"Dataset size: {actual_size}, Train size: {train_size}, Test size: {test_size}")
    print(f"Pure ECG shape: {pure_ecg.shape}, Bin labels shape: {bin_labels.shape}")
    
    # Now split with correct sizes
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

# The rest of the code remains the same...
# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    avg_loss = 0
    batches = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        # if avg_loss == 0:
        #     visualize_predictions(pred[0].cpu().detach().numpy()*1000, X[0][0].cpu().detach().numpy(), y.cpu().detach().numpy()[0])

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batches +=1
        avg_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return avg_loss/batches

def test(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in dataloader:
        pred = model(X)  # Model outputs 100 values per sample
        # Compute loss
        rand = np.random()
        if total_loss == 0:
            #index is random, has not effect on 
            visualize_predictions(pred[3].cpu().detach().numpy()*1000, X[0][0].cpu().detach().numpy(), y.cpu().detach().numpy()[3])

        # Convert predictions to binary (1 if feature detected, 0 otherwise)
        pred_binary = (pred > 0.5).float()  # Thresholding for binary detection

        total_loss += loss_fn(pred, y).item()
        # Count correctly detected feature positions (overlap with label)
        correct_preds = (pred_binary * y).sum(dim=1)  # Count overlapping 1s per sample
        total_correct += correct_preds.sum().item()   # Sum across batch
        total_samples += y.sum().item()  # Total number of expected feature points


# Compute metrics
    average_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"Test --  Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {average_loss:>8f} \n")
    return average_loss, accuracy


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
optimizer = optim.Adam(model.parameters(), lr=0.005)

start_time = time.time()  # Start the timer





# Training loop

batch_size = 24
num_epochs = 7


for epoch in range(num_epochs):
    train_loss = train(train_loader,model, criterion, optimizer)
    test_loss, accuracy = test(test_loader,model, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f} - "
          f"Test Loss: {test_loss:.4f} - "
          f"Accuracy: {accuracy:.2f}%")

end_time = time.time()  # End the timer
elapsed_time = end_time - start_time  # Compute elapsed time


# import matplotlib.pyplot as plt
# import numpy as np
# arr = [
#     [-68, -67, -65, -64, -63, -62, -61, -60, -58, -57, -56, -55,
#      -54, -53, -52, -51, -50, -50, -49, -48, -47, -46, -46, -45,
#      -44, -44, -43, -42, -42, -41, -40, -39, -34, -22, 5, 39,
#      46, 2, -59, -83, -62, -32, -17, -9, -16, -2, -25, 13,
#      -30, 260, 572, 1706, 932, -533, -33, -50, -3, -45, -22, -37,
#      -30, -32, -28, -22, -15, -4, 9, 26, 44, 64, 83, 99,
#      112, 120, 121, 116, 106, 90, 71, 51, 31, 13, -4, -17,
#      -28, -36, -42, -46, -49, -50, -51, -52, -53, -53, -53, -53,
#      -54, -54, -54, -54],

#     [-73, -73, -73, -72, -72, -72, -71, -71, -70, -70, -70, -69,
#      -69, -69, -68, -68, -68, -67, -67, -67, -66, -66, -66, -65,
#      -65, -65, -64, -64, -64, -63, -63, -61, -56, -40, -2, 51,
#      79, 44, -29, -76, -76, -58, -46, -37, -48, -26, -61, -4,
#      -65, 403, 906, 2123, 849, -972, -68, -97, -13, -76, -38, -61,
#      -47, -51, -44, -35, -24, -9, 10, 32, 56, 82, 106, 127,
#      143, 151, 150, 142, 126, 105, 80, 54, 29, 6, -13, -29,
#      -42, -51, -57, -61, -64, -66, -67, -68, -68, -69, -69, -69,
#      -69, -69, -70, -70],

#     [22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29,
#      30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35,
#      36, 36, 36, 37, 37, 37, 38, 38, 37, 30, 12, -20,
#      -58, -77, -65, -29, 10, 36, 48, 47, 55, 43, 64, 30,
#      68, -227, -572, -732, 82, 722, 55, 78, 18, 53, 28, 40,
#      31, 32, 26, 21, 14, 6, -5, -18, -31, -45, -58, -69,
#      -77, -81, -80, -75, -66, -54, -41, -27, -14, -3, 7, 14,
#      20, 24, 26, 28, 29, 30, 30, 30, 31, 31, 31, 31,
#      31, 31, 31, 31],

#     [50, 43, 37, 31, 25, 19, 13, 7, 2, -4, -9, -15,
#      -20, -25, -30, -35, -39, -44, -48, -52, -56, -60, -64, -67,
#      -70, -73, -75, -78, -80, -81, -82, -81, -75, -49, 9, 79,
#      82, -45, -232, -333, -295, -194, -105, -54, -40, -17, -40, 4,
#      46, 724, 1021, 1925, -178, -1588, -146, -176, 6, -90, -33, -71,
#      -51, -61, -55, -54, -55, -56, -62, -71, -87, -109, -138, -171,
#      -207, -242, -271, -292, -301, -299, -284, -260, -229, -195, -160, -128,
#      -101, -78, -61, -48, -39, -33, -30, -28, -26, -26, -25, -25,
#      -25, -25, -26, -26]
# ]

# print(np.array(arr).shape)
# arr_tensor = torch.tensor([arr], dtype=torch.float32)
# print(arr_tensor.shape)
# # Reshape to be (batch_size, channels, length) for Conv1d

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Move the model to the same device
# model = model.to(device)

# # Assuming arr_tensor is your input tensor
# arr_tensor = arr_tensor.to(device)

# # Now, forward pass should work
# out = model.forward(arr_tensor)

# print(out)

# arr = []
# outer = 0
# for ecg_data, labels in test_loader:
#     ecg_data, labels = ecg_data.to(device), labels.to(device)
#     print(labels.cpu().detach().numpy()[1])
#     outputs = model(ecg_data)
#     print(outputs)
#     arr = np.where(outputs.cpu().detach().numpy() > 0.5, 1.0, 0.0)
#     print(outputs.shape)
#     print(labels.cpu().detach().numpy().shape)
#     outer +=1

#     x = np.arange(100)

#     # Plot the signal
#     plt.plot(x, ecg_data[0][0].cpu().detach().numpy(), label="Signal")

#     # Plot a dot at the given index
#     plt.scatter(x, outputs[0].cpu().detach().numpy()*100, color="red", zorder=5, label=f"Dot at index ")
#     plt.scatter(x, labels.cpu().detach().numpy()[0]*150, color="blue", zorder=5, label=f"Dot at index ")
#     print(labels.cpu().detach().numpy()[0]*150)
#     plt.xlabel("Index")
#     plt.ylabel("Signal Value")
#     plt.title("Signal with Dot at a Specific Index")
#     plt.legend()
#     plt.show()

#     if outer > 5:
#         break








print(f"Execution Time: {elapsed_time:.6f} seconds")
