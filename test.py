import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler

import time
from sklearn import preprocessing
window_size = 150
data_path = "test_set_diff"


from plotting_test import visualize_predictions, base_results_plot
from file_reading import dataset_builder, preprocess_ecg_data, load_challenge_data

# Define Fetal QRS Detector 
class FetalQRSWindowDetector(nn.Module):
    def __init__(self, in_channels=4, hidden_width=32, dropout_prob=0.2):
        super(FetalQRSWindowDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_width, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_width)
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.fc = nn.Linear(hidden_width * window_size, window_size)  # Adjust if needed

    def forward(self, x):
        # Ensure input is [batch, channels, length]
        x = x.view(-1, 4, window_size)  # Reshape to correct input format
        x = torch.relu(self.bn2(self.conv1(x)))
        x = self.dropout(x)  # Apply dropout after convolution
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.sigmoid(self.fc(x))  # Binary classification
        return x
    
# Load dataset
def load_data(challenge = True):
    window_length = window_size
    stride = 150
    # Simulate a batch of 4-channel windowed data: shape (batch, 4, 100)
    if challenge:
        raw_ecg_data, fqrs = load_challenge_data(directory="test_set_filtered")
    else:
        raw_ecg_data, fqrs = dataset_builder("./data", [2,9,15,28])
    print("lables :",len(fqrs)," ecg_data:  ", raw_ecg_data.shape)

    ecg_data, labels = preprocess_ecg_data(raw_ecg_data, fqrs, window_length, stride)
    print("lables :",labels.shape," ecg_data:  ", ecg_data.shape)
    pure_ecg = []
    bin_labels = []
    for index,i in enumerate(ecg_data):
        # Ensure we only take the first 4 channels if more exist
        segment = i
        label = np.zeros(window_length)
        if labels[index] != -1:
            start = int(max(0, labels[index] - 3))
            end = int(min(window_length, labels[index] + 2))
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
def train(dataloader, model, loss_fn, optimizer,batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    avg_loss = 0
    batches = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        if avg_loss == 0:
            print(len(pred[0].cpu().detach().numpy()))
            visualize_predictions(pred[0].cpu().detach().numpy(), X[0][0].cpu().detach().numpy(), y.cpu().detach().numpy()[0])
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

def test(dataloader, model, loss_fn, threshold=0.2):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            # Apply threshold for binary classification
            pred_binary = (pred > threshold).float()
            
            # Collect predictions and labels for F1 score computation
            all_labels.append(y.detach().cpu().numpy())
            all_preds.append(pred_binary.detach().cpu().numpy())
            
            total_correct += (pred_binary * y).sum().item()
            total_samples += y.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Compute F1 score (flatten arrays)
    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"Test -- Accuracy: {accuracy*100:.2f}%, Avg Loss: {avg_loss:.6f}, F1 Score: {f1:.3f}")
    return avg_loss, accuracy, f1



def train_test_cycle(loaders, model, criterion=nn.BCELoss(), batch_size=64, threshold=0.2, lr=0.001, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Add a learning rate scheduler (e.g., ReduceLROnPlateau)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.9)
    
    start_time = time.time()  # Start the timer
    progressive_accuracy, progressive_loss, progressive_f1 = [], [], []
    
    for epoch in range(epochs):
        criterion = nn.BCELoss()
        train_loss = train(loaders[0], model, criterion, optimizer, batch_size)
        test_loss, accuracy, f1_score = test(loaders[1], model, criterion, threshold)

        # Step the scheduler based on test loss
        #scheduler.step(f1_score)  # Adjust the learning rate based on the test loss

        print(f"Epoch [{epoch + 1}/{epochs}] - " +
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, " +
              f"Accuracy: {accuracy * 100:.2f}%, F1: {f1_score:.3f}")
        
        progressive_accuracy.append(accuracy)
        progressive_loss.append(test_loss)
        progressive_f1.append(f1_score)

    base_results_plot(epochs, progressive_loss, progressive_accuracy, progressive_f1)
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Compute elapsed time
    print(f"Training completed in {elapsed_time:.2f} seconds")












# # Main execution
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load datasets
# train_dataset, test_dataset = load_data(challenge=True)

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Initialize model, loss, optimizer
# model = FetalQRSWindowDetector(in_channels=4).to(device)

# criterion = nn.BCELoss()







# # Training loop

# batch_size = 32
# num_epochs = 200



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








