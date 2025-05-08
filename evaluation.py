
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from file_reading import load_challenge_data, connvert_to_difference_data
from plotting_test import basic_plot, visualize_predictions
from scipy.signal import iirnotch, filtfilt, butter
from test import FetalQRSWindowDetector, load_data, train_test_cycle
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import f1_score


def initialise_model(file_loc = "best1.pt"):
    # Main execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets

    # 1. Recreate the model architecture
    model = FetalQRSWindowDetector(in_channels=4).to(device)

    # 2. Load saved weights
    model.load_state_dict(torch.load(file_loc, map_location=device))
    model.eval()
    return model



def evaluate_model(model, test_loader, criterion, threshold=0.5, device="cpu", visualize=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs > threshold).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Visualize predictions from the first batch
            if visualize:
                sample_idx = 0
                signal = inputs[sample_idx, 0].cpu().numpy()  # Channel 0
                #visualize_predictions(preds[sample_idx], signal, labels[sample_idx].cpu().numpy())
                  # Only visualize one batch

    # Compute overall metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Ensure inputs are binary
    all_preds_bin = (all_preds > 0.5).astype(int)
    all_labels_bin = (all_labels > 0.5).astype(int)

    # Compute metrics
    f1 = f1_score(all_labels_bin, all_preds_bin, zero_division=0)
    acc = (all_preds_bin == all_labels_bin).mean()

    print(f"✅ Evaluation - Test Loss: {total_loss:.4f}, Accuracy: {acc * 100:.2f}%, F1 Score: {f1:.4f}")
    return f1, acc, total_loss
# 4. Run evaluation




batch_size = 64

train_dataset, test_dataset = load_data(challenge=True)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 3. Evaluate on the test data
criterion = nn.BCELoss()
threshold = 0.3  # or your preferred threshold

model = initialise_model()

evaluate_model(model, test_loader, criterion, threshold)

train_dataset, test_dataset = load_data(challenge=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

evaluate_model(model,test_loader,criterion,threshold)
