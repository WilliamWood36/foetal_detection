import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bss import function_pca
from file_reading import group_files_by_record, build_waveforms




import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, test_loader, device, num_samples=5):
    """ 
    Visualize model predictions for ECG signals.
    
    Args:
    - model: Trained PyTorch model
    - test_loader: DataLoader for test dataset
    - device: Torch device (cuda/cpu)
    - num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create a figure 
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    fig.suptitle('ECG Signal Predictions', fontsize=16)
    
    with torch.no_grad():
        for batch_idx, (ecg_data, labels) in enumerate(test_loader):
            if batch_idx >= 1:  # Only process first batch
                break
            
            ecg_data, labels = ecg_data.to(device), labels.to(device)
            outputs = model(ecg_data)
            
            # Convert to numpy for plotting
            ecg_np = ecg_data.cpu().numpy()
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # Plot for each sample
            for i in range(min(ecg_data.size(0), num_samples)):
                # Randomly select one channel
                channel = np.random.randint(0, ecg_np.shape[1])
                
                # Plot the original signal
                ax = axes[i] if num_samples > 1 else axes
                ax.plot(ecg_np[i, channel], label='Signal', color='blue', alpha=0.7)
                
                # Find true label indices (where labels > 0.5)
                true_label_indices = np.where(labels_np[i] > 0.5)[0]
                
                # Find predicted label indices (where outputs > 0.5)
                pred_label_indices = np.where(outputs_np[i] > 0.5)[0]
                
                # Plot true label markers as vertical lines
                for idx in true_label_indices:
                    ax.axvline(
                        x=idx, 
                        color='green', 
                        linestyle='--', 
                        linewidth=2, 
                        label='True QRS' if len(true_label_indices) > 0 else ''
                    )
                
                # Plot predicted label markers as vertical lines
                for idx in pred_label_indices:
                    ax.axvline(
                        x=idx, 
                        color='red', 
                        linestyle=':', 
                        linewidth=2, 
                        label='Predicted QRS' if len(pred_label_indices) > 0 else ''
                    )
                
                # Formatting
                ax.set_title(f'Sample {i}, Channel {channel+1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



def reshape_date(data,num_channels):
    if data.size % num_channels == 0:
        reshaped_data = data.reshape(-1, num_channels)
    else:
        raise ValueError("The data size is not compatible with the specified number of channels.")

    # Separate each channel into individual arrays
    channels = np.array([np.array(reshaped_data[:, i]) for i in range(num_channels)])
    return reshaped_data





def plot_n_components(signal, components):
    fig, axs = plt.subplots(components, 1)
    for index in range(0,components):
    # Step 3: Visualize the first two principal components (after PCA)
        #plt.plot(pca_result[:, 0][:1000], label='Principal Component 1')
        axs[index].plot(signal[:,index][:1000], label='Principal Component ')

        axs[index].set_title('PCA - First Two Principal Components')
        axs[index].set_xlabel('Samples')
        axs[index].set_ylabel('Amplitude')
        axs[index].legend()

    plt.show()


def basic_plot(signal_1,foetal_qrs, maternal_qrs):
    """
    
    Args: 
        signal_1 [int]: signal
        foetal_qrs
        maternal_qrs
    """
    print(signal_1.shape," ",maternal_qrs)
    # Step 3: Visualize the first two principal components (after PCA)
    plt.figure(figsize=(10, 6))
    

    
    plt.plot(signal_1[:500], label = "signal", color ='green')
    plt.scatter(foetal_qrs, [300] * foetal_qrs.size, color='red', label="foetal")
    plt.scatter(maternal_qrs, [500] * maternal_qrs.size, color='blue', label="maternal")
    # plt.plot(foetal_data[5][:150], label = "foetal")
    plt.title('basic plot')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Step 4: Print the explained variance ratio
