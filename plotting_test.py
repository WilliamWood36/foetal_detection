import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch


def surface_plot(x, y, z):
    # Create meshgrid
    X, Y = np.meshgrid(x, y)  # X and Y will both have shape (m, n)

    Z = np.array(z).T  # Ensure Z is also (m, n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k')

    # Labels
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Classification Threshold')
    ax.set_zlabel('F1-measure')

    # Optional: reverse x-axis if needed
    # ax.set_xlim(ax.get_xlim()[::-1])

    # Set view angle
    ax.view_init(elev=30, azim=135)

    # Color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




def base_results_plot(num_epochs, training_loss, Test_loss, f1):
    epochs = list(range(0,num_epochs))
    plt.plot(epochs, training_loss, label='Training Loss', color='red', marker='o')
    plt.plot(epochs,f1,label="F1",color='green',marker='o')
    # Accuracy plot
    plt.plot(epochs, Test_loss, label='Test_loss', color='blue', marker='o')

    plt.title('Training Loss and Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_predictions(outputs, signal, labels, num_samples=5):
    x = np.arange(len(signal))

    # Plot the signal
    plt.plot(x, signal, label="Signal")

    # Plot a dot at the given index
    plt.scatter(x, outputs, color="red", zorder=5, label=f"Dot at index ")
    plt.scatter(x, (labels ), color="blue", zorder=5, label=f"Dot at index ")
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    plt.title("Signal with Dot at a Specific Index")
    plt.legend()
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


def basic_plot(signal_1,foetal_qrs, maternal_qrs = [], length = 1000, signal_2 = []):
    """
    
    Args: 
        signal_1 [int]: signal
        foetal_qrs
        maternal_qrs
    """
    plt.figure(figsize=(12, 8))

    # --- First plot: signal_1 ---
    plt.subplot(2, 1, 1)
    plt.plot(signal_1[:length], label="signal 1", color='green')
    plt.scatter(foetal_qrs, [300] * len(foetal_qrs), color='red', label="foetal")
    if len(maternal_qrs) != 0:
        plt.scatter(maternal_qrs, [300] * len(maternal_qrs), color='blue', label="maternal")
    plt.title('Signal 1 with QRS Markers')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    if len(signal_2) != 0:
    # --- Second plot: signal_2 ---
        plt.subplot(2, 1, 2)
        plt.plot(signal_2[:length], label="signal 2", color='purple')
        plt.title('Signal 2')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Step 4: Print the explained variance ratio
# surface_plot(0,0,0)