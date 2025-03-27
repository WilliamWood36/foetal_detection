import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bss import function_pca
from file_reading import group_files_by_record, build_waveforms




import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(outputs, signal, labels, num_samples=5):
    x = np.arange(len(signal))

    # Plot the signal
    plt.plot(x, signal, label="Signal")

    # Plot a dot at the given index
    plt.scatter(x, outputs*1, color="red", zorder=5, label=f"Dot at index ")
    plt.scatter(x, (labels*1500 ), color="blue", zorder=5, label=f"Dot at index ")
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
