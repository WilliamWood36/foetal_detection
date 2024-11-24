import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
from bss import function_pca


def lowPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0)
    z = signal.lfilter(b, a, data)
    return z

def highPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0, btype="highpass")
    z = signal.lfilter(b, a, data)
    return z




def reshape_date(data,num_channels):
    if data.size % num_channels == 0:
        reshaped_data = data.reshape(-1, num_channels)
    else:
        raise ValueError("The data size is not compatible with the specified number of channels.")

    # Separate each channel into individual arrays
    channels = np.array([np.array(reshaped_data[:, i]) for i in range(num_channels)])
    return reshaped_data


# Path to the binary files
foetal_file_path = 'data/sub01_snr00dB_l1_c0_fecg1.dat'
foetal_qrs_path = 'data/sub01_snr00dB_l1_c0_fecg1.qrs'
maternal_file_path = 'data/sub01_snr00dB_l1_c0_mecg.dat'
noise_file_path = 'data/sub01_snr00dB_l1_c0_noise1.dat'


# Define the number of channels (32 ECG + 2 reference)
num_channels = 34

# Read the binary file as 16-bit integers
foetal_data = reshape_date(np.fromfile(foetal_file_path, dtype=np.int16), num_channels)[:-2]
maternal_data = reshape_date(np.fromfile(maternal_file_path, dtype=np.int16),num_channels)[:-2]
noise_data = reshape_date(np.fromfile(noise_file_path, dtype=np.int16),num_channels)[:-2]

foetal_qrs = reshape_date(np.fromfile(noise_file_path, dtype=np.int16),num_channels)



data = maternal_data + foetal_data + noise_data

for (index,unfiltered_channel)  in enumerate(data):
    z1 = highPassFilter(3, 20, unfiltered_channel)
    z = lowPassFilter(3, 30, z1)
    data[index] = z

# Define the number of channels (32 ECG + 2 reference)
num_channels = 32
num_components = 3

(pca_result,eigen) = function_pca(data, num_channels,num_components)

fig, axs = plt.subplots(num_components, 1)
for index in range(0,num_components):
# Step 3: Visualize the first two principal components (after PCA)
    #plt.plot(pca_result[:, 0][:1000], label='Principal Component 1')
    axs[index].plot(pca_result[:,index][:1000], label='Principal Component ')

    axs[index].set_title('PCA - First Two Principal Components')
    axs[index].set_xlabel('Samples')
    axs[index].set_ylabel('Amplitude')
    axs[index].legend()

plt.show()

# Step 4: Print the explained variance ratio
print(f"Explained variance ratio of the first component: {eigen.explained_variance_ratio_[0]:.4f}")
print(f"Explained variance ratio of the second component: {eigen.explained_variance_ratio_[1]:.4f}")
