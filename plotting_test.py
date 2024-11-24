import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bss import function_pca




def reshape_date(data,num_channels):
    if data.size % num_channels == 0:
        reshaped_data = data.reshape(-1, num_channels)
    else:
        raise ValueError("The data size is not compatible with the specified number of channels.")

    # Separate each channel into individual arrays
    channels = np.array([np.array(reshaped_data[:, i]) for i in range(num_channels)])
    return channels

# Path to the binary file
foetal_file_path = 'data/sub01_snr00dB_l1_c0_fecg1.dat'
maternal_file_path = 'data/sub01_snr00dB_l1_c0_mecg.dat'

# Define the number of channels (32 ECG + 2 reference)
num_channels = 34

# Read the binary file as 16-bit integers
foetal_data = reshape_date(np.fromfile(foetal_file_path, dtype=np.int16), num_channels)
maternal_data = reshape_date(np.fromfile(maternal_file_path, dtype=np.int16),num_channels)

data = maternal_data + foetal_data






print(data.shape)
# Step 3: Visualize the first two principal components (after PCA)
plt.figure(figsize=(10, 6))

plt.plot(np.abs(foetal_data[5][:150]), label = "foetal")
plt.plot(np.abs(foetal_data[1][:150]), label = "foetal")
# plt.plot(foetal_data[5][:150], label = "foetal")
plt.title('PCA - First Two Principal Components')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Step 4: Print the explained variance ratio
