import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# Function to apply low-pass filter using Butterworth filter
def lowPassFilter(order, cutoff, data, fs=1000.0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    z = signal.lfilter(b, a, data)
    return z

# Function to apply high-pass filter using Butterworth filter
def highPassFilter(order, cutoff, data, fs=1000.0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    z = signal.lfilter(b, a, data)
    return z

# Data loading and preprocessing

data_titles = ['AECG1', 'AECG2', 'AECG3', 'AECG4']  # ECG signal names
file_path = 'a26.csv'

# Read a portion of the CSV file
sample_size = 5000  # Number of samples to read
offset = 6000  # Offset to start reading

read_data = pd.read_csv(file_path)[offset:sample_size + offset]  # Read data from CSV

# Initialize empty array for processed data
data = np.zeros(shape=(len(data_titles), sample_size))

# Apply filters to each of the ECG signals
for index, data_name in enumerate(data_titles):
    # Apply high-pass filter to remove baseline drift and low-frequency noise (cutoff=20 Hz)
    z1 = highPassFilter(3, 20, read_data[data_name])
    
    # Apply low-pass filter to remove high-frequency noise (cutoff=30 Hz)
    z = lowPassFilter(3, 30, z1)
    
    # Store the filtered data
    data[index] = z

# Center the data by subtracting the mean
for index in range(0, len(data_titles)):
    data[index] = data[index] - np.mean(data[index])

# Step 1: Standardize the data (important for ICA)
# ICA is sensitive to the variance of each feature (channel). 
# Hence, we standardize the data to have zero mean and unit variance.
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data.T).T  # Transpose for standardizing across samples

# Step 2: Apply ICA
ica = FastICA(n_components=4, random_state=0)  # Choose the number of components you want (here, 4 for all channels)
ica_result = ica.fit_transform(data_standardized.T)  # Perform ICA

# Step 3: Plot the independent components obtained from ICA
plt.figure(figsize=(10, 6))
for i in range(4):  # We plot the first 4 independent components
    plt.subplot(4, 1, i + 1)
    plt.plot(ica_result[:, i], label=f'Independent Component {i + 1}')
    plt.title(f'Independent Component {i + 1}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout()
plt.show()
