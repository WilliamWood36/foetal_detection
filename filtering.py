import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq



def remove_baseline_wander(signal, fs=1000, bcutoff=0.7, ucutoff=75, order=2):
    """
    Removes baseline wander and high-frequency noise using a bandpass filter.

    Parameters:
    - signal: Input signal (1D array)
    - fs: Sampling frequency (Hz)
    - bcutoff: Low cutoff frequency for baseline wander (Hz), typically 0.5–0.7 Hz
    - ucutoff: High cutoff frequency to remove high-frequency noise (Hz), e.g. 75 Hz
    - order: Filter order (default is 2)
    """
    nyquist = 0.5 * fs
    low = bcutoff / nyquist
    high = ucutoff / nyquist

    b, a = butter(order, [low, high], btype='band')  # Bandpass filter
    filtered = filtfilt(b, a, np.array(signal).reshape(1, -1)[0])
    return np.array(filtered).reshape(-1, 1)




import numpy as np
from scipy.spatial import KDTree

def sampen_kdtree(signal, m=2, r=0.2, max_entropy=3, fallback_signal=None):
    """
    Compute Sample Entropy using a KD-Tree.
    If the entropy is too high (i.e., signal is too noisy), it uses a fallback signal.
    
    Args:
        signal (np.ndarray): 1D input signal.
        m (int): Embedding dimension.
        r (float): Tolerance (as a fraction of signal std).
        max_entropy (float): Max allowed entropy before flagging as noisy.
        fallback_signal (np.ndarray): Optional less noisy signal to replace if entropy is too high.

    Returns:
        float: Sample entropy.
        np.ndarray: The signal used (original or fallback).
    """
    fallback_signal = np.zeros(len(signal))
    signal = np.array(signal)
    N = len(signal)
    std = np.std(signal)
    if std == 0:
        return 0.0, signal  # Flat signal

    # Form embedding vectors for m and m+1
    def _embed(sig, dim):
        return np.array([sig[i:i+dim] for i in range(N - dim + 1)])
    
    emb_m = _embed(signal, m)
    emb_m1 = _embed(signal, m+1)

    # Use KD-tree to count number of close neighbors
    def _count_matches(emb, tolerance):
        tree = KDTree(emb)
        count = 0
        for vec in emb:
            neighbors = tree.query_ball_point(vec, tolerance)
            count += len(neighbors) - 1  # exclude self-match
        return count

    tolerance = r * std
    B = _count_matches(emb_m, tolerance)
    A = _count_matches(emb_m1, tolerance)

    # Avoid division by zero
    if B == 0 or A == 0:
        entropy = np.inf
    else:
        entropy = -np.log(A / B)
    # Noise handling
    if entropy > max_entropy and fallback_signal is not None:
        print(f"[Warning] Signal entropy {entropy:.2f} exceeds max ({max_entropy}). Replacing with fallback.")
        plt.figure(figsize=(8, 3))
        plt.plot(signal, label=f"Noisy Signal (Entropy: {entropy:.2f})", color='red')
        plt.title("Noisy Signal (Too High SampEn)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return sampen_kdtree(fallback_signal, m, r, max_entropy)

    return entropy, signal




def lowPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0)
    z = signal.lfilter(b, a, data)
    return z

def highPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0, btype="highpass")
    z = signal.lfilter(b, a, data)
    return z





# data_titles = ['AECG1', 'AECG2','AECG3','AECG4']

# file_path = 'test_set/a26.csv'

# # Read the CSV file without skipping rows
# sample_size = 5000
# offset = 3000

# read_data = pd.read_csv(file_path)[offset:sample_size + offset]
# data = np.zeros(shape=(len(data_titles), sample_size))
# non_filtered_data = data
# for index,data_name in enumerate(data_titles):
#     z1 = highPassFilter(3, 20, read_data[data_name])
#     z = lowPassFilter(3, 30, z1)
#     data[index] = z
    
    
    


# #center the data
# for index in range(0,len(data_titles)):
#     data[index] = data[index] - np.mean(data[index])


# #compute dot product of the array and its transposition
# data_transp = np.transpose(data)
# data_square = np.dot(data,data_transp)

# eigen_vals, eigen_vects = LA.eig(data_square)

# print(eigen_vals)


# #eigen values in descending order
# sorted_indexs = np.argsort(-eigen_vals)
# eigen_vals_sorted = eigen_vals[sorted_indexs]
# eigen_vects_sorted = eigen_vects[:, sorted_indexs]

# p = eigen_vects_sorted

# yd = np.dot(data.T,p).T

# print(yd.shape)

# # Access the Elapsed_time and AECG1 columns
# try:
#     figure, axis = plt.subplots(len(data_titles) +1, 1, figsize=(10, 8))
    
#     # Original plot
#     for index, name in enumerate(yd):
#         x = read_data['Elapsed_time'][500:]
#         y = name[500:]
#         axis[index+1].set_title(eigen_vals_sorted[index])
#         axis[index+1].plot(x, y, color='blue')


#     x = read_data['Elapsed_time'][500:]
#     y = yd[0][500:] - yd[1][500:]
#     axis[0].plot(x,y,color = 'red')


    
#     plt.subplots_adjust(top=0.95)
#     plt.show()
# except KeyError as e:
#     print(f"KeyError: {e}. Please check the column names.")
