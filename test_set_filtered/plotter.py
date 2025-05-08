import pandas as pd
import matplotlib.pyplot as plt

# Load the ECG data
ecg_data = pd.read_csv("test_set_filtered/a56.csv", header=None)

# Load the annotations (R-peak locations in samples)
with open("test_set_filtered/a56.fqrs.txt", "r") as file:
    annotations = [int(line.strip()) for line in file.readlines()]

# Define a function to plot a segment of ECG data from all 4 channels with annotations
def plot_ecg_segment(start, end, annotations, ecg_data):
    time = range(start, end)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for i in range(4):
        axs[i].plot(time, ecg_data.iloc[start:end, i], label=f'Channel {i+1}')
        # Mark annotations (R-peaks) within this segment
        segment_peaks = [a for a in annotations if start <= a < end]
        axs[i].scatter(segment_peaks, ecg_data.iloc[segment_peaks, i], color='red', label='R-peaks')
        # axs[i].legend(loc='upper right')
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        # axs[i].set_ylabel('Amplitude')
    axs[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    plt.show()

# Plot a sample segment (can be iterated in practice)
segment_start = 1000
segment_end = 2000
plot_ecg_segment(segment_start, segment_end, annotations, ecg_data)
