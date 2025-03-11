from file_reading import group_files_by_record, build_waveforms, reshape_date
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Path to the binary files
foetal_file_path = '../data/sub01/snr00dB/sub01_snr00dB_l1_c0_fecg1.dat'
foetal_qrs_path = '../data/sub01/snr00dB/sub01_snr00dB_l1_c0_fecg1.qrs'


# Define the number of channels (32 ECG + 2 reference)
num_channels = 34

# Read the binary file as 16-bit integers
foetal_data = reshape_date(np.fromfile(foetal_file_path, dtype=np.int16), num_channels)[:-2]

#foetal_qrs = reshape_date(np.fromfile(foetal_qrs_path, dtype=np.int16),num_channels)



signal_data = foetal_data[4] 
fs = 250


print(f"Sampling Frequency (fs): {fs} Hz")

# Signal length
signal_length = len(signal_data) / fs
print(f"Length of the signal: {signal_length:.4f} seconds")


Ts = 1 / fs
print(f"Sampling Interval (s): {Ts:.6f} seconds")

# Highest occuring frequency
n = len(signal_data)
frequencies = np.fft.fftfreq(n, d=Ts)
fft_values = np.fft.fft(signal_data)
fft_magnitude = np.abs(fft_values)

positive_frequencies = frequencies[:n//2]
positive_magnitude = fft_magnitude[:n//2]

highest_frequency = positive_frequencies[np.argmax(positive_magnitude)]
print(f"Highest Occurring Frequency: {highest_frequency} Hz")

# Plot frequencys
plt.plot(positive_frequencies, positive_magnitude)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()


f_orig, t_orig, Sxx_orig = signal.spectrogram(signal_data[:1000], 250)



plt.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig), cmap='viridis')
plt.title("spectogram of pure foetal signal")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.show()

