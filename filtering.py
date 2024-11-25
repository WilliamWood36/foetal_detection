import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq





def lowPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0)
    z = signal.lfilter(b, a, data)
    return z

def highPassFilter(order, cutoff, data):
    print(cutoff)
    b, a = signal.butter(order, cutoff, fs=1000.0, btype="highpass")
    z = signal.lfilter(b, a, data)
    return z





data_titles = ['AECG1', 'AECG2','AECG3','AECG4']

file_path = 'a26.csv'

# Read the CSV file without skipping rows
sample_size = 5000
offset = 3000

read_data = pd.read_csv(file_path)[offset:sample_size + offset]
data = np.zeros(shape=(len(data_titles), sample_size))
non_filtered_data = data
for index,data_name in enumerate(data_titles):
    z1 = highPassFilter(3, 20, read_data[data_name])
    z = lowPassFilter(3, 30, z1)
    data[index] = z
    
    
    


#center the data
for index in range(0,len(data_titles)):
    data[index] = data[index] - np.mean(data[index])


#compute dot product of the array and its transposition
data_transp = np.transpose(data)
data_square = np.dot(data,data_transp)

eigen_vals, eigen_vects = LA.eig(data_square)

print(eigen_vals)


#eigen values in descending order
sorted_indexs = np.argsort(-eigen_vals)
eigen_vals_sorted = eigen_vals[sorted_indexs]
eigen_vects_sorted = eigen_vects[:, sorted_indexs]

p = eigen_vects_sorted

yd = np.dot(data.T,p).T

print(yd.shape)

# Access the Elapsed_time and AECG1 columns
try:
    figure, axis = plt.subplots(len(data_titles) +1, 1, figsize=(10, 8))
    
    # Original plot
    for index, name in enumerate(yd):
        x = read_data['Elapsed_time'][500:]
        y = name[500:]
        axis[index+1].set_title(eigen_vals_sorted[index])
        axis[index+1].plot(x, y, color='blue')


    x = read_data['Elapsed_time'][500:]
    y = yd[0][500:] - yd[1][500:]
    axis[0].plot(x,y,color = 'red')


    
    plt.subplots_adjust(top=0.95)
    plt.show()
except KeyError as e:
    print(f"KeyError: {e}. Please check the column names.")
