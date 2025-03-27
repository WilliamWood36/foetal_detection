import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq


data_titles = ['AECG1', 'AECG2','AECG3']
global adj 


def lowPassFilter(order, cutoff, data):
    b, a = signal.butter(order, cutoff, fs=1000.0)
    z = signal.lfilter(b, a, data)
    return z

def highPassFilter(order, cutoff, data):
    print(cutoff)
    b, a = signal.butter(order, cutoff, fs=1000.0, btype="highpass")
    z = signal.lfilter(b, a, data)
    return z


def update_plot(val):
    print(adj.get_hp_val(),adj.get_lp_val())
    update_plot_vals(adj.get_hp_val(),adj.get_lp_val())


def array_average(data):
    
    average = np.zeros(len(data[0]))
    
    for line in data:
        average += np.array(line)
    

    return average


def fourior_trans(data,freq_plot_spacing):
    ft_signal = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data))

    return



def update_plot_vals(hp_cutoff,lp_cutoff):
    filtered_data = []

    for i in range(0,len(data_titles)):
        axis_index = i * 2 + 1
        axis[axis_index].cla()  # Clear the second axis for the filtered data
        to_filter = data[data_titles[i]]  # Use the first data title for filtering

        # hp and lp data in that order
        z1 = highPassFilter(3, hp_cutoff, to_filter)
        z = lowPassFilter(3, lp_cutoff, z1)





        filtered_data.append(z)

        #formatting the plots
        axis[axis_index].plot(data['Elapsed_time'], z, label=f'Filtered {data_titles[i]}', color='red')
        axis[axis_index].set_title(f'Filtered {data_titles[i]} with HP cutoff: {int(hp_cutoff)} Hz, LP cutoff: {int(lp_cutoff)}')
        axis[axis_index].set_ylabel(data_titles[i])
        axis[axis_index].legend()

    average = array_average(filtered_data)

    axis[-1].cla()
    ft_signal = fft(average)

    N = len(average)

    freq = fftfreq(N,0.001)[:N//2]
    axis[-1].plot(freq, 2.0/N *np.abs(ft_signal[0:N//2]))
    axis[-1].set_xlim([0,50])

    axis[len(data_titles)*2].cla()
    axis[len(data_titles)*2].plot(data['Elapsed_time'], average, label=f'average of 3 data', color='green')
    plt.draw()  # Redraw the figure


#class containing setup and access for adjustments
class adjustments:
        # Highpass Slider

    def __init__(self, hp_initial_val,lp_initial_val):
        ax_slider = plt.axes([0.2, 0.01, 0.7, 0.03])  # [left, bottom, width, height]
        self.hp_slider = Slider(ax_slider, 'Highpass Cutoff (Hz)', 1, 100, valinit=hp_initial_val)


        # Lowpass Slider
        bx_slider = plt.axes([0.2, 0.06, 0.7, 0.03])  # [left, bottom, width, height]
        self.lp_slider = Slider(bx_slider, 'Lowpass Cutoff (Hz)', 1, 100, valinit=lp_initial_val)

        self.hp_slider.on_changed(update_plot)
        self.lp_slider.on_changed(update_plot)

    def get_lp_val(self):
        return self.lp_slider.val

    def get_hp_val(self):
        return self.hp_slider.va


# Specify the path to your CSV file
file_path = 'test_set/a26.csv'
fqrs_file_path = 'test_set/a26.csv'

# Read the CSV file without skipping rows
data = pd.read_csv(file_path)

label_data = pd.read_csv(fqrs_file_path)

temp = []
data.columns = data.columns.str.strip()  # Removes leading/trailing spaces
fqrs_data = [int(ys) for ys in label_data])


# Display the first few rows and column names
print("First few rows of the DataFrame:")
print(data.head())
print("\nColumn names:")
print(data.columns)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Access the Elapsed_time and AECG1 columns
try:
    figure, axis = plt.subplots(len(data_titles) * 2+2, 1, figsize=(10, 8))
    
    # Original plot
    for index, name in enumerate(data_titles):
        x = data['Elapsed_time']
        y = data[name]
        
        axis[index * 2].plot(x, y, label=name, color='blue')
        axis[index * 2].set_title(f'Original {name}')
        axis[index * 2].set_ylabel(name)
        axis[index * 2].legend()



    # Initial filtered plot
    hp_initial_cutoff = 30
    lp_initial_cutoff = 20

    adj = adjustments(20,30)

    
    update_plot_vals(hp_initial_cutoff, lp_initial_cutoff)
    

    # Highpass Slider
    
    plt.subplots_adjust(top=0.95)
    plt.show()
except KeyError as e:
    print(f"KeyError: {e}. Please check the column names.")
