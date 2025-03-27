# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:48:01 2024

@author: acd21ww
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path


import re
import os
import glob
from collections import defaultdict
#used for temocing 
def clean_data():
    directory = "test_set"  

    # List all files in the directory
    files = os.listdir(directory)
    print("egg \n",files)
    # Use regex to extract the numeric part of filenames
    a_files = {re.match(r"a(\d+)\.csv", f).group(1): f for f in files if re.match(r"a\d+\.csv", f)}

    for i in a_files:
        file_name = directory +"/a"+ i + ".csv"
        print(file_name)
        with open(file_name, "r") as infile:
            lines = infile.readlines()  # Read all lines

        # Modify only the first two lines
        for i in range(min(2, len(lines))): 
            lines[i] = lines[i].replace("'", "\"")
        lines[1]  = "\n"
                
            
        with open(file_name, "w") as outfile:
            outfile.writelines(lines)

def load_challenge_data(channels=4):
    dataset = []
    data_titles = ['AECG1', 'AECG2','AECG3','AECG4']
    


    directory = "test_set"  

    # List all files in the directory
    files = os.listdir(directory)

    # Use regex to extract the numeric part of filenames
    a_files = {re.match(r"a(\d+)\.csv", f).group(1): f for f in files if re.match(r"a\d+\.csv", f)}
    s_files = {re.match(r"a(\d+)\.fqrs\.txt$", f).group(1): f for f in files if re.match(r"a\d+\.fqrs\.txt$", f)}


    # Find matching pairs
    paired_files = [(a_files[key], s_files[key]) for key in a_files if key in s_files]

    # Read the CSV file without skipping rows
    dataset = []
    lables = []

    for X,y in paired_files:
        data = pd.read_csv(directory+"/"+X)
        label_data = pd.read_csv(directory+"/"+y)

        temp = []
        data.columns = data.columns.str.strip()  # Removes leading/trailing spaces
        lables.append([int(ys) for ys in label_data])

        for i in range(0,channels):
            temp.append(data[data_titles[i]])
            
        dataset.append(temp)

    return np.array(dataset), lables


def reshape_date(data,num_channels):
    if data.size % num_channels == 0:
        reshaped_data = data.reshape(-1, num_channels)
    else:
        raise ValueError("The data size is not compatible with the specified number of channels.")

    # Separate each channel into individual arrays
    channels = np.array([np.array(reshaped_data[:, i]) for i in range(num_channels)])
    return channels

def dataset_builder(base_directory, signal_indexes, seperate=False):
    """
    Process all grouped files based on the folder layout.

    Args:
        base_directory (str): The base directory containing the folder layout.
    """
    grouped_files = group_files_by_record(base_directory)
    if not grouped_files:
        print("No valid files found.")
        return

    waves = ["wave1","wave2"]
    foetalDataset = []
    dataset = []
    total_maternal_waves = 0
    for record_name, files in grouped_files.items():
        print(f"Processing Record: {record_name}")


        waveforms = build_waveforms(files)
        
        for w in waves:
            #print("adding wave: ",w)
            dataset.append(np.array((waveforms["wave1"][signal_indexes])))
            foetalDataset.append(waveforms["foetal_qrs"])
        

    return np.array(dataset), foetalDataset
            


def group_files_by_record(base_directory, data_extension=".dat", annotation_extension=".qrs", header_extension=".hea"):
    """
    Traverse the folder layout and group files by their base record name.

    Args:
        base_directory (str): The base directory containing subject folders.
        data_extension (str): File extension for data files (default: '.dat').
        annotation_extension (str): File extension for annotation files (default: '.qrs').

    Returns:
        dict: A dictionary where keys are base record names, and values are lists of tuples
              containing paths to related files (data and annotation files).
    """
    grouped_files = defaultdict(list)

    # Traverse each subject folder
    for subject_folder in sorted(os.listdir(base_directory)):
        subject_path = os.path.join(base_directory, subject_folder)
        if not os.path.isdir(subject_path):
            continue  # Skip non-folder items
        
        # Traverse each signal folder inside the subject folder
        for signal_folder in sorted(os.listdir(subject_path)):
            signal_path = os.path.join(subject_path, signal_folder)
            if not os.path.isdir(signal_path):
                continue  # Skip non-folder items
            
            # Find data and annotation files in this signal folder
            files = glob.glob(os.path.join(signal_path, f"*{data_extension}")) + \
                    glob.glob(os.path.join(signal_path, f"*{annotation_extension}")) + \
                    glob.glob(os.path.join(signal_path, f"*{header_extension}"))

            for file in files:
                base_name = "_".join(os.path.basename(file).split("_")[:4])  # Extract the common record prefix
                #addgenerate 2 waveforms for each record, one with each source of noise
                grouped_files[base_name].append(file)

    return grouped_files





def build_waveforms(files):
    
    """ 
    
    takes paths for each recording and generates 2 signals, one for each noise file. 
    
    Args:
        files [str]: file paths to be used to build the waves
        
    Returns:
        ((Waveform1,Waveform2),(maternal qrs, foetal qrs))
    """
    foetal_qrs, maternal_qrs = [], []
    fecg_signal = []
    waveform1 =np.zeros((32,75000))
    waveform2 = waveform1
    for file in files:

        if Path(file).suffix == ".dat":
   
            data_to_add = reshape_date(np.fromfile(file, dtype=np.int16),34)[:-2]
            if "noise1"  in file:
                waveform1 += data_to_add
            elif "noise2" in file:
                waveform2 += data_to_add
            else:
                waveform1 += data_to_add
                waveform2 += data_to_add
                if "fecg1" in file:
                    fecg_signal = data_to_add
                
        elif Path(file).suffix == ".qrs":

            qrs_data = wfdb.io.rdann(os.path.splitext(file)[0],"qrs",shift_samps=True).sample
  
            if "fecg" in file:
                foetal_qrs = qrs_data
            else:
                maternal_qrs = qrs_data
    return_dict = {
            "wave1": waveform1,
            "wave2": waveform2,
            "maternal_qrs": maternal_qrs,
            "foetal_qrs": foetal_qrs
        }
    return return_dict


import numpy as np

def preprocess_ecg_data(ecg_data, qrs_positions, window_size=500, stride=50):
    """
    Splits ECG data into overlapping windows and assigns QRS positions.
    
    Args:
    - ecg_data: (14, 4, 75000) -> 14 trials of 4-channel ECG signals (time, channels, samples)
    - qrs_p 3ositions: List of QRS positions for each trial (List of lists)
    - window_size: Number of samples per window
    - stride: Step size for sliding window
    
    Returns:
    - X: (num_windows, 4, window_size) -> List of segmented ECG windows
    - y: (num_windows,) -> QRS positions relative to each window (-1 if no QRS found)
    """
    # this only works for analogus data, 
    num_trials, num_channels, num_samples = ecg_data.shape
    X, y = [], []

    for trial in range(num_trials):
        trial_data = ecg_data[trial]  # Shape: (4, 75000)
        trial_qrs_positions = qrs_positions[trial]
        num_windows = (num_samples - window_size) // stride + 1  # Compute number of windows
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            segment = trial_data[:, start:end]  # Shape: (4, window_size)
            # Use global qrs_positions array for all channels
            qrs_in_window = [q - start for q in trial_qrs_positions if start <= int(q) < end]

            # Assign the first QRS position found, or -1 if none exist
            label = qrs_in_window[0] if qrs_in_window else -1
            X.append(segment)  
            y.append(label)  
    

            if i % 10000 == 0:
                x = np.arange(len(segment[0]))
                # Plot the signal
                plt.plot(x, segment[0], label="Signal")
                # Plot a dot at the given index
                plt.scatter(label, 50, color="red", zorder=5, label=f"Dot at index ")

                plt.xlabel("Index")
                plt.ylabel("Signal Value")
                plt.title("Signal with Dot at a Specific Index")
                plt.legend()
                #plt.show()
    
    return np.array(X), np.array(y)  # Return as NumPy arrays for ML compatibility

 



    
