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
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import re
import os
import glob
from collections import defaultdict

from filtering import highPassFilter, lowPassFilter, remove_baseline_wander
#used for temocing 

donwsample_factor = 4
frequency = 250



def downsample_by_averaging(signal, factor=donwsample_factor,fs=frequency):
    """
    Downsamples the signal by averaging every `factor` samples.
    
    Parameters:
    - signal: 1D NumPy array
    - factor: Downsampling factor (e.g., 4 for 1000 Hz -> 250 Hz)
    
    Returns:
    - downsampled signal as a 1D NumPy array
    """

    trimmed_length = len(signal) - (len(signal) % factor)  # make it divisible by factor
    trimmed_signal = signal[:trimmed_length]
    downsampled = trimmed_signal.reshape(-1, factor).mean(axis=1)
    return downsampled

def connvert_to_difference_data():
    directory = "test_set_diff"
    files = os.listdir(directory)
    print("egg \n", files)

    # Use regex to extract the numeric part of filenames
    a_files = {
        re.match(r"a(\d+)\.csv", f).group(1): f
        for f in files if re.match(r"a\d+\.csv", f)
    }

    for i in a_files:
        file_name = os.path.join(directory, "a" + i + ".csv")
        print(f"Processing: {file_name}")

        # Read CSV into a DataFrame
        df = pd.read_csv(file_name)

        # Convert all columns to numeric (if not already), then calculate difference
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
        df_diff = df.diff().fillna(0)  # Replace NaN (from first row) with 0

        # Save back to the same file (or modify if needed)
        df_diff.to_csv(file_name, index=False)


def clean_data():
    directory = "test_set_filtered"  

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

def preprocess_multi(data, scaler,Bscutoff = 1,Ucutoff=75, notch = 50, fs = 250, downsample = True ):
    filtered = [] 
    for w in data:
        filtered.append(preprocess_single(w,scaler,Bscutoff,Ucutoff,notch,fs,downsample))
    return filtered

def preprocess_single(data, scaler,Bscutoff = 10,Ucutoff=75, notch = 50, fs = 250, downsample = True ):
    filt = remove_baseline_wander(data,fs=frequency,bcutoff=Bscutoff,ucutoff=Ucutoff)
    if downsample:
        filt = downsample_by_averaging(filt)
    down = scaler.fit_transform(filt.reshape(-1,1))
    return np.squeeze(np.array(down))
    




def load_challenge_data(directory = "test_set_filtered",channels=4, lp=0.7,hp=75):
    # returns 
    dataset = []
    data_titles = ['AECG1', 'AECG2','AECG3','AECG4']
    
    # List all files in the directory
    files = os.listdir(directory)

    # Use regex to extract the numeric part of filenames
    a_files = {re.match(r"a(\d+)\.csv", f).group(1): f for f in files if re.match(r"a\d+\.csv", f)}
    s_files = {re.match(r"a(\d+)\.fqrs\.txt$", f).group(1): f for f in files if re.match(r"a\d+\.fqrs\.txt$", f)}

    # Find matching pair
    paired_files = [(a_files[key], s_files[key]) for key in a_files if key in s_files]

    # Read the CSV file without skipping rows
    dataset = []
    lables = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for X,y in paired_files:
        data = pd.read_csv(directory+"/"+X)
        label_data = pd.read_csv(directory+"/"+y)   
        temp = []
        lables.append([int(ys)/donwsample_factor for ys in label_data.iloc[:, 0] ])

        for i in range(0,channels):
            temp.append(preprocess_single(np.array(data[data_titles[i]],dtype="float").reshape(-1,1),scaler,fs=250,Bscutoff=lp,Ucutoff=hp))
            
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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    for record_name, files in grouped_files.items():
        print(f"Processing Record: {record_name}")


        waveforms = build_waveforms(files)
        
        for w in waves:
            #print("adding wave: ",w)
            #print("w shape",np.array(waveforms["wave1"][signal_indexes]).shape, ",",np.array((waveforms["wave1"][signal_indexes])).shape)
            dataset.append(preprocess_multi(np.array((waveforms["wave1"][signal_indexes])),scaler, downsample=False))
            foetalDataset.append(waveforms["foetal_qrs"])
        
    #print(np.array(dataset).shape)
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

def preprocess_ecg_data(ecg_data, qrs_positions, window_size=500, stride=50,filter=True):
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
            # for i in range(0,len(segment)):
            #     z = segment[i]
            #     z1 = highPassFilter(3, 20, z)
            #     segment[i] =  lowPassFilter(3, 50, z1)
                
            # Use global qrs_positions array for all channels
            qrs_in_window = [q for q in trial_qrs_positions if start <= int(q) < end]
            label = qrs_in_window[0] - start if qrs_in_window else -1
            # Assign the first QRS position found, or -1 if none exist
            X.append(segment)  
            y.append(label)  

    return np.array(X), np.array(y)  # Return as NumPy arrays for ML compatibility

 


    
