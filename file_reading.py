# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:48:01 2024

@author: acd21ww
"""

from matplotlib import pyplot as plt
import numpy as np
import wfdb
from pathlib import Path

import os
import glob
from collections import defaultdict



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
            print("adding wave: ",w)
            #dataset.append((waveforms["wave1"][signal_indexes], waveforms["foetal_qrs"]))
            dataset.append(np.array((waveforms["wave1"][signal_indexes])))
            foetalDataset.append(waveforms["foetal_qrs"])
        
        print(np.array(dataset).shape,np.array(foetalDataset[0]).shape)
    print(f"Raw ECG data shape: {np.array(dataset).shape}")
    print(f"Foetal QRS shape: {np.array(foetalDataset[0]).shape}")
    return np.array(dataset), np.array(foetalDataset[0])
            


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
            "wave1": fecg_signal,
            "wave2": fecg_signal,
            "maternal_qrs": maternal_qrs,
            "foetal_qrs": foetal_qrs
        }
    return return_dict


def preprocess_ecg_data(ecg_data, qrs_positions, window_size=200, stride=50):
    """
    Splits ECG data into windows with each QRS complex centered.
    For regions with no QRS complex, uses regular stride-based windows.
    
    Args:
    - ecg_data: (4, 75000) -> 4-channel ECG signals
    - qrs_positions: List of true QRS positions in the full signal
    - window_size: Number of samples per window
    - stride: Step size for sliding window (used for regions without QRS)
    
    Returns:
    - X: List of tuples (ECG segment, QRS position)
    """
    # Reshape ecg_data if needed
    if len(ecg_data.shape) > 2:
        ecg_data = ecg_data.reshape(ecg_data.shape[0], -1)
    
    signal_length = ecg_data.shape[1]
    half_window = window_size // 2
    X = []
    
    # First, create windows centered on each QRS complex
    for qrs_pos in qrs_positions:
        # Skip if QRS is too close to the edges
        if qrs_pos < half_window or qrs_pos >= signal_length - half_window:
            continue
        
        # Create window centered on QRS complex
        start = qrs_pos - half_window
        end = start + window_size
        segment = ecg_data[:, start:end]
        
        # QRS position relative to window start (should be at half_window)
        relative_pos = half_window
        
        X.append((segment, relative_pos))
    
    # Then fill in gaps with regular stride-based windows
    # Track regions we've already covered with QRS-centered windows
    covered_regions = []
    for qrs_pos in qrs_positions:
        if qrs_pos < half_window or qrs_pos >= signal_length - half_window:
            continue
        start = max(0, qrs_pos - half_window)
        end = min(signal_length, qrs_pos + half_window)
        covered_regions.append((start, end))
    
    # Sort covered regions
    covered_regions.sort()
    
    # Merge overlapping regions
    if covered_regions:
        merged_regions = [covered_regions[0]]
        for current in covered_regions[1:]:
            prev_start, prev_end = merged_regions[-1]
            current_start, current_end = current
            
            if current_start <= prev_end:
                # Regions overlap, merge them
                merged_regions[-1] = (prev_start, max(prev_end, current_end))
            else:
                # No overlap, add as new region
                merged_regions.append(current)
        
        covered_regions = merged_regions
    
    # Now create windows for uncovered regions
    current_pos = 0
    for start_covered, end_covered in covered_regions:
        # Process region before the current covered region
        while current_pos + window_size <= start_covered:
            segment = ecg_data[:, current_pos:current_pos + window_size]
            # No QRS in this window
            X.append((segment, -1))
            current_pos += stride
        
        # Skip the covered region
        current_pos = end_covered
    
    # Process any remaining uncovered region at the end
    while current_pos + window_size <= signal_length:
        segment = ecg_data[:, current_pos:current_pos + window_size]
        # Check if there's any QRS in this window (should be none based on our logic)
        qrs_in_window = [q - current_pos for q in qrs_positions if current_pos <= q < current_pos + window_size]
        label = -1  # Should be no QRS here
        X.append((segment, label))
        current_pos += stride
    
    return X

        # x = np.arange(100)

        # # Plot the signal
        # plt.plot(x, segment[0], label="Signal")
        # print(segment[0]," ", label)
        # # Plot a dot at the given index
        # plt.scatter(label, 50, color="red", zorder=5, label=f"Dot at index ")

        # plt.xlabel("Index")
        # plt.ylabel("Signal Value")
        # plt.title("Signal with Dot at a Specific Index")
        # plt.legend()
        # plt.show()



