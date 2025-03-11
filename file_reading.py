# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:48:01 2024

@author: acd21ww
"""

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

    dataset = []
    total_maternal_waves = 0
    for record_name, files in grouped_files.items():
        print(f"Processing Record: {record_name}")


        waveforms = build_waveforms(files)
        
        for w in waves:
            print("adding wave: ",w)
            dataset.append((waveforms["wave1"][signal_indexes], waveforms["foetal_qrs"]))
    return dataset
            


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
    foetal_qrs = []
    maternal_qrs = []
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
