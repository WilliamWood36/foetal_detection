# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:21:15 2024

@author: bigey
"""

import matplotlib.pyplot as plt
import numpy as np
import wfdb
import wfdb.io


foetal_file_path = 'sub01_snr00dB_l1_c0_fecg1.dat'
foetal_qrs_path = 'sub01_snr00dB_l1_c0_fecg1.qrs'


import wfdb


import os
import glob

def get_all_files_in_layout(base_directory, data_extension=".dat", annotation_extension=".qrs"):
    """
    Traverse the folder layout and retrieve all data and annotation file pairs.

    Args:
        base_directory (str): The base directory containing subject folders.
        data_extension (str): File extension for data files (default: '.dat').
        annotation_extension (str): File extension for annotation files (default: '.qrs').

    Returns:
        list: A list of tuples containing paths to data and corresponding annotation files.
    """
    paired_files = []

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
            
            # Find and pair data and annotation files in this signal folder
            data_files = sorted(glob.glob(os.path.join(signal_path, f"*{data_extension}")))
            for data_file in data_files:
                base_name = os.path.splitext(os.path.basename(data_file))[0]
                annotation_file = os.path.join(signal_path, f"{base_name}{annotation_extension}")
                
                if os.path.exists(annotation_file):
                    paired_files.append((data_file, annotation_file))
                else:
                    print(f"Warning: No annotation file found for {data_file}")

    return paired_files

def process_files_in_layout(base_directory):
    """
    Process all data and annotation files based on the folder layout.

    Args:
        base_directory (str): The base directory containing the folder layout.
    """
    files = get_all_files_in_layout(base_directory)
    
    if not files:
        print("No valid data and annotation file pairs found.")
        return

    for data_file, annotation_file in files:
        print(f"Processing: Data File: {data_file}, Annotation File: {annotation_file}")
        x(data_file, annotation_file)  # Placeholder function for processing

# Example usage
base_directory_path = "./path_to_subjects"  # Replace with the path to your base folder
process_files_in_layout(base_directory_path)







foetal_annotations = wfdb.io.rdann("data/sub01_snr00dB_l1_c0_fecg1","qrs",shift_samps=True).sample


