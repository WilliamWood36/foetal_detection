# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:21:15 2024

@author: bigey
"""

import numpy as np

from file_reading import group_files_by_record, build_waveforms
from plotting_test import basic_plot,plot_n_components
from bss import function_pca
from detectors.pca_window import highest_point
foetal_annotations = []
maternal_annotations = []



def average_distance_with_filter(reference, estimated, threshold=5.0):
    """
    Calculate the average distance between points in the reference and estimated arrays,
    allowing the estimated array to have more values than the reference array.
    Each reference value is matched to the closest estimated value within the threshold range.
    Tracks unpaired estimated values and unmatched reference values.

    Parameters:
    - reference (list or array): The correct reference values.
    - estimated (list or array): The estimated values.
    - threshold (float): The maximum allowable distance for an estimated value to be considered valid.

    Returns:
    - tuple: (average_distance, unmatched_references, unpaired_estimated)
        - average_distance (float): Average distance between valid points, or None if no matches.
        - unmatched_references (int): Number of reference values without a match.
        - unpaired_estimated (int): Number of estimated values that were never paired.
    """
    used_indices = set()
    valid_distances = []
    unmatched_references = 0

    for ref_val in reference:
        # Find all estimated values within the threshold of the current reference value
        candidates = [
            (abs(ref_val - est_val), idx)
            for idx, est_val in enumerate(estimated)
            if idx not in used_indices and abs(ref_val - est_val) <= threshold
        ]

        if candidates:
            # Select the closest estimated value
            closest_distance, closest_index = min(candidates, key=lambda x: x[0])
            valid_distances.append(closest_distance)
            used_indices.add(closest_index)  # Mark this estimated value as used
        else:
            # No match found for this reference value
            unmatched_references += 1

    # Calculate the number of unpaired estimated values
    unpaired_estimated = len(estimated) - len(used_indices)

    # Compute average distance
    average_distance = np.mean(valid_distances) if valid_distances else None

    return average_distance, unmatched_references, unpaired_estimated



def check_pairs(estimated, actual):
    mle = 0
    index = 0
    for (a,b) in zip(estimated,actual):
        index +=1
        val = abs(a-b)
        if index ==1:
            mle = val
        else:
            mle = mle * (index-1)/index + val/index
        if val > 2:
            print(a," actual: ",b)
    return mle
     
    

def process_files_in_layout(base_directory,detector_function):
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

    net_mle = []
    net_missing = []
    net_extra = []
    records = 0
    total_maternal_waves = 0
    for record_name, files in grouped_files.items():
        print(f"Processing Record: {record_name}")


        waveforms = build_waveforms(files)
        
        for w in waves:
            print(w)
            total_maternal_waves += waveforms["maternal_qrs"].size
            
            eval = np.array(detector_function(waveforms[w]))
            

            
            mle, missing, extra = average_distance_with_filter(eval,waveforms["maternal_qrs"],6.0)
            
            net_mle.append(mle)
            net_missing.append(missing)
            net_extra.append(extra)

        #basic_plot(waveforms["wave1"][5][:2000],eval,waveforms["maternal_qrs"][:15])


    print("*" * 35)
    print(f"MLE of : {np.mean(net_mle)}")
    print("Missed Waves: ")
    print(f"Missed Waves: {np.sum(net_missing)} \nExtra Waves: {np.sum(net_extra)}")
    print(f"Over: {total_maternal_waves} total beats")
    print("*" * 35)






# Example usage
base_directory_path = "./data"  # Replace with the path to your base folder
process_files_in_layout(base_directory_path, highest_point)








