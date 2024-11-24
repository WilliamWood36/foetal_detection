# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:21:15 2024

@author: bigey
"""


from file_reading import group_files_by_record, build_waveforms
from plotting_test import basic_plot,plot_n_components
from bss import function_pca

def process_files_in_layout(base_directory):
    """
    Process all grouped files based on the folder layout.

    Args:
        base_directory (str): The base directory containing the folder layout.
    """
    grouped_files = group_files_by_record(base_directory)
    if not grouped_files:
        print("No valid files found.")
        return

    for record_name, files in grouped_files.items():
        print(f"Processing Record: {record_name}")


        waveforms = build_waveforms(files)
        
        pca,eigen = function_pca(waveforms["wave1"], 32, 5)
        
        plot_n_components(pca, 5)
        basic_plot(waveforms["wave1"][5][:500],waveforms["foetal_qrs"][:3],waveforms["maternal_qrs"][:3])
        break
        for file in files:
            print(f"    File: {file}")
        # Placeholder for further processing logic
        # x(files)

# Example usage
base_directory_path = "./data"  # Replace with the path to your base folder
process_files_in_layout(base_directory_path)








