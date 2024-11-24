import matplotlib.pyplot as plt
import numpy as np
import wfdb
import wfdb.io


foetal_file_path = 'sub01_snr00dB_l1_c0_fecg1.dat'
foetal_qrs_path = 'sub01_snr00dB_l1_c0_fecg1.qrs'


import wfdb



# Provided data (first 200 samples)
data = wfdb.rdrecord('data/sub01_snr00dB_l1_c0_fecg1',sampto=300,channels=[1,2,3])

annotation = wfdb.io.rdann("data/sub01_snr00dB_l1_c0_fecg1","qrs",shift_samps=True).sample
print(annotation)

wfdb.plot_wfdb(record=data, annotation=annotation)