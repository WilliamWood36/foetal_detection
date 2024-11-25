import sys
import numpy as np
import inspect

this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('detectors')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'Diss')

from bss import function_pca


def highest_point(waveforms, window_len=150):


    pca,eigen = function_pca(waveforms, 32, 5)

    data = pca[:,0]

    maximums = []

    windows = int((data.size - data.size % window_len)/(window_len/2))-1
    for index in range(0,windows):

        max_index = 0
        average = 0
        
        for inner in range(0,window_len):
            access_index = int((index*(window_len/2)) + inner)

            if data[access_index] > data[max_index]:  
                max_index = access_index
  
    #    if len(maximums) > 3 and abs(average - data[access_index]) < (average * 0.4):
    #        average = average * (len(average)-1)/len(average) + data[access_index]/len(average)
            
            
        maximums.append(max_index)
        
    maximums = list(dict.fromkeys(maximums))
    return maximums




