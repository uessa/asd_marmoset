# -*- coding: utf-8 -*-
import numpy as np
import pathlib
from multiprocessing import Pool
import glob

def find_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                if not (ground_truth_data[t, 2] in type_call):
                    type_call.append(ground_truth_data[t, 2])
    print(type_call)
    print(len(type_call))

if __name__ == "__main__":
    # path
    path_text = pathlib.Path()

    # Get file list
    list_text = path_text.glob("./*.txt")
    find_calltype(list_text)