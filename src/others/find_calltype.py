# -*- coding: utf-8 -*-
#-------------------------------------#
# 
#
#-------------------------------------#
import numpy as np
import pathlib
# from multiprocessing import Pool
import glob
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchaudio
import re

def find_calltype(list_text):
    type_call = ['Noise']
    for data in list_text:
        ground_truth_data = np.loadtxt(data, dtype=str, delimiter="\t")
        for t in range(ground_truth_data.shape[0]):
            if ground_truth_data[t, 1] == 'Call' or ground_truth_data[t, 1] == 'Calls':
                if not (ground_truth_data[t, 2] in type_call):
                    type_call.append(ground_truth_data[t, 2])
                    print("file:", data)
                    print("min: {}\t max: {}".format(ground_truth_data[t, 0], ground_truth_data[t, 3]))
                    print("label:", ground_truth_data[t, 2])
                    print("")
    print("type_call:", type_call)
    print("len(type_call):", len(type_call))


def output_emptylabel(list_npy):
    for data in list_npy:
        spec = np.load(data)
        data = str(data)
        pattern = '(.*_alone_.*).npy'
        filename = re.findall(pattern ,data)[0]
        x = np.zeros(spec.shape[1], dtype=int)

        np.savetxt(filename + ".txt", x, fmt='%1.f')
        print(filename)
        print(spec.shape[1])
        print(x[0])

        # break

if __name__ == "__main__":
    # path
    # path_text = pathlib.Path("/home/muesaka/projects/marmoset/raw/marmoset_23ue_text/nogtxt_copy")
    path_npy = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_vpa_48kHz/test")

    # Get file list
    # list_text = path_text.glob("*.txt")
    list_npy = path_npy.glob("*.npy")

    # find_calltype(list_text)
    output_emptylabel(list_npy)