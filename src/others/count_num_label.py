# -*- coding: utf-8 -*-
#-------------------------------------#
# 
#
#-------------------------------------#
import numpy as np
import pathlib
from multiprocessing import Pool
import glob

def count_num_label(test_list_label):
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_all = 0
    for data in test_list_label:
        label = np.loadtxt(data, dtype=int)
        num_all += len(label)
        for t in range(len(label)):
            if label[t] == 0:
                num_0 += 1
            elif label[t] == 1:
                num_1 += 1
            elif label[t] == 2:
                num_2 += 1
            elif label[t] == 3:
                num_3 += 1
            elif label[t] == 4:
                num_4 += 1

    print('type0 : %5s' % (num_0))
    print('type1 : %5s' % (num_1))
    print('type2 : %5s' % (num_2))
    print('type3 : %5s' % (num_3))
    print('type4 : %5s' % (num_4))
    print('all : %5s' % (num_all))

if __name__ == "__main__":
    # Get file list
    list_label = glob.glob("*.txt")
    sort_list_label = sorted(list_label)
    count_num_label(sort_list_label)