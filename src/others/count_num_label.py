# -*- coding: utf-8 -*-
import numpy as np
import pathlib
from multiprocessing import Pool
import glob

def count_num_label(sort_list_label, train_list_label, valid_list_label, test_list_label):
    print(test_list_label)
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    num_8 = 0
    num_9 = 0
    num_10 = 0
    num_11 = 0
    num_all = 0
    for data in test_list_label:
        label = np.loadtxt(data, dtype=int)
        print(len(label))
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
            elif label[t] == 5:
                num_5 += 1
            elif label[t] == 6:
                num_6 += 1
            elif label[t] == 7:
                num_7 += 1
            elif label[t] == 8:
                num_8 += 1
            elif label[t] == 9:
                num_9 += 1
            elif label[t] == 10:
                num_10 += 1
            elif label[t] == 11:
                num_11 += 1

    print('type0 : %5s' % (num_0))
    print('type1 : %5s' % (num_1))
    print('type2 : %5s' % (num_2))
    print('type3 : %5s' % (num_3))
    print('type4 : %5s' % (num_4))
    print('type5 : %5s' % (num_5))
    print('type6 : %5s' % (num_6))
    print('type7 : %5s' % (num_7))
    print('type8 : %5s' % (num_8))
    print('type9 : %5s' % (num_9))
    print('type10 : %5s' % (num_10))
    print('type11 : %5s' % (num_11))
    print('all : %5s' % (num_all))

if __name__ == "__main__":
    # Get file list
    list_label = glob.glob("*.txt")
    sort_list_label = sorted(list_label)
    train_list_label = []
    valid_list_label = []
    test_list_label = []
    for i in range(0, 10):
        train_list_label.append(sort_list_label[i])
    for i in range(10, 11):
        valid_list_label.append(sort_list_label[i])
    for i in range(11, 14):
        test_list_label.append(sort_list_label[i])
    count_num_label(sort_list_label, train_list_label, valid_list_label, test_list_label)