# -*- coding: utf-8 -*-

# import os
# import re
# import sys
# import glob
# import math
# import torch
# import pprint
# import pathlib
# import textgrid
# import functools
# import itertools
import numpy as np
import pandas as pd
# import torch.nn as nn
# import seaborn as sns
# import japanize_matplotlib
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from matplotlib.backends.backend_pdf import PdfPages

def masked_cross_entropy(outputs, labels):
    f1 = labels == 5 # [5 ... 6 ... 7 ...] == 5 -> [True ... False ...]
    f2 = labels == 6 # [5 ... 6 ... 7 ...] == 6 -> f1
    f3 = labels == 7 # [5 ... 6 ... 7 ...] == 7 -> f1
    flag1 = f1 | f2 | f3
    print("flag1:",flag1)

    f = np.array([5, 6, 7])
    flag2 = labels == f
    print("flag2:",flag2)

    labels1 = labels
    labels2 = labels
    labels1[flag1] = -1
    labels2[flag2] = -1          
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # loss = criterion(outputs, labels)
    print("labels1:",labels1)
    print("labels2:",labels2)
    # return labels

a = np.array([5,0,1,2,3,4,5, 6,1,2,3,4, 7,1,2,3,4, ])
b = a
masked_cross_entropy(a, b)