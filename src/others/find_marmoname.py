# -*- coding: utf-8 -*-
#-------------------------------------#
# 
#
#-------------------------------------#
import os
import re
import sys
import glob
import math
import pprint
import pathlib
import textgrid
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages



if __name__ == "__main__":

    path = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/label"

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    files = sorted(files, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    marmo = []
    for i, file in enumerate(files):
        date = float(re.findall(r'VOC_.*_.*_(.*)W', file)[0]) # "週"
        name = re.findall(r'VOC_.*_(.*)_.*', file)[0] # "名前"

        if not (name in marmo):
            marmo.append(name)

        print(file)
        print(name)
        
    print(marmo)