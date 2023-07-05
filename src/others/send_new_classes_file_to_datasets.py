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
import shutil
import pprint
import pathlib
import textgrid
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_calc_uevpa_type/ue/" # for tests or vpasも忘れずに
    outputpath = "/home/muesaka/projects/marmoset/datasets/subset_calc_uevpa_type/ue_new/"

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾マッチ

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]

    # fileのfindとcp処理
    
    names = trains # 対象の個体ジャンル

    for i, file in enumerate(files):

        name = (re.findall('VOC_.*_(.*)_.*',file))[0] # "名前"
        file2 = (re.findall('(VOC_.*_.*_.*)',file))[0] # cp先ファイル名
        
        # 一致しなかったらcontinue
        if name not in names:
            continue
        
        # cp処理
        file1 = labelpath + file
        file2 = outputpath + file2

        print(file1)
        print(file2)
        shutil.copyfile(file1, file2)

        # break   


