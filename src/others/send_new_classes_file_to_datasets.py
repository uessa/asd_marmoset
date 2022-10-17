# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import re
import pprint
import itertools
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import shutil

if __name__ == "__main__":

    labelpath = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/label/" # GroundTruth frame .txt Path
    # resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_check_othercalls/test/results/" # Estimate frame .txt Path
    outputpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test_wo_pheetrill/"

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    # results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]

    # fileのfindとcp処理
    names = vpas # 対象の個体名
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


