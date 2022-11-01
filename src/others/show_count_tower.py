# -*- coding: utf-8 -*-
'''
・正解・推定ラベル [000011111000000222222...]
・グルーピング     [0102...]
・カウント辞書化   {'Phee':100, 'Trill':150, ... }
・比率化           {'Phee':0.2, 'Trill':0.3, ... }
    →比率化で評価

・UEの正解・推定
・VPAの正解・推定
       | ------ | ------ | ------- | ------- |
    ⇒ | UE正解 | UE推定 | VPA正解 | VPA推定 |
       | ------ | ------ | ------- | ------- |
    のように並べたらよさそう？（理想）

    →指定ディレクトリのファイルだけで比率を見る？
    例）ue の test, test/results だけで個体ごとの比率（正解, 推定）
    例）vpa の test, test/results だけで個体ごとの比率（正解, 推定）
    例）23ue の text/label だけで個体ごとの比率（正解）
    例）11vpa の text/label だけで個体ごとの比率（正解）

    →カウント辞書を個体ごと全週ぶんでやり，最後に比率化

'''

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
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

    # pathの指定
    labelpath1 = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test/" 
    resultpath1 = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test_wo_pheetrill/results/"

    # list の生成
    files1 = [f for f in os.listdir(labelpath1) if os.path.isfile(os.path.join(labelpath1, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results1 = [f for f in os.listdir(resultpath1) if os.path.isfile(os.path.join(resultpath1, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    # pprint.pprint(files1)

    # files1, results1 の sort
    files1 = sorted(files1, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results1 = sorted(results1, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    files1 = sorted(files1, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results1 = sorted(results1, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    # pprint.pprint(files1)

    # 保存用
    list_label1 = [] # 
    list_results1 = [] # 

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]

    # 対象個体のリスト
    marmos = tests

    # 個体ごとに平均比率
    for marmo in marmos:
        for file in files1:
            # ファイル名から週と名前を抽出
            date = re.findall('VOC_.*_.*_(.*)W',file)[0] # "週"
            name = re.findall('VOC_.*_(.*)_.*',file)[0] # "名前"
            # 指定個体でないならcontinue
            if name != marmo:
                continue
            # nparray型のフレームラベル
            label = np.loadtxt(labelpath1 + file, dtype=int)
            results = np.loadtxt(resultpath1 + file, dtype=int)
            # フレームのラベルをgroupingで連続同値を順列にまとめて単純化
            grouped_label = [k for k,g in itertools.groupby(label)] 
            grouped_results = [k for k,g in itertools.groupby(results)] 
            # grouped_label = label
            # grouped_results = results






        