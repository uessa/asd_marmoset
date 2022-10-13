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

np.seterr(divide='raise')

# 混合行列作成
def make_confusion_matrix(results, labels, classes, output_dir, name, date):
    cm_label_estimate = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Tsik', 'Ek', 'Ek-Tsik', 'Cough', 'Cry', 'Chatter', 'Breath', 'Unknown']
    cm_label = ['No Call', 'Phee', 'Trill', 'Twitter','Other Calls']
    
    cm = confusion_matrix(labels, results, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]) # labelとresult反転 14クラスの混合行列
    cm = np.delete(cm, slice(len(cm_label), len(cm_label_estimate)), 0) # 0-4行目を除く行を削除
    print(len(cm_label))
    print(len(cm_label_estimate))
    print(cm)
    print(type(cm))

    

    # 行毎に確率値を出して色分け
    cm_prob = cm / np.sum(cm, axis=1, keepdims=True)

    # cm = cm[:, :5]
    # cm = cm.T
    # cm_prob = cm_prob[:, :5]
    # cm_prob = cm_prob.T

    # 2クラス分類：font=25,annot_kws35, 12クラス分類：font=15,annot_kws10, 5クラス分類：font=15,annot_kws20, cbar=False
    fig = plt.figure(figsize=(20, 8))
    plt.rcParams["font.size"] = 15
    sns.heatmap(
        cm_prob,
        annot=cm,
        cmap="GnBu",
        xticklabels=cm_label_estimate,
        yticklabels=cm_label,
        fmt=".10g",
        # square=True,
    )

    plt.ylabel("Estimated Label")
    plt.xlabel("Ground Truth Label")
    plt.yticks(rotation=90,rotation_mode="anchor",ha="center",va="baseline")
    plt.xticks(rotation=0,)
    plt.ylim(5, 0)
    plt.title(name + " (" + date + " weeks) ")
    plt.tight_layout()

    dirpath = output_dir + "confusion_matrix/"
    os.makedirs(dirpath, exist_ok=True)
    filename = dirpath + "ConfMat_" + name + "_" + date + ".pdf"
    fig.savefig(filename)
    plt.close()

if __name__ == "__main__":

    ue_eng = {"カルビ": "kalbi", "あいぴょん": "aipyon", "あやぴょん": "ayapyon", 
              "あさぴょん": "asapyon", "真央": "mao", "ブラウニー": "brownie", 
              "ビスコッティ": "biscotti",  "ビスコッティー": "biscotti", "ビスコッテイー": "biscotti", "ビスコッテイ": "biscotti",
              "花月": "kagetsu", "黄金": "kogane", 
              "阿伏兎":"abuto", "スカイドン":"skydon", "ドラコ":"dorako", "テレスドン":"telesdon", 
              "三春":"miharu", "会津":"aizu", "鶴ヶ城":"tsurugajo", "マティアス":"matias",
              "ミコノス":"mikonos", "エバート":"ebert", "マルチナ":"martina", "ぶた玉":"butatama",
              "イカ玉":"ikatama", "梨花":"rika", "信成":"nobunari",}
    sal_eng = {"サブレ": "sable", "スフレ": "souffle",}
    vpa_eng = {"平磯": "hiraiso", "阿字ヶ浦": "azigaura", "高萩": "takahagi", "三崎": "misaki",
               "馬堀": "umahori", "八朔": "hassaku", "日向夏": "hyuganatsu", "桂島": "katsurashima",
               "松島": "matsushima",}
    vpakids_eng = {"つぐみ": "tsugumi", "ひばり": "hibari",}
    marmo_eng = {**ue_eng, **sal_eng, **vpa_eng, **vpakids_eng}

    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls"} # ラベル番号の辞書
    call_init = {v: 0 for k,v in call_label.items()} # カウント用辞書

    labelpath = "/home/muesaka/projects/marmoset/src/others/test/test/" # GroundTruth frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/src/others/test/results/" # Estimate frame .txt Path
    outputpath = labelpath

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]

    files.sort()
    results.sort()
    
    list_label = [] # (name,date,dic)のtupleを保存するリスト
    list_results = [] # (name,date,dic)のtupleを保存するリスト

    for i, file in enumerate(files):

        # nparray型のフレームラベル
        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)

        # ファイル名から週と名前を抽出
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"

        # confusion matrix
        name = marmo_eng[name[0]]
        date = date[0]
        make_confusion_matrix(label, results, [0,1,2,3,4], labelpath, name, date)
        
        break
