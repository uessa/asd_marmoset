# -*- coding: utf-8 -*-

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
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages


# 混同行列1つをプロット
def make_confusion_matrix(label, pred, num_classes, outputpath, name, date): # 
    # cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Tsik', 'Ek', 'Ek-Tsik', 'Cough', 'Cry', 'Chatter', 'Breath', 'Unknown']
    cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls', 'Phee-Trill', 'Trill-Phee', 'Unknown']
    cm_pred = ['No Call', 'Phee', 'Trill', 'Twitter','Other Calls']
    classes = np.arange(num_classes)
        
    cm = confusion_matrix(pred, label, classes) # num_classes^2 の混同行列の作成
    cm = np.delete(cm, slice(len(cm_pred), len(cm_label)), 0) # cm_predの次元数に合わせてcmの行を削除
    print(name, date)
    print(cm)

    # 行毎に確率値を出して色分け
    # cm_prob = cm / np.sum(cm, axis=1, keepdims=True)

    fig = plt.figure(figsize=(12, 5))
    plt.rcParams["font.size"] = 15
    sns.heatmap(
        cm,
        annot=cm,
        cmap="GnBu",
        xticklabels=cm_label,
        yticklabels=cm_pred,
        fmt=".10g",
        # square=True,
    )

    plt.ylabel("Estimated Label")
    plt.xlabel("Manually atattched Label")
    plt.yticks(rotation=0,rotation_mode="anchor",ha="right",)
    plt.xticks(rotation=30,)
    plt.ylim(len(cm_pred), 0)
    title = "{} ({} weeks) ".format(name, date)
    plt.title(title)
    plt.tight_layout()

    outputpath = outputpath / "confusion_matrix"
    os.makedirs(outputpath, exist_ok=True)
    filename = outputpath / "confusion_{}_{}.pdf".format(name, date)
    fig.savefig(filename)
    plt.close()
    print("save: {}".format(filename))

# 混同行列まとめてプロット用
def temp_confusion_matrix(label, pred, num_classes, name, date): # 
    # cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Tsik', 'Ek', 'Ek-Tsik', 'Cough', 'Cry', 'Chatter', 'Breath', 'Unknown']
    cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls', 'Phee-Trill', 'Trill-Phee', 'Unknown']
    cm_pred = ['No Call', 'Phee', 'Trill', 'Twitter','Other Calls']
    classes = np.arange(num_classes)
        
    cm = confusion_matrix(pred, label, classes) # num_classes^2 の混同行列の作成
    cm = np.delete(cm, slice(len(cm_pred), len(cm_label)), 0) # cm_predの次元数に合わせてcmの行を削除
    print(name, date)
    print(cm)

    # 行毎に確率値を出して色分け
    # cm_prob = cm / np.sum(cm, axis=1, keepdims=True)

    fig = plt.figure(figsize=(12, 5))
    plt.rcParams["font.size"] = 15
    sns.heatmap(
        cm,
        annot=cm,
        cmap="GnBu",
        xticklabels=cm_label,
        yticklabels=cm_pred,
        fmt=".10g",
        # square=True,
    )

    plt.ylabel("Estimated Label")
    plt.xlabel("Manually atattched Label")
    plt.yticks(rotation=0,rotation_mode="anchor",ha="right",)
    plt.xticks(rotation=30,)
    plt.ylim(len(cm_pred), 0)
    title = "{} ({} weeks) ".format(name, date)
    plt.title(title)
    plt.tight_layout()

if __name__ == "__main__":

    # 個体名の英語辞書
    ue_eng = {"カルビ": "kalbi", "あいぴょん": "aipyon", "あやぴょん": "ayapyon", 
              "あさぴょん": "asapyon", "真央": "mao", "ブラウニー": "brownie", 
              "ビスコッティー": "biscotti", "ビスコッテイー": "biscotti", 
              "花月": "kagetsu", "黄金": "kogane", "阿伏兎":"abuto", "スカイドン":"skydon", 
              "ドラコ":"dorako", "テレスドン":"telesdon", "イカ玉":"ikatama", "梨花":"rika", "信成":"nobunari",
              "三春":"miharu", "会津":"aizu", "鶴ヶ城":"tsurugajo", "マティアス":"matias",
              "ミコノス":"mikonos", "エバート":"ebert", "マルチナ":"martina", "ぶた玉":"butatama",}
    sal_eng = {"サブレ": "sable", "スフレ": "souffle",}
    vpa_eng = {"平磯": "hiraiso", "阿字ヶ浦": "azigaura", "高萩": "takahagi", "三崎": "misaki","馬堀": "umahori", 
               "八朔": "hassaku", "日向夏": "hyuganatsu", "桂島": "katsurashima","松島": "matsushima",}
    vpakids_eng = {"つぐみ": "tsugumi", "ひばり": "hibari",}
    marmo_eng = {**ue_eng, **sal_eng, **vpa_eng, **vpakids_eng}

    # train,valid,testおよびvpaに振り分けられた個体名リスト
    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"].sort()
    valids = ["鶴ヶ城","ミコノス","イカ玉"].sort()
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"].sort()
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"].sort()

    # 正解データ，推定データのディレクトリ，結果出力のディレクトリ
    # filepath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test_wo_pheetrill")
    # resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test/results")
    filepath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test_wo_pheetrill/labels")
    resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test_wo_pheetrill/results")
    outputpath = filepath

    # 正解データ，推定データのファイル名をリスト化
    files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    # リストのソート
    files.sort(key=lambda x:float(re.findall('VOC_.*_.*_(.*)W', x)[0])) # "週"sort by int()
    files.sort(key=lambda x:re.findall('VOC_.*_(.*)_.*', x)[0]) # "名前"sort

    # 可視化処理
    count = 0
    for i, file in enumerate(files):

        # nparray型のフレームデータ
        label = np.loadtxt(filepath / file, dtype=int) # 正解
        pred = np.loadtxt(resultpath / file, dtype=int) # 推定

        # ファイル名から週と名前を抽出
        date = re.findall('VOC_.*_.*_(.*)W', file)[0] # "週"
        name = re.findall('VOC_.*_(.*)_.*', file)[0] # "名前"

        # 混同行列
        name = marmo_eng[name] # 個体名を英名に
        # temp_confusion_matrix(label, pred, 8, name, date) # 混同行列（一時保存）
        make_confusion_matrix(label, pred, 8, outputpath, name, date) # 混同行列（各個保存）

        break
        
