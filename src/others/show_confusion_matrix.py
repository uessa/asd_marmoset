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
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages


# 混同行列1つをプロット
def make_confusion_matrix(cm, outputpath): # 
    cm_label = ['no call', 'phee', 'trill', 'twitter', 'tsik', 'ek', 'ek-tsik', 'cough', 'cry', 'chatter', 'breath']
    # cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls']
    cm_pred = ['no call', 'phee', 'trill', 'twitter','other calls']
        
    cm = np.delete(cm, slice(len(cm_pred), len(cm_label)), 1) # cm_predの次元数に合わせてcmを削除
    print(cm)

    # 行毎に確率値を出して色分け
    cm_prob = cm / np.sum(cm, axis=1, keepdims=True)

    fig = plt.figure(figsize=(8,6))
    # fig = plt.figure(figsize=(8,4))
    plt.rcParams["font.size"] = 18 # フォントサイズ
    plt.rcParams["font.family"] = "Times New Roman" # フォントファミリー

    sns.heatmap(
        cm_prob,
        annot=cm,
        cmap="Blues",
        # cmap="Greys",
        xticklabels=cm_pred,
        yticklabels=cm_label,
        fmt=".10g",
        cbar_kws=dict(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    )

    plt.xlabel("Estimated Label", fontsize=25)
    plt.ylabel("Annotated Label", fontsize=25)
    plt.yticks(rotation=0,rotation_mode="anchor",ha="right",)
    plt.xticks(rotation=30,)
    plt.ylim(len(cm_label), 0)

    plt.tight_layout()

    outputpath = outputpath
    os.makedirs(outputpath, exist_ok=True)
    filename = outputpath / "confusion.pdf"
    fig.savefig(filename)
    plt.close()
    print("save: {}".format(filename))

if __name__ == "__main__":

    
    # 正解データ，推定データのディレクトリ，結果出力のディレクトリ
    labelpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test")
    resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results_5class_before")
    outputpath = pathlib.Path("./LabelRatio/")

    # 正解データ，推定データのファイル名をリスト化
    labels = list(labelpath.glob("*.txt"))
    results = list(resultpath.glob("*.txt"))

    labels = sorted(labels)
    results = sorted(results)

    # 可視化処理
    label_classes = 11
    # label_classes = 5
    pred_classes = 5
    cm = np.zeros((label_classes, label_classes), dtype=int)
    for i in range(len(labels)):

        # nparray型のフレームデータ
        label = np.loadtxt(labels[i], dtype=int) # 正解
        pred = np.loadtxt(results[i], dtype=int) # 推定

        print("label",len(label))
        print(labels[i])
        print("pred",len(pred))
        print(results[i])

        # wo classを指定し削除
        tmp_label = []
        tmp_pred = []
        for j in range(len(label)):
            if label[j] == 11 or label[j] == 12 or label[j] == 13:
            # if label[j] == 5 or label[j] == 6 or label[j] == 7:
                continue
            else:
                tmp_label.append(label[j])
                tmp_pred.append(pred[j])
        label = np.array(tmp_label, dtype=int)
        pred = np.array(tmp_pred, dtype=int)

        print("label",len(label))
        print("pred",len(pred))

        # 混同行列インスタンスの生成
        cm += confusion_matrix(label, pred, labels=range(label_classes))
        # break
        
    # 可視化
    make_confusion_matrix(cm, outputpath)
        
