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



if __name__ == "__main__":

    ue_eng = {"あいぴょん": "aipyon", "あやぴょん": "ayapyon", "あさぴょん": "asapyon", "真央": "mao",
                 "ブラウニー": "brownie", 
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
    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls", 5:"Phee-Trill", 6:"Trill-Phee", 7:"Unknown"}
    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]


    is_plot = 1 # plot on/off

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test_wo_pheetrill/" # GroundTruth frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test/results/" # Estimate frame .txt Path
    # labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test_wo_pheetrill/labels/" # GroundTruth frame .txt Path
    # resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_hkawauchi/test_wo_pheetrill/results/" # Estimate frame .txt Path

    # pdf = PdfPages(labelpath + 'frame_5ue.pdf') # filename

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    files = sorted(files, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results = sorted(results, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    files = sorted(files, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results = sorted(results, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    for i, file in enumerate(files):
        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)

        total = len(label)
        corr = np.sum(label == results)
        acc = np.round(corr / total * 100, 1)
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"
        print(
            "Correct/Total (Acc): {}/{} ({}%)".format(corr, total, acc),
            name[0],
            date[0],
        )

        if is_plot: #is_plot = 1でプロット

            # sec translate
            fftlen = 2048 # frame length
            fs = 96000 # sampling rate
            stime = fftlen / 2 / fs # frame -> sec translated
            label_sec = np.arange(len(label)) * stime
            results_sec = np.arange(len(results)) * stime

            # frame plot option
            x_min = 0 # min_sec
            x_max = len(label) * stime # max_sec
            lw = 0.5 # plot line width
            plt.figure(figsize=(50,3),dpi=400) # plot figure size
            title = "{} (weeks={}, acc={}%)".format(marmo_eng[name[0]], date[0], acc) # plot title

            # Ground Truth Label
            plt.subplot(2, 1, 1)
            plt.plot(label_sec, label == 1, "b", label="1:"+call_label[1], lw=lw) # Phee
            plt.plot(label_sec, label == 2, "g", label="2:"+call_label[2], lw=lw) # Trill
            plt.plot(label_sec, label == 3, "r", label="3:"+call_label[3], lw=lw) # Twitter
            plt.plot(label_sec, label == 5, "c", label="5:"+call_label[5], lw=lw) # Phee-Trill
            plt.plot(label_sec, label == 6, "m", label="6:"+call_label[6], lw=lw) # Trill-Phee
            # plt.plot(label_sec, label == 4, "c", label="4:"+call_label[4], lw=lw) # Other Calls
            plt.xlim([x_min,x_max])
            plt.yticks([0,1],["NoCall","Call"])
            plt.title(title)
            plt.legend(fontsize=16, bbox_to_anchor=(0, -.2), loc='upper left', ncol=5, )
            plt.ylabel("G.T.")

            # Estimated Label
            plt.subplot(2, 1, 2)
            plt.plot(results_sec, results == 1, "b",label="1:"+call_label[1], lw=lw) # Phee
            plt.plot(results_sec, results == 2, "g",label="2:"+call_label[2], lw=lw) # Trill
            plt.plot(results_sec, results == 3, "r",label="3:"+call_label[3], lw=lw) # Twitter
            # plt.plot(results_sec, results == 4, "c",label="4:"+call_label[4], lw=lw) # Other Calls
            plt.xlim([x_min,x_max])
            plt.yticks([0,1],["NoCall","Call"])
            plt.ylabel("Est.")
            plt.xlabel("Time (sec) ")

            # plot option after plot1,2
            plt.tight_layout() # plot layout spacing
            plt.savefig(labelpath + "frame_" + marmo_eng[name[0]] + date[0] + ".pdf")
            plt.close()
            

