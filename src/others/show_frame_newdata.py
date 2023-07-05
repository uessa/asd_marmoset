# -*- coding: utf-8 -*-
#-------------------------------------#
# specとframeを並べてpdf保存するスクリプト．2021/2022両データに対応
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

import librosa.display
import seaborn as sns
from matplotlib.colors import Normalize


if __name__ == "__main__":

    # ue_eng = {"あいぴょん": "aipyon", "あやぴょん": "ayapyon", "あさぴょん": "asapyon", "真央": "mao",
    #              "ブラウニー": "brownie", 
    #              "ビスコッティ": "biscotti",  "ビスコッティー": "biscotti", "ビスコッテイー": "biscotti", "ビスコッテイ": "biscotti",
    #              "花月": "kagetsu", "黄金": "kogane", 
    #              "阿伏兎":"abuto", "スカイドン":"skydon", "ドラコ":"dorako", "テレスドン":"telesdon", 
    #              "三春":"miharu", "会津":"aizu", "鶴ヶ城":"tsurugajo", "マティアス":"matias",
    #              "ミコノス":"mikonos", "エバート":"ebert", "マルチナ":"martina", "ぶた玉":"butatama",
    #              "イカ玉":"ikatama", "梨花":"rika", "信成":"nobunari",}
    # sal_eng = {"サブレ": "sable", "スフレ": "souffle",}
    # vpa_eng = {"平磯": "hiraiso", "阿字ヶ浦": "azigaura", "高萩": "takahagi", "三崎": "misaki",
    #            "馬堀": "umahori", "八朔": "hassaku", "日向夏": "hyuganatsu", "桂島": "katsurashima",
    #            "松島": "matsushima",}
    # vpakids_eng = {"つぐみ": "tsugumi", "ひばり": "hibari",}
    # marmo_eng = {**ue_eng, **sal_eng, **vpa_eng, **vpakids_eng}
    # call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls", 5:"Phee-Trill", 6:"Trill-Phee", 7:"Unknown"}
    # trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
    #             "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    # valids = ["鶴ヶ城","ミコノス","イカ玉"]
    # tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    # lnames = ["Phee","Trill","Twitter","Other Calls"]
    # vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]

    ######################## 2022年の名前 リスト ########################
    tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    # tag = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 32, 48]
    tag = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    lab = ["Phee", "Trill", "Twitter", "Other Calls"] 

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results/" # GroundTruth frame .txt Path
    # labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka_48kHz/test/" # GroundTruth frame .txt Path
    resultpath = labelpath # Estimate frame .txt Path

    # npypath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka_48kHz/test/"
    npypath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/" # GroundTruth frame .txt Path


    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    npyfiles = [f for f in os.listdir(npypath) if os.path.isfile(os.path.join(npypath, f)) and f[-3:] == "npy"] # 末尾3文字まで（.txt）マッチ

    ######################## 2022データのソート ########################
    files = sorted(files, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    results = sorted(results, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    npyfiles = sorted(npyfiles, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))

    files = sorted(files, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])
    results = sorted(results, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])
    npyfiles = sorted(npyfiles, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])


    ######################## 2021データのソート ########################
    # files = sorted(files, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    # results = sorted(results, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    # npyfiles = sorted(npyfiles, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))

    # files = sorted(files, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    # results = sorted(results, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    # npyfiles = sorted(npyfiles, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    
    is_plot = 1


    for i, file in enumerate(files):

        ######################## Load data ########################
        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)
        npys = np.load(npypath + npyfiles[i])

        if npys.ndim == 3: # npyが2chだったら
            npys = npys[:, :, 0] # 1ch目だけ使う

        ######################## 2022データ ########################
        pattern = '[^_]*_alone_week([0-9]+).*'
        date = re.findall(pattern ,file)
        pattern = '([^_]*)_alone_week[0-9]+.*'
        name = re.findall(pattern ,file)


        ######################## 2021データ ########################
        # pattern = 'VOC_.*_.*_(.*)W'
        # date = re.findall(pattern,file) # "週"
        # pattern = 'VOC_.*_(.*)_.*'
        # name = re.findall(pattern,file) # "名前"


        ######################## Plot ########################
        if is_plot: 

            print(files[i])
            print(npyfiles[i])

            ######################## Translated to sec. ########################
            fftlen = 2048 # frame length
            fs = 48000 # sampling rate
            stime = fftlen / 2 / fs # frame -> sec translated
            label_sec = np.arange(len(label)) * stime
            results_sec = np.arange(len(results)) * stime

            ######################## Optioning plot ########################
            x_min = 0 # min_sec
            x_max = len(label) * stime # max_sec
            lw = 0.8 # plot line width

            time_slice_num = 3
            split_num = 120

            fig, ax = plt.subplots(time_slice_num*2, 1, figsize=(50, time_slice_num*4), dpi=512)
            # fig, ax = plt.subplots(time_slice_num*2, 1, figsize=(50, time_slice_num*4), dpi=100)

            title = "\"{}\" (weeks={})".format(name[0], date[0]) # title
            line_label = {0: "solid", 1: "dashed", 2: "dashdot", 3: "dotted", 4: "solid"}
            # line_label = {0: "solid", 1: "solid", 2: "solid", 3: "solid", 4: "solid"}
            color_label = {0: "black", 1: "crimson", 2: "darkgreen", 3: "mediumblue", 4: "gold"}

            ######################## Axis forloop ########################
            for i in range(time_slice_num*2):
                # print(i)

                ######################## Plot spec-frame ########################
                if i % 2 == 0:

                    # スペクトログラム
                    ref = np.median(np.abs(npys))
                    powspec = librosa.amplitude_to_db(np.abs(npys), ref=ref)
                    powspec = np.squeeze(powspec)

                    librosa.display.specshow(
                        powspec,
                        sr=48000,
                        hop_length=1024,
                        # n_fft=2048,
                        cmap="rainbow_r",
                        # cmap="gray",
                        x_axis="time",
                        y_axis="hz",
                        norm=Normalize(vmin=-10, vmax=2),
                        # norm=Normalize(vmin=-30, vmax=2),
                        ax=ax[i],
                        rasterized=True,
                    )
                    # print("spec")                  
                else:

                    ax[i].plot(results_sec, results == 1, "crimson",label="1:"+call_label[1], lw=lw, linestyle=line_label[1]) # Phee
                    ax[i].plot(results_sec, results == 2, "darkgreen",label="2:"+call_label[2], lw=lw, linestyle=line_label[2]) # Trill
                    ax[i].plot(results_sec, results == 3, "mediumblue",label="3:"+call_label[3], lw=lw, linestyle=line_label[3]) # Twitter
                    ax[i].plot(results_sec, results == 4, "gold",label="4:"+call_label[4], lw=lw, linestyle=line_label[4]) # Other Calls
                    ax[i].plot(results_sec, results == 5, "black", lw=lw, linestyle=line_label[0]) # No Call -> black

                    ax[i].set_yticks([0,1],["NoCall","Call"], fontsize=20)
                    # print("frame")
                
                if i == 0 or i == 1:
                    ax[i].set_xlim([0, split_num])
                    ax[i].set_xticks(np.arange(0, split_num+1, step=5), fontsize=15)
                elif i == 2 or i == 3:
                    ax[i].set_xlim([split_num, split_num*2])
                    ax[i].set_xticks(np.arange(split_num, split_num*2+1, step=5), fontsize=15)
                else:
                    ax[i].set_xlim([split_num*2, split_num*3])
                    ax[i].set_xticks(np.arange(split_num*2, split_num*3+1, step=5), fontsize=15)
                    if i == 5:
                        ax[i].legend(fontsize=40, bbox_to_anchor=(1, 1), loc='upper right', ncol=4, )
                    else:
                        continue               

            ######################## Label titles ########################
            
            fig.supxlabel('Time [s]', fontsize=40)
            fig.supylabel('Estimation', fontsize=40)
            fig.suptitle("{}".format(title), fontsize=40)


            ######################## Saving plot ########################
            plt.tight_layout(rect=[0.02,0,1,1]) # plot layout spacing
            savepath = labelpath + "frame/" + "frame_" + name[0] + "_" + date[0] + ".pdf"
            plt.savefig(savepath)
            plt.close()

            print("save: {}".format(savepath))
            
            ######################## Break forloop ########################
            break

