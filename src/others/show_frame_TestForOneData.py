# -*- coding: utf-8 -*-
#-------------------------------------#
# specとframeを並べてpdf保存するスクリプト．2021/2022両データに対応
# 6/19 2021データのspec-frame-frameの3表示をするように変更
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

def make_fig(waveforms, predicted, labels, out_dir):
    # Spectrogram_GroundTruthLabel_EstimateLabel
    predicted = np.squeeze(predicted)
    labels = np.squeeze(labels)
    start = 1
    # stop = labels.size()[0]
    stop = labels.size
    end_time = stop * 1024 / 96000
    time = np.linspace(0, end_time, stop)

    plt.rcParams["font.family"] = "Times New Roman" # フォントファミリー
    plt.rc('pdf', fonttype=42) # フォントを埋め込む（Type1，つまり42を指定）
    line_width = 2.5 # 線の太さ
    title_size = 40 # タイトルの大きさ
    legend_size = 35 # legendの大きさ
    tick_size = 30 # tick大きさ
    LEN = [60, 80]
    # LEN = [200, 220]


    # スペクトログラム
    ref = np.median(np.abs(waveforms))
    powspec = librosa.amplitude_to_db(np.abs(waveforms), ref=ref)
    powspec = np.squeeze(powspec)
    fig, ax = plt.subplots(
        3, 1, figsize=(40, 10), gridspec_kw={"height_ratios": [3, 1, 1]}, dpi=200
    )
    ax[0].set_title("Spectrogram", fontsize=title_size)
    librosa.display.specshow(
        powspec,
        sr=96000,
        hop_length=1024,
        cmap="rainbow_r",
        # cmap="RdYlBu",
        x_axis="time",
        y_axis="hz",
        norm=Normalize(vmin=-10, vmax=2),
        ax=ax[0],
        rasterized=True,
    )
    
    # LEN = [197.2, 206.8]
    # LEN = [200, 202]

    ax[0].set_xlim(LEN)
    # ax[0].set_xticks([198,200,202,204,206])
    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    # ax[0].set_xticks([200, 210, 220, 230, 240, 250], [200, 210, 220, 230, 240, 250])
    # ax[0].set_xticklabels([200, "", 210, "", 220, "", 230, "", 240, "", 250])
    # ax[0].set_xticklabels([115, "", 120, "", 125])
    ax[0].set_yticks([0, 20000, 40000])
    ax[0].set_yticklabels([0, 10, 40])
    ax[0].set_ylabel("Frequency [kHz]", fontsize=tick_size)
    ax[0].set_xlabel("")

    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(ax[0].get_xticks(), y=-0.1)

    ax[0].tick_params(labelsize=tick_size)
    # plt.colorbar(format="%+2.0fdB")



    # 正解ラベル
    ax[1].set_title("Annotated Label", fontsize=title_size)
    color_label = {0: "black", 1: "crimson", 2: "darkgreen", 3: "mediumblue", 4: "gold"}
    call_label = {0: "no call", 1: "phee", 2: "trill", 3: "twitter", 4: "other calls"}
    # line_label = {0: "dashdot", 1: "dashed", 2: "dashed", 3: "dotted", 4: "solid"}
    line_label = {0: "solid", 1: "dashed", 2: "dashdot", 3: "dotted", 4: "solid"}

    for i in range(max(labels)):
        first_plot = True
        labels_convert = labels == (i + 1)
        j = 0
        while j < len(labels):
            if labels_convert[j]:
                start_call = j - 1
                while labels_convert[j]:
                    j = j + 1
                end_call = j + 1
                if first_plot:
                    ax[1].plot(
                        time[start_call:end_call],
                        labels_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=line_width,
                        color=color_label[i + 1],
                        label=call_label[i + 1],
                    )
                    first_plot = False
                else:
                    ax[1].plot(
                        time[start_call:end_call],
                        labels_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=line_width,
                        color=color_label[i + 1],
                    )
            else:
                j = j + 1

    k = 0
    first_plot = True
    while k < len(labels):
        if labels[k] == 0:
            start_no_call = k
            while labels[k] == 0:
                k = k + 1
                if k >= len(labels):
                    break
            end_no_call = k - 1
            if first_plot:
                ax[1].plot(
                    time[start_no_call:end_no_call],
                    labels[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=line_width,
                    color=color_label[0],
                    label=call_label[0],
                )
                first_plot = False
            else:
                ax[1].plot(
                    time[start_no_call:end_no_call],
                    labels[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=line_width,
                    color=color_label[0],
                )
        else:
            k = k + 1
            if k >= len(labels):
                break
    # plt.plot(time, labels/4)
    # plt.plot(time, predicted*0.8, linewidth=1)
    ax[1].set_xlim(LEN)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls'])
    ax[1].set_yticks([0.0, 1.0])
    ax[1].set_yticklabels(["No Call", "Call"])

    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(ax[1].get_xticks(), y=-0.1)

    # ax[1].legend(loc="center right", fontsize="18")
    ax[1].tick_params(labelsize=tick_size)

    # 推定ラベル
    ax[2].set_title("Estimated Label", fontsize=title_size)
    for i in range(max(predicted)):
        first_plot = True
        predicted_convert = predicted == (i + 1)
        j = 0
        while j < len(predicted):
            if predicted_convert[j]:
                start_call = j - 1
                while predicted_convert[j]:
                    j = j + 1
                end_call = j + 1
                if first_plot:
                    ax[2].plot(
                        time[start_call:end_call],
                        predicted_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=line_width,
                        color=color_label[i + 1],
                        label=call_label[i + 1],
                    )
                    first_plot = False
                else:
                    ax[2].plot(
                        time[start_call:end_call],
                        predicted_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=line_width,
                        color=color_label[i + 1],
                    )
            else:
                j = j + 1

    k = 0
    first_plot = True
    while k < len(predicted):
        if predicted[k] == 0:
            start_no_call = k
            while predicted[k] == 0:
                k = k + 1
                if k >= len(labels):
                    break
            end_no_call = k - 1
            if first_plot:
                ax[2].plot(
                    time[start_no_call:end_no_call],
                    predicted[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=line_width,
                    color=color_label[0],
                    label=call_label[0],
                )
                first_plot = False
            else:
                ax[2].plot(
                    time[start_no_call:end_no_call],
                    predicted[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=line_width,
                    color=color_label[0],
                )
        else:
            k = k + 1
            if k >= len(labels):
                break
    # plt.plot(time, predicted/4)
    ax[2].set_xlim(LEN)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls'])
    ax[2].set_yticks([0.0, 1.0])
    ax[2].set_yticklabels(["No Call", "Call"])
    ax[2].set_xlabel("Time [s]", fontsize=tick_size)

    ax[2].set_xticks(ax[2].get_xticks())
    ax[2].set_xticklabels(ax[2].get_xticks(), y=-0.1)

    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -1.1), fontsize=legend_size, ncol=5)

    ax[2].tick_params(labelsize=tick_size)
    plt.tight_layout()
    # plt.savefig(out_dir / "spec_labels.pdf")
    plt.savefig("/home/muesaka/projects/marmoset/src/others/LabelRatio/spec_labels.pdf")






if __name__ == "__main__":

    ######################## 2022年の名前 リスト ########################
    tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}

    # labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results/" # GroundTruth frame .txt Path
    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/" # GroundTruth frame .txt Path
    # labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/" # GroundTruth frame .txt Path

    # resultpath = labelpath # Estimate frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results/"
    # resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/results/"

    npypath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/"
    # npypath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/"
    # npypath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/" # GroundTruth frame .txt Path


    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    npyfiles = [f for f in os.listdir(npypath) if os.path.isfile(os.path.join(npypath, f)) and f[-3:] == "npy"] # 末尾3文字まで（.txt）マッチ

    ######################## 2022データのソート ########################
    # files = sorted(files, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    # results = sorted(results, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    # npyfiles = sorted(npyfiles, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))

    # files = sorted(files, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])
    # results = sorted(results, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])
    # npyfiles = sorted(npyfiles, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])


    ######################## 2021データのソート ########################
    files = sorted(files, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results = sorted(results, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    npyfiles = sorted(npyfiles, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))

    files = sorted(files, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results = sorted(results, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    npyfiles = sorted(npyfiles, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    
    is_plot = 1


    for i, file in enumerate(files):

        ######################## Load data ########################
        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)
        npys = np.load(npypath + npyfiles[i])

        if npys.ndim == 3: # npyが2chだったら
            npys = npys[:, :, 0] # 1ch目だけ使う

        ######################## 2022データ ########################
        # pattern = '[^_]*_alone_week([0-9]+).*'
        # date = re.findall(pattern ,file)
        # pattern = '([^_]*)_alone_week[0-9]+.*'
        # name = re.findall(pattern ,file)


        ######################## 2021データ ########################
        pattern = 'VOC_.*_.*_(.*)W'
        date = re.findall(pattern,file) # "週"
        pattern = 'VOC_.*_(.*)_.*'
        name = re.findall(pattern,file) # "名前"


        ######################## plot ########################
        if is_plot: 

            print(files[i])
            print(npyfiles[i])

            make_fig(npys, results, label, "LabelRatio")
            

