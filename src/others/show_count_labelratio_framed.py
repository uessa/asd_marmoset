# -*- coding: utf-8 -*-
#-------------------------------------#
# 2023/04/19 2021年データの可視化を確認
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

if __name__ == "__main__":

    ######################## 2021年の名前のリスト ########################
    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]
    call_label = {1: "phee",2: "trill", 3: "twitter", 4: "other calls"}
    call_init = {'phee':0, 'trill':0, 'twitter':0, 'other calls':0}
    tag = [3,4,5,6,7,8,9,10,11,12,13,14]
    lab = ["phee","trill","twitter","other calls"]    

    ######################## ファイルリスト作成 ########################
    labelpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/test_8class_before")
    # labelpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/test_8class_before")


    resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results_5class_before")
    # resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/results_5class_before")
    labellist = list(labelpath.glob("*.txt"))
    resultlist = list(resultpath.glob("*.txt"))
    labellist = sorted(labellist)
    resultlist = sorted(resultlist)

    ######################## tagでforループ ########################
    label_data = []
    result_data =  []
    for i in tag:
        print("--- week: {} ----------------------------------------".format(i))
        print("")

        # カウント辞書初期化
        dict_label = call_init.copy()
        dict_result = call_init.copy()
        
        ######################## labelpathでforループ ########################
        weeks = []
        for j in range(len(labellist)):

            ######################## 週iでなければcontinue ########################
            pattern = 'VOC_[0-9]+[-_][0-9]+_+[^_]*_+([^_]*).*'
            tmp = re.findall(pattern ,str(labellist[j]))[0].replace("W","")
            tmp = tmp.split(".")[0]
            if not int(tmp) == i:
                continue
            
            ######################## np.loadtxt ########################
            labels = np.loadtxt(labellist[j],dtype=int)
            results = np.loadtxt(resultlist[j],dtype=int)

            ######################## labelsでforループ ########################
            for n in range(len(labels)):
                label = labels[n]
                result = results[n]

                ######################## labelとresultを照応し該当のothercallsのみを飛ばす ########################
                if label == 8 or label == 9 or label == 10 or label == 11:
                    print(label)

                # labelが有声，1,2,3,4であるとき
                if label == 1 or label == 2 or label == 3 or label == 4:
                    tmp1 = call_label[label]
                    dict_label[tmp1] = dict_label.get(tmp1, 0) + 1

                    # さらに，resultが有声，1,2,3,4であるとき（labelが有声であるとき）
                    if result == 1 or result == 2 or result == 3 or result == 4:
                        tmp2 = call_label[result]
                        dict_result[tmp2] = dict_result.get(tmp2, 0) + 1

            print("label",dict_label)
            print("result",dict_result)
            print("")

        ######################## 週ごとの集計をappend ########################
        label_split = []
        result_split = []
        d1 = {}
        d2 = {}

        label_total = sum(dict_label.values())
        result_total = sum(dict_result.values())

        for n in dict_label:
            d1[n] = dict_label[n]/label_total
            d2[n] = dict_result[n]/result_total


        ######################## Ratio or Total選択 ########################
        ''' Ratio '''
        d1_ratio = list(d1.items())
        d2_ratio = list(d2.items())

        ''' Total '''
        # d1_ratio = list(dict_label.items())
        # d2_ratio = list(dict_result.items())

        print("label")
        for m in d1_ratio:
            print(m[0].ljust(10), '{}'.format(m[1]))
            label_split.append(m[1])
        print("")

        print("result")
        for m in d2_ratio:
            print(m[0].ljust(10), '{}'.format(m[1]))
            result_split.append(m[1])
        print("")

        label_data.append(label_split)
        result_data.append(result_split)

        # break

    ######################## dataframe, hatch, cmap作成 ########################
    df_label = pd.DataFrame(label_data, index=tag, columns=lab)
    df_result = pd.DataFrame(result_data, index=tag, columns=lab)
    print(df_label)
    print(df_result)
    hatches = ['', '', '', ''] # hatchパターン
    plt.rcParams["font.family"] = "Times New Roman" # フォントファミリー
    cmap = plt.get_cmap("Blues") # cmap color
    a = [cmap(0.1), cmap(0.3), cmap(0.6), cmap(1.0)] # cmap index
    label_color = {"phee":a[0] ,"trill":a[1],"twitter":a[2],"other calls":a[3]} # label to color index


    ######################## Target Label ###########################
    ax = plt.subplot(111)
    df_label.plot.bar(ax=ax, stacked=True, edgecolor="black", 
                        color=label_color, legend=False)
    patches = ax.patches
    for i, patch in enumerate(patches):
        patch.set_hatch(hatches[i//len(df_label.index)])
    hans, labs = ax.get_legend_handles_labels()
    plt.xlabel("Week", fontsize=25)
    plt.xticks(fontsize=20, rotation=45)

    ''' Ratio '''
    plt.ylabel("Proportion of calls", fontsize=25)
    ax.legend(fontsize=20, loc="lower right", handles=hans[::-1], labels=labs[::-1])
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig("./LabelRatio/ratio_frame_ue_target.pdf")
    # plt.savefig("./LabelRatio/ratio_frame_vpa_target.pdf")
    plt.close()

    ''' Total UE '''
    # plt.ylabel("Total Frame", fontsize=20)
    # ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
    # plt.yticks([0,10000,20000,30000,40000,50000,60000,70000], 
    #             ["0","10k","20k","30k","40k","50k","60k","70k"], 
    #             fontsize=20, rotation=0)
    # plt.ylim(0,72000)
    # plt.tight_layout()
    # plt.savefig("./LabelRatio/total_frame_ue_target.svg")
    # plt.close()

    ''' Total VPA '''
    # plt.ylabel("Total Frame", fontsize=20)
    # ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
    # plt.yticks([0,20000,40000,60000,80000,100000,120000,140000], 
    #             ["0","20k","40k","60k","80k","100k","120k","140k"], 
    #             fontsize=20, rotation=0)
    # plt.ylim(0,145000)
    # plt.tight_layout()
    # plt.savefig("./LabelRatio/total_frame_vpa_target.svg")
    # plt.close()


    ######################## Estimated Label ###########################
    ax = plt.subplot(111)
    df_result.plot.bar(ax=ax, stacked=True, edgecolor="black", 
                        color=label_color, legend=False)
    patches = ax.patches
    for i, patch in enumerate(patches):
        patch.set_hatch(hatches[i//len(df_result.index)])
    hans, labs = ax.get_legend_handles_labels()
    plt.xlabel("Week", fontsize=25)
    plt.xticks(fontsize=20, rotation=45)

    ''' Ratio '''
    plt.ylabel("Proportion of calls", fontsize=25)
    ax.legend(fontsize=20, loc="lower right", handles=hans[::-1], labels=labs[::-1])
    plt.yticks(fontsize=20,rotation=0)
    plt.tight_layout()
    plt.savefig("./LabelRatio/ratio_frame_ue_estimated.pdf")
    # plt.savefig("./LabelRatio/ratio_frame_vpa_estimated.pdf")
    plt.close()

    ''' Total UE '''
    # plt.ylabel("Total Frame", fontsize=20)
    # ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
    # plt.yticks([0,10000,20000,30000,40000,50000,60000,70000], 
    #             ["0","10k","20k","30k","40k","50k","60k","70k"], 
    #             fontsize=20, rotation=0)
    # plt.ylim(0,72000)
    # plt.tight_layout()
    # plt.savefig("./LabelRatio/total_frame_ue_estimated.svg")
    # plt.close()

    ''' Total VPA '''
    # plt.ylabel("Total Frame", fontsize=20)
    # ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
    # plt.yticks([0,20000,40000,60000,80000,100000,120000,140000], 
    #             ["0","20k","40k","60k","80k","100k","120k","140k"], 
    #             fontsize=20, rotation=0)
    # plt.ylim(0,145000)
    # plt.tight_layout()
    # plt.savefig("./LabelRatio/total_frame_vpa_estimated.svg")
    # plt.close()
