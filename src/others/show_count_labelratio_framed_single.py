# -*- coding: utf-8 -*-
#-------------------------------------#
# 2023/04/20 2022年データの可視化用にソース変更
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

    ######################## 2022年の名前 リスト ########################
    tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    # tag = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 32, 48]
    tag = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    lab = ["Phee", "Trill", "Twitter", "Other Calls"] 
    

    ######################## 描画モード 年度 UE/VPAのパス を選択 ########################
    mode = "ue ratio"
    # mode = "ue total"
    # mode = "vpa ratio"
    # mode = "vpa total"

    year = "2021"
    # year = "2022"

    resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka_48kHz/test/results") # 2021 UE
    # resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka_48kHz/test/results") # 2021 VPA
    # resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results") # 2022 UE
    # resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_vpa_48kHz/test/results") # 2022 VPA

    ######################## ファイルリスト生成 ########################
    resultlist = list(resultpath.glob("*.txt"))
    resultlist = sorted(resultlist)

    ######################## tagでforループ ########################
    result_data =  []
    for i in tag:
        print("--- week: {} ----------------------------------------".format(i))
        print("")

        # カウント辞書初期化
        dict_result = call_init.copy()
        
        ######################## resultpathでforループ ########################
        for j in range(len(resultlist)):

            ######################## 2022データ 週iでなければcontinue ########################
            # pattern = '[^_]*_alone_week([0-9]+).*'
            # tmp = re.findall(pattern ,str(resultlist[j]))[0]
            # if not int(tmp) == i:
            #     continue
            
            ######################## 2021データ 週iでなければcontinue ########################
            pattern = 'VOC_[0-9]+[-_][0-9]+_+[^_]*_+([^_]*).*'
            tmp = re.findall(pattern ,str(resultlist[j]))[0].replace("W","")
            tmp = tmp.split(".")[0]
            if not int(tmp) == i:
                continue
            
            ######################## np.loadtxt ########################
            results = np.loadtxt(resultlist[j],dtype=int)

            ######################## resultsでforループ ########################
            for n in range(len(results)):
                result = results[n]

                ######################## labelとresultを照応し該当のothercallsのみを飛ばす ########################
                # if label == 8 or label == 9 or label == 10 or label == 11:
                #     print(label)

                # # labelが有声，1,2,3,4であるとき
                # if label == 1 or label == 2 or label == 3 or label == 4:
                #     tmp1 = call_label[label]
                #     dict_label[tmp1] = dict_label.get(tmp1, 0) + 1

                #     # さらに，resultが有声，1,2,3,4であるとき（labelが有声であるとき）
                #     if result == 1 or result == 2 or result == 3 or result == 4:
                #         tmp2 = call_label[result]
                #         dict_result[tmp2] = dict_result.get(tmp2, 0) + 1

                if result == 1 or result == 2 or result == 3 or result == 4:
                    tmp2 = call_label[result]
                    dict_result[tmp2] = dict_result.get(tmp2, 0) + 1    

            print("result",dict_result)
            print("")

        ######################## 週ごとの集計をappend ########################
        result_split = []
        d2 = {}

        result_total = sum(dict_result.values())

        for n in dict_result:
            d2[n] = dict_result[n]/result_total


        ######################## Ratio or Total選択 ########################

        # Ratio
        if mode == 'ue ratio' or mode == 'vpa ratio':
            d2_ratio = list(d2.items())
        # Total
        else:
            d2_ratio = list(dict_result.items())

        print("result")
        for m in d2_ratio:
            print(m[0].ljust(10), '{}'.format(m[1]))
            result_split.append(m[1])
        print("")

        result_data.append(result_split)

        # break

    ######################## dataframe, hatch, cmap作成 ########################
    df_result = pd.DataFrame(result_data, index=tag, columns=lab)
    print(df_result)
    hatches = ['', '', '', ''] # hatchパターン
    cmap = plt.get_cmap("Blues") # cmap color
    a = [cmap(0.1), cmap(0.3), cmap(0.6), cmap(1.0)] # cmap index
    label_color = {"Phee":a[0] ,"Trill":a[1],"Twitter":a[2],"Other Calls":a[3]} # label to color index

    ######################## モードごとにプロット ###########################
    ax = plt.subplot(111)
    df_result.plot.bar(ax=ax, stacked=True, edgecolor="black", 
                        color=label_color, legend=False)
    patches = ax.patches
    for i, patch in enumerate(patches):
        patch.set_hatch(hatches[i//len(df_result.index)])
    hans, labs = ax.get_legend_handles_labels()
    plt.xlabel("Week", fontsize=20)
    plt.xticks(fontsize=15, rotation=45)


    ''' UE Ratio '''
    if mode == 'ue ratio':
        plt.ylabel("Ratio frame", fontsize=20)
        plt.yticks(fontsize=15, rotation=0)
        ax.legend(fontsize=20, loc="lower left", handles=hans[::-1], labels=labs[::-1])
        plt.tight_layout()
        if year == "2021":
            plt.savefig("./LabelRatio/ratio_frame_2021_estimated_ue.pdf")
        else:
            plt.savefig("./LabelRatio/ratio_frame_2022_estimated_ue.pdf")
        plt.close()

    ''' UE Total '''
    if mode == 'ue total':
        plt.ylabel("Total frame", fontsize=20)
        plt.yticks(fontsize=15, rotation=0)
        ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
        plt.tight_layout()
        if year == "2021":
            plt.savefig("./LabelRatio/total_frame_2021_estimated_ue.pdf")
        else:
            plt.savefig("./LabelRatio/total_frame_2022_estimated_ue.pdf")
        
        plt.close()
    
    ''' VPA Ratio '''
    if mode == 'vpa ratio':
        plt.ylabel("Ratio frame", fontsize=20)
        plt.yticks(fontsize=15, rotation=0)
        ax.legend(fontsize=20, loc="lower right", handles=hans[::-1], labels=labs[::-1])
        plt.tight_layout()
        if year == "2021":
            plt.savefig("./LabelRatio/ratio_frame_2021_estimated_vpa.pdf")
        else:
            plt.savefig("./LabelRatio/ratio_frame_2022_estimated_vpa.pdf")
        plt.close()

    ''' VPA Total '''
    if mode == 'vpa total':
        plt.ylabel("Total Frame", fontsize=20)
        plt.yticks(fontsize=15, rotation=0)
        ax.legend(fontsize=20, loc="upper right", handles=hans[::-1], labels=labs[::-1])
        plt.tight_layout()
        if year == "2021":
            plt.savefig("./LabelRatio/total_frame_2021_estimated_vpa.pdf")
        else:
            plt.savefig("./LabelRatio/total_frame_2022_estimated_vpa.pdf")
        plt.close()