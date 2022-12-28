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

correct_label = {'B': 'Breath',
                'Bbreath': 'Breath',
                'Brearth': 'Breath',
                'Breasth': 'Breath',
                'Breath': 'Breath',
                'Breathbreath': 'Breath',
                'Breathg': 'Breath',
                'Breathj': 'Breath',
                'Breathy': 'Breath',
                'Breatn': 'Breath',
                'Breatrh': 'Breath',
                'Breatyh': 'Breath',
                'Brerath': 'Breath',
                'Brwath': 'Breath',
                'Btreath': 'Breath',
                'Bvreath': 'Breath',
                'C Ry': 'Cry',
                'Chatter': 'Chatter',
                'Chirp': 'Chirp',
                'Chirpchirp': 'Chirp',
                'Cough': 'Cough',
                'Coughcough': 'Cough',
                'Cry': 'Cry',
                'E': 'Ek',
                'Ek': 'Ek',
                'Ek-Tsik': 'Ek',
                'Ekek': 'Ek',
                'Ekj': 'Ek',
                'Ex': 'Ek',
                'Hee': 'Phee',
                'Intermittentphee': 'Others',
                'Others': 'Others',
                'Phee': 'Phee',
                'Phee-Trill': 'Phee-Trill',
                'Pheecry': 'Phee',
                'PheeCry': 'Phee',
                'Pheee': 'Phee',
                'Pheephee': 'Phee',
                'Phees': 'Phee',
                'Reath': 'Breath',
                'See': 'Others',
                'Sneeze': 'Others',
                'Snooze': 'Others',
                'Treill': 'Trill',
                'Treill-Phee': 'Trill-Phee',
                'Trill': 'Trill',
                'Trill-': 'Trill-Phee',
                'Trill-Hee': 'Trill-Phee',
                'Trill-Phee': 'Trill-Phee',
                'Trill-phee': 'Trill-Phee',
                'trill-Phee': 'Trill-Phee',
                'Trilll': 'Trill',
                'Tsik': 'Tsik',
                'Tsiktsik': 'Tsik',
                'Ttwitter': 'Twitter',
                'Twiitter': 'Twitter',
                'Twiter': 'Twitter',
                'Twitteer': 'Twitter',
                'Twitter': 'Twitter',
                'Twitters': 'Twitter',
                'Twittetr': 'Twitter',
                'Twittrer': 'Twitter',
                'Twittter': 'Twitter',
                'Twitttter': 'Twitter',
                'Unk': 'Unknown',
                'Unkinown': 'Unknown',
                'unknown': 'Unknown',
                'Unknow': 'Unknown',
                'Unknown': 'Unknown',
                'Unknownhow': 'Unknown',
                'Unknownbreath': 'Unknown'}

if __name__ == "__main__":
    ue_eng = {"あいぴょん": "aipyon", "あやぴょん": "ayapyon", "あさぴょん": "asapyon", "真央": "mao",
                 "ブラウニー": "brownie", 
                 "ビスコッティ": "biscotti",  "ビスコッティー": "biscotti", "ビスコッテイー": "biscotti", "ビスコッテイ": "biscotti",
                 "花月": "kagetsu", "黄金": "kogane", 
                 "阿伏兎":"abuto", "スカイドン":"skydon", "ドラコ":"dorako", "テレスドン":"telesdon", 
                 "三春":"miharu", "会津":"aizu", "鶴ヶ城":"tsurugajo", "マティアス":"matias",
                 "ミコノス":"mikonos", "エバート":"ebert", "マルチナ":"martina", "ぶた玉":"butatama",
                 "イカ玉":"ikatama", "梨花":"rika", "信成":"nobunari","カルビ":"kalbi"}
    sal_eng = {"サブレ": "sable", "スフレ": "souffle",}
    vpa_eng = {"平磯": "hiraiso", "阿字ヶ浦": "azigaura", "高萩": "takahagi", "三崎": "misaki",
               "馬堀": "umahori", "八朔": "hassaku", "日向夏": "hyuganatsu", "桂島": "katsurashima",
               "松島": "matsushima",}
    vpakids_eng = {"つぐみ": "tsugumi", "ひばり": "hibari",}
    marmo_eng = {**ue_eng, **sal_eng, **vpa_eng, **vpakids_eng}
    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    # temp = ["あいぴょん"]

    # path = pathlib.Path("/datanet/users/muesaka/marmoset/Recorder")  # Marmosetの音声ディレクトリ（/あやぴょん, /あさぴょん, ...）
    path = pathlib.Path("")  # Marmosetの音声ディレクトリ（/あやぴょん, /あさぴょん, ...）
# 

    types = ['UE', 'VPA']
    tag = [3,4,5,6,7,8,9,10,11,12,13,14]
    # lab = ["Phee","Twitter","Others","Trill","Ek","Pr","Tsik"]
    lab = ["Phee","Trill","Twitter","Other Calls"]
    label_color = {"Phee":"red","Trill":"blue","Twitter":"green","Other Calls":"black"}

    # labelとresultを照応し，片方をプロットする．
    # 照応することで，wolabelの部分のothercallsを飛ばすことができる
    # 現状は簡単化のためにフレームでやる

    # ファイルリストの作成
    labelpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test")
    resultpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results_5class_before")
    labellist = list(labelpath.glob("*.txt"))
    resultlist = list(resultpath.glob("*.txt"))
    labellist = sorted(labellist)
    resultlist = sorted(resultlist)

    # 週でforループ
    label_data = []
    result_data =  []
    for i in tag:
        print("---week: {}----------------------------------------".format(i))
        print("")

        # カウント辞書初期化
        dict_label = call_init.copy()
        dict_result = call_init.copy()
        
        # 週iごとの処理：1処理1ファイル
        weeks = []
        for j in range(len(labellist)):

                # パターンマッチング：週iでなければcontinue
                pattern = 'VOC_[0-9]+[-_][0-9]+_+[^_]*_+([^_]*).*'
                tmp = re.findall(pattern ,str(labellist[j]))[0].replace("W","")
                tmp = tmp.split(".")[0]
                if not int(tmp) == i:
                    continue
                
                # 発声ラベルを数え上げ
                ## 確認print
                # print(labellist[j])
                # print(resultlist[j])
                # print("")

                ## no.loadtxt
                labels = np.loadtxt(labellist[j],dtype=int)
                results = np.loadtxt(resultlist[j],dtype=int)

                ## カウント
                for n in range(len(labels)):
                    label = labels[n]
                    result = results[n]

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

        # 週で集計，data.appendする
        label_split = []
        result_split = []
        d1 = {}
        d2 = {}

        label_total = sum(dict_label.values())
        result_total = sum(dict_result.values())

        for n in dict_label:
            d1[n] = dict_label[n]/label_total
            d2[n] = dict_result[n]/result_total

        d1_ratio = list(d1.items())
        d2_ratio = list(d2.items())
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

    # 積み上げグラフ
    ## dataframe作成
    dataset_label = pd.DataFrame(label_data, index=tag, columns=lab)
    dataset_result = pd.DataFrame(result_data, index=tag, columns=lab)
    print(dataset_label)
    print(dataset_result)

    ## Manually Atacched Label
    dataset_label.plot.bar(stacked=True, color=label_color)
    plt.legend(fontsize=8, bbox_to_anchor=(0, 1), loc='lower left', ncol=4, )
    plt.ylabel("Call Ratio")
    plt.xlabel("Week")
    plt.tight_layout()
    plt.savefig("./LabelRatio/test_gt.pdf")
    plt.close()

    ## Estimated Label
    dataset_result.plot.bar(stacked=True, color=label_color)
    plt.legend(fontsize=8, bbox_to_anchor=(0, 1), loc='lower left', ncol=4, )
    plt.ylabel("Call Ratio")
    plt.xlabel("Week")
    plt.tight_layout()
    plt.savefig("./LabelRatio/test_est.pdf")
    plt.close()     