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
def make_confusion_matrix(labels, results, classes, output_dir, filename):
    cm_label = ["No Call", "Phee", "Trill", "Twitter", "Other Calls"]
    cm = confusion_matrix(labels, results, classes)
    # cm = pd.DataFrame(data=cm, index=cm_label, columns=cm_label)
    print(cm)

    # cm_label_estimate = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 
    #                        'Trill-Phee', 'Tsik', 'Ek', 'Ek-Tsik', 'Cough', 'Cry', 'Chatter', 'Breath', 'Unknown']
    # cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Unknown', 'Other Calls']

    # 行毎に確率値を出して色分け
    cm_prob = cm / np.sum(cm, axis=1, keepdims=True)

    # cm = cm[:, :5]
    # cm = cm.T
    # cm_prob = cm_prob[:, :5]
    # cm_prob = cm_prob.T

    # 2クラス分類：font=25,annot_kws35, 12クラス分類：font=15,annot_kws10, 5クラス分類：font=15,annot_kws20, cbar=False
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 15
    sns.heatmap(
        cm_prob,
        annot=cm,
        cmap="GnBu",
        xticklabels=cm_label,
        yticklabels=cm_label,
        fmt=".10g",
        center=0
        # square=True,
    )
    # sns.heatmap(cm, center=0)
    plt.xlabel("Estimated Label")
    plt.ylabel("Ground Truth Label")
    plt.yticks(rotation=90,rotation_mode="anchor",ha="center",va="baseline")
    plt.ylim(5, 0)
    plt.title(filename)
    plt.tight_layout()

    dirpath = output_dir + "confusion_matrix/"
    os.makedirs(dirpath, exist_ok=True)
    filename = dirpath + "ConfMat_" +filename + ".pdf"
    fig.savefig(filename)
    plt.close()

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

    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls"} # ラベル番号の辞書
    call_init = {v: 0 for k,v in call_label.items()} # カウント用辞書

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_check_othercalls/test/" # GroundTruth frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_check_othercalls/test//results/" # Estimate frame .txt Path
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
        filename = marmo_eng[name[0]] + "_" + date[0]
        make_confusion_matrix(label, results, [0,1,2,3,4], labelpath, filename)
        
        # フレームのラベルをgroupingで連続同値を順列にまとめて単純化
        grouped_label = [k for k,g in itertools.groupby(label)] 
        grouped_results = [k for k,g in itertools.groupby(results)] 

        # カウント用辞書
        num_label = call_init.copy()    
        num_results = call_init.copy()  

        # labelのカウント
        for k in grouped_label:
            j = call_label[k]
            num_label[j] = num_label.get(j,0) + 1

        # resultsのカウント
        for k in grouped_results:   
            j = call_label[k]
            num_results[j] = num_results.get(j,0) + 1
        
        # 確認用出力
        print(file,date,name)

        # ファイルごとにlistへ(個体名，週，カウント結果)をappend
        list_label.append((name[0], math.floor(float(date[0])), num_label)) # tupleとして追加していく
        list_results.append((name[0], math.floor(float(date[0])), num_results)) # tupleとして追加していく        
        # list_label.append((name[0], float(date[0]), num_label)) # listにtupleとして追加していく
        # list_results.append((name[0], float(date[0]), num_results)) # listにtupleとして追加していく

        break


    is_plot = 0 #プロットするかどうか

    # listをtupleの要素でsortしておく
    list_label.sort(key = lambda x: x[1]) # 2番目の要素＝週でsort
    list_label.sort() # 1番目の要素=名前でsort
    list_results.sort(key = lambda x: x[1]) # 2番目の要素＝週でsort
    list_results.sort() # 1番目の要素=名前でsort
    
    if is_plot: 
        
        # 個体名ごと
        for lname in lnames:
        
            # ラベル名ごと
            for fname in vpas:

                week = np.empty(0,dtype=int)
                count_label = []
                count_results = []

                select_label = list(filter(lambda x: x[0] == fname, list_label))
                select_results = list(filter(lambda x: x[0] == fname, list_results))

                for i in range(len(select_label)):
                    week = np.append(week,select_label[i][1])
                    count_label.append(select_label[i][2][lname])
                    count_results.append(select_results[i][2][lname])
                
                # print("week",week)
                # print("G.T.",count_label)
                # print("Est.",count_results)

                # plot1
                # 棒グラフ（隣接させた）を，時系列で並べる
                plt.bar(week-0.2/2, count_label, label="Ground Truth", width=0.2) #隣接左側に正解
                plt.bar(week+0.2/2, count_results,label="Estimated", width=0.2) #隣接右側にに推定

                fname = marmo_eng[fname]
                plt.xlabel("Week")
                plt.ylabel("Count")
                plt.legend()
                plt.title("UE_" + fname + "_" + lname)
                week = np.arange(1,15,1)
                plt.xticks(week)
                plt.xlim(0,15)
                plt.tight_layout()

                # ディレクトリ作成 show_counts/[個体名]/
                dirpath = outputpath + "show_counts/" + fname + "/"
                os.makedirs(dirpath, exist_ok=True)

                # ファイル作成 /show_count_[個体名]_[ラベル名].pdf
                filename = dirpath + "show_count_" + fname + "_" +  lname + ".pdf"
                plt.savefig(filename)
                plt.close()

            #     break
            # break
       
            
