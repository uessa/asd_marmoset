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
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
# from sklearn.linear_model import TweedieRegressor

# 混合行列作成

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

    labelpath1 = "/home/muesaka/projects/marmoset/raw/marmoset_23ue_text/5class_label/" # GroundTruth frame .txt Path
    resultpath1 = "/home/muesaka/projects/marmoset/raw/marmoset_23ue_text/5class_label/" # Estimate frame .txt Path
    labelpath2 = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/label_5class/" # GroundTruth frame .txt Path
    resultpath2 = "/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/label_5class/" # Estimate frame .txt Path
    outputpath = labelpath1

    files1 = [f for f in os.listdir(labelpath1) if os.path.isfile(os.path.join(labelpath1, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results1 = [f for f in os.listdir(resultpath1) if os.path.isfile(os.path.join(resultpath1, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    files2 = [f for f in os.listdir(labelpath2) if os.path.isfile(os.path.join(labelpath2, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results2 = [f for f in os.listdir(resultpath2) if os.path.isfile(os.path.join(resultpath2, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["Phee","Trill","Twitter","Other Calls"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]

    files1 = sorted(files1, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results1 = sorted(results1, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    files1 = sorted(files1, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results1 = sorted(results1, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    files2 = sorted(files2, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results2 = sorted(results2, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    files2 = sorted(files2, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results2 = sorted(results2, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    
    list_label1 = [] # (name,date,dic)のtupleを保存するリスト
    list_results1 = [] # (name,date,dic)のtupleを保存するリスト
    list_label2 = [] # (name,date,dic)のtupleを保存するリスト
    list_results2 = [] # (name,date,dic)のtupleを保存するリスト

    # print(files1)

    # ue
    for i, file in enumerate(files1):

        # nparray型のフレームラベル
        label = np.loadtxt(labelpath1 + file, dtype=int)
        results = np.loadtxt(resultpath1 + file, dtype=int)

        # ファイル名から週と名前を抽出
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"
        
        # フレームのラベルをgroupingで連続同値を順列にまとめて単純化
        grouped_label = [k for k,g in itertools.groupby(label)] 
        grouped_results = [k for k,g in itertools.groupby(results)] 
        # grouped_label = label
        # grouped_results = results

        # カウント用辞書
        num_label = call_init.copy()    
        num_results = call_init.copy()  

        # labelのカウント
        for k in grouped_label:
            if k >= 5:
                continue
            j = call_label[k]
            num_label[j] = num_label.get(j,0) + 1

        # resultsのカウント
        for k in grouped_results:
            if k >= 5:
                continue  
            j = call_label[k]
            num_results[j] = num_results.get(j,0) + 1
        
        # 確認用出力
        print(file, date, name)
        # print("label", num_label)
        # print("results", num_results, "\n")

        # ファイルごとにlistへtuple(個体名，週，カウント結果，モデル)をappend
        list_label1.append((name[0], math.floor(float(date[0])), num_label, "UE")) # tupleとして追加していく
        list_results1.append((name[0], math.floor(float(date[0])), num_results, "UE")) # tupleとして追加していく

        # if i == 10:

        #     break 

    # vpa
    for i, file in enumerate(files2):

        # nparray型のフレームラベル
        label = np.loadtxt(labelpath2 + file, dtype=int)
        results = np.loadtxt(resultpath2 + file, dtype=int)

        # ファイル名から週と名前を抽出
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"
        
        # フレームのラベルをgroupingで連続同値を順列にまとめて単純化
        grouped_label = [k for k,g in itertools.groupby(label)] 
        grouped_results = [k for k,g in itertools.groupby(results)] 
        # grouped_label = label
        # grouped_results = results

        # カウント用辞書
        num_label = call_init.copy()    
        num_results = call_init.copy()  

        # labelのカウント
        for k in grouped_label:
            if k >= 5:
                continue
            j = call_label[k]
            num_label[j] = num_label.get(j,0) + 1

        # resultsのカウント
        for k in grouped_results:
            if k >= 5:
                continue  
            j = call_label[k]
            num_results[j] = num_results.get(j,0) + 1
        
        # 確認用出力
        print(file, date, name)
        print("label", num_label)
        print("results", num_results, "\n")

        # ファイルごとにlistへtuple(個体名，週，カウント結果，モデル)をappend
        list_label2.append((name[0], math.floor(float(date[0])), num_label, "VPA")) # tupleとして追加していく
        list_results2.append((name[0], math.floor(float(date[0])), num_results, "VPA")) # tupleとして追加していく

        # if i == 10:

        #     break   
        
                

    # プロット
    is_plot = 1 #プロットするかどうか
    if is_plot: 
        # fig, (ax1, ax2) = plt.subplots(1,2)
        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        label_name = "Phee"

        # データ列作成
        # UE's label and result
        week = np.empty(0,dtype=int)
        count_label = []
        count_results = []

        for i in range(len(list_label1)):
            week = np.append(week, list_label1[i][1])
            count_label.append(list_label1[i][2][label_name])
            count_results.append(list_results1[i][2][label_name])

        # プロット
        # plt.subplot(1, 2, 1)
        ax1.scatter(week, count_label, label="UE", marker=".") # ue gt
        # ax1.plot(week, np.poly1d(np.polyfit(week, count_label, 2))(week))

        ax2.scatter(week, count_results, label="UE", marker=".") # ue est
        # ax1.plot(week, np.poly1d(np.polyfit(week, count_results, 2))(week))


        # glm = PoissonRegressor()  # デフォルト
        glm = PoissonRegressor(
                                alpha=0,  # ペナルティ項
                                fit_intercept=True,  # 切片
                                max_iter=300,  # ソルバーの試行回数
                            )
        X = week
        X = X.reshape(len(X), 1)
        pprint.pprint(X.shape)
        y = np.array(count_label)
        # y = y.reshape(len(y), 1)
        pprint.pprint(y.shape)
        glm.fit(X, y)
        # print("score:", glm.score(X_test, y_test))

        y_hat = glm.predict(np.arange(np.max(X)).reshape(np.max(X), 1) + 1 )
        ax1.plot(np.arange(np.max(X)) + 1, y_hat)

        # print(y_hat)

        # 可視化
        # fig = plt.figure(figsize=(6.0, 6.0))
        # plt.plot(X, y, "o")
        # plt.plot(X, y_hat, "*", color="r")
        # plt.xlabel('x (total_bill)'), plt.ylabel('y (tips)')
        # plt.xlim(0, 60), plt.ylim(0, 12)
        # plt.show()


        # df = pd.DataFrame([week, count_label], index=["week", "count"]).T
        # sns.lineplot(x="week", y="count", data=df, ax=ax1)

        # df = pd.DataFrame([week, count_results], index=["week", "count"]).T
        # sns.lineplot(x="week", y="count", data=df, ax=ax2)







        # VPA's label and results
        week = np.empty(0,dtype=int)
        count_label = []
        count_results = []
        for i in range(len(list_label2)):
            week = np.append(week, list_label2[i][1])
            count_label.append(list_label2[i][2][label_name])
            count_results.append(list_results2[i][2][label_name])
        
        # プロット
        # plt.subplot(1, 2, 1)
        ax1.scatter(week, count_label, label="VPA", marker="x") # vpa gt

        # ax1.plot(week, np.poly1d(np.polyfit(week, np.array(count_label), 2))(week))

        ax1.set_ylabel("Counts (call)")
        ax1.set_xlabel("Weeks")
        ax1.set_title("Ground Truth")
        ax1.legend(loc="upper right")

        # ax2.subplot(1, 2, 2)
        ax2.scatter(week, count_results, label="VPA", marker="x") # vpa est
        # ax2.plot(week, np.poly1d(np.polyfit(week, count_results, 2))(week))
        ax2.set_ylabel("Counts (call)")
        ax2.set_xlabel("Weeks")
        ax2.set_title("Estimation")
        ax2.legend(loc="upper right")

        # df = pd.DataFrame([week, count_label], index=["week", "count"]).T
        # sns.lineplot(x="week", y="count", data=df, ax=ax1)

        # df = pd.DataFrame([week, count_results], index=["week", "count"]).T
        # sns.lineplot(x="week", y="count", data=df, ax=ax2)

        X = week
        X = X.reshape(len(X), 1)
        pprint.pprint(X.shape)
        y = np.array(count_label)
        # y = y.reshape(len(y), 1)
        pprint.pprint(y.shape)
        glm.fit(X, y)
        # print("score:", glm.score(X_test, y_test))

        y_hat = glm.predict(np.arange(np.max(X)).reshape(np.max(X), 1) + 1 )
        ax1.plot(np.arange(np.max(X)) + 1, y_hat)
        
        # 保存
        fig.suptitle(label_name)
        plt.savefig(outputpath + "count_call_" + label_name + ".pdf")




        