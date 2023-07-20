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

    call_label = {0: "no call", 1: "phee", 2: "trill", 3: "twitter", 4: "other"} # ラベル番号の辞書
    call_init = {v: 0 for k,v in call_label.items()} # カウント用辞書

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/" # GroundTruth frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results/" # Estimate frame .txt Path
    outputpath = labelpath

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    lnames = ["phee","trill","twitter","other"]
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
        # make_confusion_matrix(label, results, [0,1,2,3,4], labelpath, filename)
        
        # modeごとにフレームをそのままか回数化するか選ぶ
        plot_mode = "frame"
        
        if plot_mode == "frame":
            grouped_label = label
            grouped_results = results
        else:
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
        
        # label and resultsのカウント
        # for n in range(len(grouped_label)):
        #     print(len(grouped_label))
        #     print(len(grouped_results))
        #     k = grouped_label[n]
        #     # print(grouped_label[n])
        #     # print(grouped_results[n])
        #     l = grouped_results[n]
        #     if k == 5 or k == 6 or k == 7:
        #         continue
        #     j1 = call_label[k]
        #     j2 = call_label[l]
        #     num_label[j1] = num_label.get(j1,0) + 1
        #     num_results[j2] = num_results.get(j2,0) + 1


        # 確認用出力
        # print(file,date,name)

        # ファイルごとにlistへ(個体名，週，カウント結果)をappend
        list_label.append((name[0], math.floor(float(date[0])), num_label)) # tupleとして追加していく
        list_results.append((name[0], math.floor(float(date[0])), num_results)) # tupleとして追加していく        
        # list_label.append((name[0], float(date[0]), num_label)) # listにtupleとして追加していく
        # list_results.append((name[0], float(date[0]), num_results)) # listにtupleとして追加していく

        # break


    is_plot = 1 #プロットするかどうか
    # plt.rcParams["font.family"] = "Times New Roman" # 英論文用フォントファミリー
        
    plt.rc('pdf', fonttype=42) # フォントを埋め込む（Type1，つまり42を指定）

    # listをtupleの要素でsortしておく
    list_label.sort(key = lambda x: x[1]) # 2番目の要素＝週でsort
    list_label.sort() # 1番目の要素=名前でsort
    list_results.sort(key = lambda x: x[1]) # 2番目の要素＝週でsort
    list_results.sort() # 1番目の要素=名前でsort
    
    if is_plot: 
        
        
        for fname in tests:
        
            # fname = marmo_eng[fname]
            # data = {'Week': week}
            # print(len(week))
            data = {}
            # print(list_label)
            for lname in lnames:
                print(lname, fname)
                
                week = np.empty(0,dtype=int)
                count_label = []
                count_results = []

                select_label = list(filter(lambda x: x[0] == fname, list_label))
                select_results = list(filter(lambda x: x[0] == fname, list_results))

                print(len(select_label))
                for i in range(len(select_label)):
                    week = np.append(week,select_label[i][1])
                    count_label.append(select_label[i][2][lname])
                    count_results.append(select_results[i][2][lname])
                
                # csv 
                data['week'] = week
                print(len(week))
                data[lname + "_annot."] = count_label
                print(len(count_label))
                data[lname + "_est."] = count_results
                print(len(count_label))
                
                
                # plot1
                # 棒グラフ（隣接させた）を，時系列で並べる
                plt.bar(week-0.2/2, count_label, label="Ground Truth", width=0.2) #隣接左側に正解
                plt.bar(week+0.2/2, count_results,label="Estimated", width=0.2) #隣接右側にに推定

                
                plt.xlabel("Week")
                plt.ylabel("Count")
                # plt.legend()
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
                # plt.savefig(filename)
                # print("save: {}".format(filename))
                # plt.close()
            
            # csv
            # print(data)
            df = pd.DataFrame(data)
            # print(df)

        
            
            
            # グラフを作成
            plt.figure()
            colors = ['b', 'g', 'r', 'c']  # ペアごとに使用する色を指定
            for i in range(1, len(df.columns), 2):
                col_gt = df.columns[i]
                col_est = df.columns[i + 1]
                plt.plot(df['week'], df[col_gt], color=colors[i//2], linestyle='-', label=f'{col_gt}')
                plt.plot(df['week'], df[col_est], color=colors[i//2], linestyle='--', label=f'{col_est}')

            plt.legend(loc="upper right")
            plt.xlabel("Week")
            plt.xticks(df['week'])
            
            if plot_mode == "frame":
                plt.ylabel("Count of Frame")
                plt.title(f"Distribution of Call Classification Frames ({fname})")
                df_pdf_filename = dirpath + "week_frame_" + fname + ".pdf"            
                csv_filename = dirpath + "week_frame_" + fname + ".csv"
                df.to_csv(csv_filename, index=False)
                
            elif plot_mode == "count":
                plt.ylabel("Count of Count")
                plt.title(f"Distribution of Call Classification Counts ({fname})")
                df_pdf_filename = dirpath + "week_count_" + fname + ".pdf"
                
                csv_filename = dirpath + "week_count_" + fname + ".csv"
                df.to_csv(csv_filename, index=False)
            
            # pdfファイルに保存
            plt.savefig(df_pdf_filename, format="pdf")

            # グラフを表示せずにメモリを解放（必要に応じて）
            plt.close()

            
                # break
            # break