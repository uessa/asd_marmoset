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
import PythonCodeTest

# 混合行列作成

if __name__ == "__main__":


    call_label = {0: "no call", 1: "phee", 2: "trill", 3: "twitter", 4: "other"} # ラベル番号の辞書
    call_init = {v: 0 for k,v in call_label.items()} # カウント用辞書

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results/" # GroundTruth frame .txt Path
    # labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_vpa_48kHz/test/results/" # GroundTruth frame .txt Path
    outputpath = labelpath
    
    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    lnames = ["phee","trill","twitter","other"]
    
    # modeごとにフレームをそのままか回数化するか選ぶ
    plot_mode = "count"
    type_names = tests
    # 平滑化
    thr = 10
    




    files.sort()
    
    list_label = [] # (name,date,dic)のtupleを保存するリスト

    for i, file in enumerate(files):

        # nparray型のフレームラベル
        label = np.loadtxt(labelpath + file,dtype=int)

        # ファイル名から週と名前を抽出
        date = re.findall('[^_]*_alone_week([0-9]+).*',file) # "週"
        name = re.findall('([^_]*)_alone_week[0-9]+.*',file) # "名前"

        filename = name[0] + "_" + date[0]
    

        # フレームの場合
        if plot_mode == "frame":
            grouped_label = label
            
        # 回数の場合
        else:
            label = PythonCodeTest.smooth_labels(label, thr=thr)
            grouped_label = [k for k,g in itertools.groupby(label)] 

        # カウント用辞書
        num_label = call_init.copy()    

        # labelのカウント
        for k in grouped_label:
            j = call_label[k]
            num_label[j] = num_label.get(j,0) + 1
        

        # 確認用出力
        # print(file,date,name)

        # ファイルごとにlistへ(個体名，週，カウント結果)をappend
        list_label.append((name[0], math.floor(float(date[0])), num_label)) # tupleとして追加していく
        # list_label.append((name[0], float(date[0]), num_label)) # listにtupleとして追加していく
        # list_results.append((name[0], float(date[0]), num_results)) # listにtupleとして追加していく

        # break


    is_plot = 1 #プロットするかどうか
    # plt.rcParams["font.family"] = "Times New Roman" # 英論文用フォントファミリー
        
    plt.rc('pdf', fonttype=42) # フォントを埋め込む（Type1，つまり42を指定）

    # listをtupleの要素でsortしておく
    list_label.sort(key = lambda x: x[1]) # 2番目の要素＝週でsort
    list_label.sort() # 1番目の要素=名前でsort

    
    if is_plot: 
        
        
        # for fname in tests:
        for fname in type_names:
        
            # fname = marmo_eng[fname]
            # data = {'Week': week}
            # print(len(week))
            data = {}
            # print(list_label)
            for lname in lnames:
                print(lname, fname)
                
                week = np.empty(0,dtype=int)
                count_label = []

                select_label = list(filter(lambda x: x[0] == fname, list_label))

                for i in range(len(select_label)):
                    week = np.append(week,select_label[i][1])
                    count_label.append(select_label[i][2][lname])
                
                # csv 
                data['week'] = week
                # print(len(week))
                data[lname] = count_label
                # print(len(count_label))
                
                
                # plot1
                # 棒グラフ（隣接させた）を，時系列で並べる
                plt.bar(week-0.2/2, count_label, label="Ground Truth", width=0.2) #隣接左側に正解

                
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
            plt.figure(figsize=(10,5))
            colors = ['b', 'g', 'r', 'black']  # ペアごとに使用する色を指定
            # print(df)
            # print("df.columns",len(df.columns))
            for i in range(1, len(df.columns)):
                col_gt = df.columns[i]
                plt.plot(df['week'], df[col_gt], color=colors[i-1], linestyle='-', label=f'{col_gt}')

            plt.legend(loc="upper right")
            plt.xlabel("Week")
            plt.xticks(fontsize=10, rotation=45)
            plt.xticks(df['week'])
            plt.grid(color='lightgray')
            
            if plot_mode == "frame":
                plt.ylabel("Count of Frame")
                plt.title(f"Distribution of Call Classification Frames ({fname})")
                df_pdf_filename = dirpath + "week_frame_" + fname + ".pdf"            
                csv_filename = dirpath + "week_frame_" + fname + ".csv"
                df.to_csv(csv_filename, index=False)
                
            elif plot_mode == "count":
                plt.ylabel("Count of Segment")
                plt.title(f"Distribution of Call Classification Segments ({fname})")
                df_pdf_filename = dirpath + "week_count_" + fname + ".pdf"
                
                csv_filename = dirpath + "week_count_" + fname + ".csv"
                df.to_csv(csv_filename, index=False)
            
            # pdfファイルに保存
            plt.savefig(df_pdf_filename, format="pdf")
            print("save:", df_pdf_filename)

            # グラフを表示せずにメモリを解放（必要に応じて）
            plt.close()

            
                # break
            # break