# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import re
import pprint
import itertools
import math

if __name__ == "__main__":

    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls"} # ラベル番号の辞書
    call_init = {v: 0 for k,v in call_label.items()} # カウント用辞書

    is_plot = 1 #プロットするかどうか

    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test/"
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa/test/results/"
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

        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)
        
        grouped_label = [k for k,g in itertools.groupby(label)] # グルーピング
        grouped_results = [k for k,g in itertools.groupby(results)] # グルーピング

        num_label = call_init.copy()    # カウント用辞書作成
        num_results = call_init.copy()  # カウント用辞書作成

        for k in grouped_label: # labelのカウント
            j = call_label[k]
            num_label[j] = num_label.get(j,0) + 1

        for k in grouped_results:   # resultsのカウント
            j = call_label[k]
            num_results[j] = num_results.get(j,0) + 1
        
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"

        print(file,date,name)

        # list_label.append((name[0], float(date[0]), num_label)) # listにtupleとして追加していく
        # list_results.append((name[0], float(date[0]), num_results)) # listにtupleとして追加していく
        list_label.append((name[0], math.floor(float(date[0])), num_label)) # listにtupleとして追加していく
        list_results.append((name[0], math.floor(float(date[0])), num_results)) # listにtupleとして追加していく        

    list_label.sort(key = lambda x: x[1]) # tupleの2番目の要素＝週でsort
    list_label.sort() # tupleの1番目の要素=名前でsort

    list_results.sort(key = lambda x: x[1]) # tupleの2番目の要素＝週でsort
    list_results.sort() # tupleの1番目の要素=名前でsort

    if is_plot: 

        for fname in vpas:
            for lname in lnames:

                week = np.empty(0,dtype=int)
                count_label = []
                count_results = []

                select_label = list(filter(lambda x: x[0] == fname, list_label))
                select_results = list(filter(lambda x: x[0] == fname, list_results))

                for i in range(len(select_label)):
                    week = np.append(week,select_label[i][1])
                    count_label = np.append(count_label, select_label[i][2][lname])
                    count_results = np.append(count_results, select_results[i][2][lname])
                
                print("week",week)
                print("G.T.",count_label)
                print("Est.",count_results)

                count_diff = np.abs(count_results - count_label)


                plt.bar(week, count_diff, label="abs diff", width=0.2, color='#d62728') #差分

                plt.xlabel("week")
                plt.ylabel("count")
                plt.legend()
                plt.title("VPA_" + fname + "_" + lname)
                week = np.arange(1,15,1)
                plt.xticks(week)
                plt.xlim(0,15)
                plt.ylim(0,190)

                plt.tight_layout()

                dirpath = outputpath + "show_counts/" + lname + "/"
                os.makedirs(dirpath, exist_ok=True)

                filename = dirpath + "show_count_" + lname + "_" +  fname + ".png"
                plt.savefig(filename)
                plt.close()

            #     break
            # break
       
            
