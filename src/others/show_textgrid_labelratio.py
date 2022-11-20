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
    trains = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valids = ["鶴ヶ城","ミコノス","イカ玉"]
    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Ek':0, 'Pr':0, 'Tsik':0, 'Others':0}
    temp = ["あいぴょん"]

    path = pathlib.Path("/datanet/users/muesaka/marmoset/Recorder")  # Marmosetの音声ディレクトリ（/あやぴょん, /あさぴょん, ...）

    # tag = ['UE', 'VPA']
    tag = [3,4,5,6,7,8,9,10,11,12,13,14]
    # lab = ["Phee","Twitter","Others","Trill","Ek","Pr","Tsik"]
    lab = ["Phee","Trill","Twitter","Ek","Pr","Tsik","Others"]

    # UE，VPAのタグそれぞれで処理
    # for j,names in enumerate([tests+valids+trains, vpas]):
    for j,names in enumerate([temp]):

        print(tag[j])

        # 個体名でforループ
        for name in names:

            data = []
            
            pattern = str(path) + "/" + name + "/*.TextGrid"
            weeks = glob.glob(pattern)

            # パターンマッチング 週抽出と並べ替え
            for i,week in enumerate(weeks):
                pattern = 'VOC_[0-9]+[-_][0-9]+_+[^_]*_+([^_]*).*'
                weeks[i]= [week, re.findall(pattern ,week)[0].replace("W","")]
            weeks = sorted(weeks, key=lambda x: int(x[1]))
            week_tmp = []
            for num in range(len(weeks)):
                week_tmp.append(weeks[num][0])
            weeks = week_tmp
            

            # 週ごとにforループ
            lenweek = tag
            for i in lenweek:

                # callの数え上げ用辞書
                dict_label = call_init.copy()

                for week in weeks:
                    pattern = 'VOC_[0-9]+[-_][0-9]+_+[^_]*_+([^_]*).*'
                    j = re.findall(pattern ,week)[0].replace("W","")

                    if str(i) != j:
                        continue
                    print(i,j)
                    print(week)
                    text = textgrid.TextGrid.fromFile(week) # text = [0:鳴き声, 1:ドア音][interval]    
                    # TextGridのIntervalごと
                    for k in range(len(text[0])):
                        call = text[0][k].mark # Phee, Trill, ...
                        call = call.replace(' ', '') # " Unknown" -> "Unknown"
                        call = call.upper() # phee-trill -> PHEE-TRILL
                        call = call.title() # PHEE-TRILL -> Phee-Trill

                        # 空のIntervalはcontinue
                        if call == "":
                            continue

                        # 辞書で修正
                        call = correct_label[call]

                        # いらないものならcontinue
                        if (call == "Breath" or
                            call == "Chirp" or
                            call == "Chatter"):
                            continue

                        elif (call == "Phee-Trill" or
                            call == "Trill-Phee"):
                            call = "Pr"

                        elif (call == "Cry" or
                            call == "Cough" or
                            call == "Unknown"):
                            call = "Others"

                        dict_label[call] = dict_label.get(call, 0) + 1 # {"Phee":0, "Trill":0, ...}
                # 個体ごとに一度集計
                data_split = []
                total = sum(dict_label.values())
                d={} #空辞書の定義
                count = 0
                for n in dict_label:
                    if dict_label[n] == 0:
                        d[n] = 0
                    else:
                        d[n] = dict_label[n] / total #割合の計算
                        count += dict_label[n]
                # d_ratio = sorted(d.items(), key=lambda x: x[0], reverse=True)
                d_ratio = list(d.items())
                for m in d_ratio:
                    print(m[0].ljust(10), '{}'.format(m[1]))
                    data_split.append(m[1])
                print("count=",count)
                print("")
                print(len(data_split))
                data.append(data_split)
        
        # 集計
        # total = sum(dict_label.values())
        # d={} #空辞書の定義
        # count = 0
        # for n in dict_label:
        #     d[n] = dict_label[n] / total #割合の計算
        #     count += dict_label[n]
        # # d_ratio = sorted(d.items(), key=lambda x: x[1], reverse=True)
        # d_ratio = list(d.items())
        # for m in d_ratio:
        #     print(m[0].ljust(10), '{}'.format(m[1]))
        #     data.append(m[1])
        # print("count=",count)
        # print("")


            # 積み上げ棒グラフ
            dataset = pd.DataFrame(data, 
                            index=tag, 
                            columns=lab)
            print(dataset)
            print("")
            dataset.plot.bar(stacked=True)
            plt.legend() 
            plt.savefig("./LabelRatio/LabelRatioWeeks_{}.pdf".format(name))
            plt.close()
    