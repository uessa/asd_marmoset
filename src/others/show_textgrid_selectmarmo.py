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

if __name__ == "__main__":
    path = os.path.join("/datanet/users/muesaka/Marmoset/Recorder")  # Marmosetの音声ディレクトリ（/あやぴょん/, /あさぴょん/, ...）
    vpa_names = ["三崎","馬掘"]  # ASD個体
    ue_names = ["ぶた玉","イカ玉"]  # 通常個体
    num_week = 13  # 使用する週の合計数
    cnt_ue = 0
    cnt_vpa = 0
    col_ue = ["tomato","orangered","indianred","firebrick","darkred"]
    col_vpa = ["darkgreen","darkseagreen","limegreen","olivedrab","lightsage"]
    label_names = ["Trill","Phee","Trill-Phee","Phee-Trill","Twitter","Tsik","Cry","Ek","Cough","Unknown"]

    in_names = vpa_names + ue_names # 対象の個体指定
    print("num_week", num_week)  
    print("in_names",in_names)

    # label_names[]ごとにin_names[]についてプロット
    for label_name in label_names:

        cnt_ue = 0
        cnt_vpa = 0

        for name in in_names:

            # .TextGridファイルのpathをListに
            pattern = path + name + "/*.TextGrid"
            files = glob.glob(pattern)
            files.sort()

            # .TextGridファイルをopen
            list_tg = []
            for i in range(num_week):
                tg = textgrid.TextGrid.fromFile(files[i])
                list_tg.append(tg)  # list_tg[週][0:鳴き声, 1:ドア音][interval]

            # 全ラベルのカウント
            num_one_label = []
            week_one_label = []
            for n in range(num_week):
                num_intervals = len(list_tg[n][0])  # interval数
                dict_label = dict()
                for i in range(num_intervals):
                    k = list_tg[n][0][i].mark  # ラベル（TrillとかPheeとか）
                    k = k.strip()  # 空白削除
                    k = k.upper()  # 全大文字
                    k = k.title()  # タイトルケース（Phee-trill -> Phee-Trill, TRILL -> Trill）
                    if k == "":
                        continue
                    dict_label[k] = dict_label.get(k, 0) + 1  # dict型のdict_labelにラベルを登録＆カウント

                #        print(dict_label)
                # 単一ラベルのカウントをリストに
                num_one_label.append(dict_label.get(label_name, 0))  # [1,5,2,...]などを期待
                week_one_label.append((n+1) + 1)


            print(name, label_name, num_one_label)  # 個体名 ラベル名 [a1,a2,...,an]

            if name in ue_names: 
                plt.plot(week_one_label, num_one_label, label="UE:" + name, color=col_ue[cnt_ue])
                cnt_ue = cnt_ue + 1
            else:
                plt.plot(week_one_label, num_one_label, label="VPA:"+ name, color=col_vpa[cnt_vpa])
                cnt_vpa = cnt_vpa + 1

        plt.xlabel("Week")
        plt.ylabel("Count")

        title = label_name
        plt.title(title)
        plt.xlim(1,14)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        select_name = str(cnt_ue) + "ue" + str(cnt_vpa) + "vpa"
        dirpath = "./output/one_label/" + select_name + "/"
        os.makedirs(dirpath, exist_ok=True)

        filename = dirpath + label_name + "_" +  select_name + ".png"
        plt.savefig(filename)
        plt.close()
