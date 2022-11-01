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

    path = pathlib.Path("/datanet/users/muesaka/marmoset/Recorder")  # Marmosetの音声ディレクトリ（/あやぴょん, /あさぴょん, ...）
    # files = os.listdir(path) 
    # names = [f for f in files if os.path.isdir(os.path.join(path, f))] # names [あやぴょん, あさぴょん, ...]

    tag = ['     [UE]', '     [VPA]']
    for j,names in enumerate([trains+valids+tests, vpas]):
        dict_label = dict() # callの数え上げ用辞書
        for name in names:
            pattern = str(path) + "/" + name + "/*.TextGrid"
            weeks = glob.glob(pattern)

            # 週ごと
            for week in weeks:
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
                          call == "Tsik" or
                          call == "Cough" or
                          call == "Unknown"):
                          call = "Others"

                    dict_label[call] = dict_label.get(call, 0) + 1 # {"Phee":0, "Trill":0, ...}

        print(tag[j])
        total = sum(dict_label.values())
        d={}#空辞書の定義
        for n in dict_label:
            d[n] = dict_label[n] / total#割合の計算
        d_ratio = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for m in d_ratio:
            print(m[0].ljust(10), '{:.3}'.format( round(m[1], 3) ))
        # d = sorted(dict_label.items(), key=lambda i: i[1], reverse=True)
        # pprint.pprint(d_ratio)
        print("")
        