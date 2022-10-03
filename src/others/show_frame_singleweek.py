# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
import japanize_matplotlib
import re
from matplotlib.backends.backend_pdf import PdfPages

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
    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls"}

    is_plot = 1 # if plot on/off
    pdf = PdfPages('23ue_test_5ue.pdf') # filename
    labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue/test_tested/" # GroundTruth frame .txt Path
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue/test_tested/results/" # Estimate frame .txt Path

    files = [f for f in os.listdir(labelpath) if os.path.isfile(os.path.join(labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    files = sorted(files, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results = sorted(results, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    files = sorted(files, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])
    results = sorted(results, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    for i, file in enumerate(files):
        label = np.loadtxt(labelpath + file,dtype=int)
        results = np.loadtxt(resultpath + file,dtype=int)

        total = len(label)
        corr = np.sum(label == results)
        acc = np.round(corr / total * 100, 1)
        date = re.findall('VOC_.*_.*_(.*)W',file) # "週"
        name = re.findall('VOC_.*_(.*)_.*',file) # "名前"
        print(
            "Correct/Total (Acc): {}/{} ({}%)".format(corr, total, acc),
            name[0],
            date[0],
        )

        # if date[0] != "ビスコッテイー_3W":
        #     continue

        if is_plot: #is_plot = 1でプロット

            plt.figure(figsize=(50,3),dpi=400)
            fftlen = 2048 # frame length
            fs = 96000 # sampling rate
            stime = fftlen / 2 / fs # frame -> sec translated

            label_sec = np.arange(len(label)) * stime
            results_sec = np.arange(len(results)) * stime

            x_min = 0
            x_max = len(label) * stime # max_sec

            lw = 0.5 # line width
            title = "{} (weeks={}, acc={}%)".format(marmo_eng[name[0]], date[0], acc) # hassaku (weeks=11, acc=95.1%)

            plt.subplot(2, 1, 1)
            plt.plot(label_sec, label == 1, "b",label="1:"+call_label[1],lw=lw)
            plt.plot(label_sec, label == 2, "g",label="2:"+call_label[2],lw=lw)
            plt.plot(label_sec, label == 3, "r",label="3:"+call_label[3],lw=lw)
            plt.plot(label_sec, label == 4, "c",label="4:"+call_label[4],lw=lw)
            plt.xlim([x_min,x_max])
            plt.yticks([0,1],["NoCall","Call"])
            plt.title(title)
            plt.ylabel("G.T.")

            plt.subplot(2, 1, 2)
            plt.plot(results_sec, results == 1, "b",label="1:"+call_label[1],lw=lw)
            plt.plot(results_sec, results == 2, "g",label="2:"+call_label[2],lw=lw)
            plt.plot(results_sec, results == 3, "r",label="3:"+call_label[3],lw=lw)
            plt.plot(results_sec, results == 4, "c",label="4:"+call_label[4],lw=lw)
            plt.xlim([x_min,x_max])
            plt.yticks([0,1],["NoCall","Call"])
            plt.ylabel("Est.")
            plt.legend(bbox_to_anchor=(1, -.2), loc='upper right', ncol=4)
            

            plt.xlabel("Time (sec) ")
            plt.tight_layout()

            pdf.savefig()
            # dirpath = labelpath + "show_frames/"
            # os.makedirs(dirpath, exist_ok=True)

            # filename1 = dirpath + "show_frame_" + date[0] + "_sec_" + str(x_min) + "-" + str(int(x_max))
            # # filename1 = dirpath + "show_frame_" + date[0]
            # filename2 = filename1 + "_" + str(acc) + "%" + ".png"
            # plt.savefig(filename2)
            plt.close()
            

    pdf.close()

