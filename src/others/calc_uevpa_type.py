# -*- coding: utf-8 -*-
#-------------------------------------#
# ueとvpaの遷移確率（train）と，（test）のL2ノルムを計算し類似度とする
# |----- 行ごとのcos類似度を平均したものと類似度とする
# |----- 系列を確率化し，logをとって足し算したものを類似度とする
#-------------------------------------#
import os
import re
import sys
import copy
import glob
import math
import pprint
import pathlib
import textgrid
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Digraph
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

import librosa.display
import seaborn as sns
from matplotlib.colors import Normalize

import numpy as np


# フレーム配列に対し，非発声ラベル0を平滑化する関数
def smooth_labels(labels, thr):
    smoothed_labels = np.copy(labels) # 出力配列
    
    count = 0 # ラベル0の連続数のカウントを初期化
    buf = 0 # ラベル0の平滑化先を初期化
    flag = "Off" # 平滑化のフラグを初期化
    
    # 先頭から順に捜査
    for i in range(len(smoothed_labels)):
        # いまのidxがラベル0である
        if smoothed_labels[i] == 0:
            # ラベル0列の左端にきたときのみ初期化
            if count == 0:
                buf = smoothed_labels[i-1] # ラベル0列の直前に出現したラベル
                count += 1 # ラベル0をカウント
                flag = "On" # 平滑化フラグオン
            # ラベル0のカウントが閾値を超えたら（※ここは＞でなく≧を用いる）
            elif count >= thr:
                flag = "Off" # 平滑化フラグオフ
            # それ以外の条件
            else:
                count += 1 # ラベル0をカウント
        # いまのidxがラベル0以外
        else:
            # 平滑化フラグオフの場合
            if flag == "Off":
                count = 0 # 初期化
                buf = 0 # 初期化
            # 平滑化フラグオンの場合
            else:
                smoothed_labels[i-thr:i] = buf # 直前までのラベル0の列をラベルbufに平滑化
                flag = "Off" # 平滑化フラグオフ
                count = 0 # 初期化
                buf = 0 # 初期化
        
    return smoothed_labels


def del_nocall_data(data):
    data = data[~(data==0)]
    data = data-1

    return data


def calc_transition(ue_unpick, vpa_unpick, pick):
    # ue_unpick = del_nocall_data(ue_unpick)
    # vpa_unpick = del_nocall_data(vpa_unpick)
    pick = del_nocall_data(pick)
    # print(f"pick {pick[0:5]}")
    
    array_ue = ue_unpick[pick[:-1], pick[1:]] # ue_unpickの確率モデルをあてはめたpickの配列
    array_vpa = vpa_unpick[pick[:-1], pick[1:]] # vpa_unpickの確率モデルをあてはめたpickの配列
    # print(f"array_ue {array_ue[0:5]}")
    # print(f"array_vpa {array_vpa[0:5]}")
    
    array_ue_log10 = np.log10(array_ue)
    array_vpa_log10 = np.log10(array_vpa)
    # print(f"array_ue_log10 {array_ue_log10[0:5]}")
    # print(f"array_vpa_log10 {array_vpa_log10[0:5]}")
    # print("")
    
    prod_array_ue = np.sum(array_ue_log10)
    prod_array_vpa = np.sum(array_vpa_log10)

    # prod_array_ue = np.sum(np.log10(array_ue))
    # prod_array_vpa = np.sum(np.log10(array_vpa))
    
    print("ue, vpa\t",prod_array_ue,prod_array_vpa)

    # prod_arrayの尤度がueとvpaのどちらの方が高いか識別
    if prod_array_ue > prod_array_vpa:
        return "UE"
    elif prod_array_ue < prod_array_vpa:
        return "VPA"
    elif prod_array_ue == prod_array_vpa:
        return "Equal" 
    
def compare_matrices(ue_unpick, vpa_unpick, pick):
    # distance_to_ue = np.linalg.norm(pick - ue_unpick)
    # distance_to_vpa = np.linalg.norm(pick - vpa_unpick)

    # if distance_to_ue < distance_to_vpa:
    #     return "UE"
    # elif distance_to_ue > distance_to_vpa:
    #     return "VPA"
    # else:
    #     return "Equal"


    # ue_unpick と vpa_unpick のベクトルの長さを計算
    norm_ue_unpick = np.linalg.norm(ue_unpick, axis=1)
    norm_vpa_unpick = np.linalg.norm(vpa_unpick, axis=1)

    # pick の各行と ue_unpick, vpa_unpick の各行の cos 類似度を計算
    cos_sim_ue = np.dot(pick, ue_unpick.T) / (np.linalg.norm(pick, axis=1)[:, np.newaxis] * norm_ue_unpick)
    cos_sim_vpa = np.dot(pick, vpa_unpick.T) / (np.linalg.norm(pick, axis=1)[:, np.newaxis] * norm_vpa_unpick)

    # pick が ue_unpick に近いか vpa_unpick に近いかを判定
    if np.mean(cos_sim_ue) > np.mean(cos_sim_vpa):
        return "UE"
    elif np.mean(cos_sim_ue) < np.mean(cos_sim_vpa):
        return "VPA"
    else:
        return "Equal" 

def compare_matrices_oneline(ue_unpick, vpa_unpick, pick):
    # ue_unpick と vpa_unpick のベクトルの長さを計算
    norm_ue_unpick = np.linalg.norm(ue_unpick, axis=1)
    norm_vpa_unpick = np.linalg.norm(vpa_unpick, axis=1)

    # pick を1つの行に結合
    pick_combined = pick.flatten().reshape(1, -1)

    # ue_unpick を pick_combined と同じ形状に変形
    ue_unpick_combined = ue_unpick.reshape(-1, pick_combined.shape[1])

    # vpa_unpick を pick_combined と同じ形状に変形
    vpa_unpick_combined = vpa_unpick.reshape(-1, pick_combined.shape[1])

    # ue_unpick_combined と pick_combined の cos 類似度を計算
    cos_sim_ue = np.dot(ue_unpick_combined, pick_combined.T) / (norm_ue_unpick[:, np.newaxis] * np.linalg.norm(pick_combined))

    # vpa_unpick_combined と pick_combined の cos 類似度を計算
    cos_sim_vpa = np.dot(vpa_unpick_combined, pick_combined.T) / (norm_vpa_unpick[:, np.newaxis] * np.linalg.norm(pick_combined))

    # pick_combined が ue_unpick_combined に近いか vpa_unpick_combined に近いかを判定
    if np.mean(cos_sim_ue) > np.mean(cos_sim_vpa):
        return "UE"
    elif np.mean(cos_sim_ue) < np.mean(cos_sim_vpa):
        return "VPA"
    else:
        return "Equal"



    
        
thr = 0

# 遷移確率行列を生成
def tp(transition_probability, label):

    data = transition_probability
    zero = np.zeros((np.max(data)+1,np.max(data)+1)) # （回数カウント用）ゼロ行列を用意
    zero += 1 # （回数カウント用）ゼロ行列に下駄をはかせる．これがないと遷移確率行列の要素に0が混ざってしまうため．

    for i in range(len(data)-1):
        j = copy.deepcopy(i)
        j += 1
        for x, y in itertools.product(range(np.max(data)+1), range(np.max(data)+1)):
            if data[i] == x and data[j] == y:
                zero[x][y] += 1

    row_sum = np.sum(zero, axis=1).reshape((np.max(data)+1,1))
    prob = zero / row_sum

    return prob

def del_nocall_tp(data, node_label):
    node_label.remove("NoCall")
    data = data[~(data==0)]
    data = data-1

    return tp(data, node_label)

if __name__ == "__main__":
    
    # ####################### 2022年の名前 リスト ########################
    # # tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    # tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu"] # UE
    # # vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    # vpas = ["Dior", "Diorella", "Francfranc", "Gabriela"] # VPA
    # call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    # call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    # tag = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # lab = ["Phee", "Trill", "Twitter", "Other Calls"]
    # node_label = ["NoCall", 'Phee', 'Trill', "Twitter", "Other"]

    # ue_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results/" # GroundTruth frame .txt Path
    # vpa_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_vpa_48kHz/test/results/" # GroundTruth frame .txt Path

    # ue_labels = [f for f in os.listdir(ue_labelpath) if os.path.isfile(os.path.join(ue_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    # vpa_labels = [f for f in os.listdir(vpa_labelpath) if os.path.isfile(os.path.join(vpa_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    # results = ue_labels + vpa_labels

    # ######################## 2022データのソート ########################
    # results = sorted(results, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    # results = sorted(results, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])

    

    ####################### 2021年の名前 リスト ########################
    train = ["カルビ","あいぴょん","真央","ブラウニー","花月","黄金","阿伏兎", 
                "テレスドン","スカイドン","三春","会津","マティアス","エバート","ぶた玉","信成"]
    valid = ["鶴ヶ城","ミコノス","イカ玉"]
    test = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎","ひばり","つぐみ","日向夏","八朔","桂島","松島"]
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    
    tag = [9,10,11]
    test_tag = [14]
    
    lab = ["Phee", "Trill", "Twitter", "Other Calls"]
    node_label = ["NoCall", 'Phee', 'Trill', "Twitter", "Other"]

    tests = ["あやぴょん","ビスコッテイー","ドラコ","マルチナ","梨花"]
    vpas = ["高萩","平磯","阿字ヶ浦","馬堀","三崎"]

    ue_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_calc_uevpa_type/ue/" # GroundTruth frame .txt Path
    # ue_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_23ue_muesaka/test/results/"
    
    vpa_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_calc_uevpa_type/vpa/" # GroundTruth frame .txt Path
    # vpa_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/results_5class_before/"

    ue_labels = [f for f in os.listdir(ue_labelpath) if os.path.isfile(os.path.join(ue_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    vpa_labels = [f for f in os.listdir(vpa_labelpath) if os.path.isfile(os.path.join(vpa_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = ue_labels + vpa_labels

    ######################## 2021データのソート ########################
    results = sorted(results, key=lambda s: float(re.findall(r'VOC_.*_.*_(.*)W', s)[0]))
    results = sorted(results, key=lambda s: re.findall(r'VOC_.*_(.*)_.*', s)[0])

    ######################## 交差検証 ########################
    print(f"({len(tests)}*{len(vpas)})*2の2値分類検定を行う (leave-one-out)")
    print("UE:\t", tests)
    print("VPA:\t", vpas)
    print("weeks:\t", tag)
    print("-------------------------------------------------")

    anss = 0
    sums = 0
    for test_individual in tests: # ue個体ごと
        for vpa_individual in vpas: # vpa個体ごと
            for tag_pick in test_tag: # tag（週）ごと．なお学習は週全て使う．
                # ####################### 1週だけでAccを算出 ########################
                # if tag_pick != tag[2]:
                #     continue
                
                print("target_week\t", tag_pick) # テスト個体の対象週

                ####################### unpick個体のリスト作成 ########################
                remaining_tests = copy.deepcopy(tests)
                remaining_tests.remove(test_individual)
                remaining_vpas = copy.deepcopy(vpas)
                remaining_vpas.remove(vpa_individual)

                # print("pick_test:", test_individual)
                # print("pick_vpa:", vpa_individual)
                # print("remaining_tests:", remaining_tests)
                # print("remaining_vpas:", remaining_vpas)
                # print("-------")


                ######################## unpick個体のみでueの遷移表作成 ########################
                data_hstack_ue_unpick = np.empty(0, dtype=int) # ue unpickのラベル列の結合用
                for i, result in enumerate(results):
                    
                    # ######################## 2022データ ########################
                    # pattern = '[^_]*_alone_week([0-9]+).*'
                    # date = re.findall(pattern ,result)[0]
                    # pattern = '([^_]*)_alone_week[0-9]+.*'
                    # name = re.findall(pattern ,result)[0]

                    ####################### 2021データ ########################
                    pattern = 'VOC_.*_.*_(.*)W'
                    date = re.findall(pattern ,result)[0]
                    pattern = 'VOC_.*_(.*)_.*'
                    name = re.findall(pattern ,result)[0]
                    
                    ####################### 特定週でのみ実施 ########################
                    date = float(date)
                    date = math.floor(date)
                    if date not in tag:
                        continue

                    ######################## ue unpickのラベル列を抽出 ########################
                    if name in remaining_tests:
                        data = np.loadtxt(ue_labelpath + result, dtype=int)
                        data = smooth_labels(data, thr)
                        data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                        data_hstack_ue_unpick = np.hstack([data_hstack_ue_unpick, data])
                        
                print("ue_unpick \t",len(data_hstack_ue_unpick))
                prob_ue_unpick = del_nocall_tp(data_hstack_ue_unpick, copy.deepcopy(node_label))

                ######################## unpick個体のみでvpaの遷移表作成 ########################
                data_hstack_vpa_unpick = np.empty(0, dtype=int) # vpa unpickのラベル列の結合用
                for i, result in enumerate(results):
                    
                    
                    # ######################## 2022データ ########################
                    # pattern = '[^_]*_alone_week([0-9]+).*'
                    # date = re.findall(pattern ,result)[0]
                    # pattern = '([^_]*)_alone_week[0-9]+.*'
                    # name = re.findall(pattern ,result)[0]
                    

                    ####################### 2021データ ########################
                    pattern = 'VOC_.*_.*_(.*)W'
                    date = re.findall(pattern ,result)[0]
                    pattern = 'VOC_.*_(.*)_.*'
                    name = re.findall(pattern ,result)[0]
                    
                    ####################### 特定週でのみ実施 ########################
                    date = float(date)
                    date = math.floor(date)
                    if date not in tag:
                        continue

                    ######################## vpa unpickのラベル列を抽出 ########################
                    if name in remaining_vpas:
                        data = np.loadtxt(vpa_labelpath + result, dtype=int)
                        data = smooth_labels(data, thr)
                        data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                        data_hstack_vpa_unpick = np.hstack([data_hstack_vpa_unpick, data])

                print("vpa_unpick \t", len(data_hstack_vpa_unpick))
                prob_vpa_unpick = del_nocall_tp(data_hstack_vpa_unpick, copy.deepcopy(node_label))

                ######################## pick個体のみでueの遷移表作成 ########################
                data_hstack_ue_pick = np.empty(0, dtype=int) # ue unpickのラベル列の結合用
                for i, result in enumerate(results):
                
                    # ######################## 2022データ ########################
                    # pattern = '[^_]*_alone_week([0-9]+).*'
                    # date = re.findall(pattern ,result)[0]
                    # pattern = '([^_]*)_alone_week[0-9]+.*'
                    # name = re.findall(pattern ,result)[0]

                    ####################### 2021データ ########################
                    pattern = 'VOC_.*_.*_(.*)W'
                    date = re.findall(pattern ,result)[0]
                    pattern = 'VOC_.*_(.*)_.*'
                    name = re.findall(pattern ,result)[0]
                    
                    ####################### 特定週でのみ実施 ########################
                    date = float(date)
                    date = math.floor(date)
                    if date not in test_tag:
                        continue
                    
                    ###################### 1週でのみ実施 ########################
                    # if date != tag_pick:
                    #     continue

                    ######################## ue unpickのラベル列を抽出 ########################
                    if name == test_individual:
                        data = np.loadtxt(ue_labelpath + result, dtype=int)
                        data = smooth_labels(data, thr)
                        data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                        data_hstack_ue_pick = np.hstack([data_hstack_ue_pick, data])

                print("ue_pick \t", len(data_hstack_ue_pick))
                # prob_ue_pick = del_nocall_tp(data_hstack_ue_pick, copy.deepcopy(node_label))
                prob_ue_pick = data_hstack_ue_pick

                ######################## pick個体のみでvpaの遷移表作成 ########################
                data_hstack_vpa_pick = np.empty(0, dtype=int) # vpa unpickのラベル列の結合用
                for i, result in enumerate(results):

                    
                    # ######################## 2022データ ########################
                    # pattern = '[^_]*_alone_week([0-9]+).*'
                    # date = re.findall(pattern ,result)[0]
                    # pattern = '([^_]*)_alone_week[0-9]+.*'
                    # name = re.findall(pattern ,result)[0]
                    

                    ####################### 2021データ ########################
                    pattern = 'VOC_.*_.*_(.*)W'
                    date = re.findall(pattern ,result)[0]
                    pattern = 'VOC_.*_(.*)_.*'
                    name = re.findall(pattern ,result)[0]

                    ####################### 特定週でのみ実施 ########################
                    date = float(date)
                    date = math.floor(date)
                    if date not in test_tag:
                        continue
                    
                    ###################### 1週でのみ実施 ########################
                    # if date != tag_pick:
                    #     continue

                    ######################## vpa unpickのラベル列を抽出 ########################
                    if name in vpa_individual:
                        data = np.loadtxt(vpa_labelpath + result, dtype=int)
                        data = smooth_labels(data, thr)
                        data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                        data_hstack_vpa_pick = np.hstack([data_hstack_vpa_pick, data])
                
                print("vpa_pick \t", len(data_hstack_vpa_pick))
                # prob_vpa_pick = del_nocall_tp(data_hstack_vpa_pick, copy.deepcopy(node_label))
                prob_vpa_pick = data_hstack_vpa_pick

                ######################## pick個体（ue, vpa）をUEかVPAか判定：cos類似度 か L2ノルム ########################
                # print(prob_ue_unpick)
                # print(prob_ue_pick)
                # print(prob_vpa_unpick)
                # print(prob_vpa_pick)

                # test_ans = compare_matrices_oneline(prob_ue_unpick, prob_vpa_unpick, prob_ue_pick)
                # vpa_ans = compare_matrices_oneline(prob_ue_unpick, prob_vpa_unpick, prob_vpa_pick)
                
                # test_ans = compare_matrices(prob_ue_unpick, prob_vpa_unpick, prob_ue_pick)
                # vpa_ans = compare_matrices(prob_ue_unpick, prob_vpa_unpick, prob_vpa_pick)
                
                test_ans = calc_transition(prob_ue_unpick, prob_vpa_unpick, prob_ue_pick)
                #----> 学習個体n週分まとめたueとvpa，テスト個体n週分まとめたueを渡す．
                #----> テスト個体はn週のそれぞれで渡したい．ので，変更してみた
                vpa_ans = calc_transition(prob_ue_unpick, prob_vpa_unpick, prob_vpa_pick)

                ######################## pick個体（ue, vpa）をUEかVPAか判定：L2ノルム（ユークリッド距離） ########################
                # print(f"prob_ue_unpick")
                # print(f"{prob_ue_unpick}")
                # print(f"prob_vpa_unpick")
                # print(f"{prob_vpa_unpick}")
                # print("")
                # print(f"prob_ue_pick")
                # print(f"{prob_ue_pick[0:10]}")
                # print(f"prob_vpa_pick")
                # print(f"{prob_vpa_pick[0:10]}")
                # print("")




                if test_ans == "UE":
                    print(f"{test_individual:12s}:UE->{test_ans}")
                    anss += 1
                elif test_ans == "Equal":
                    print(f"{test_individual:12s}:UE->{test_ans}")
                    sums -= 1
                else:
                    print(f"{test_individual:12s}:UE->{test_ans}")

                if vpa_ans == "VPA":
                    print(f"{vpa_individual:12s}:VPA->{vpa_ans}")
                    anss += 1
                elif vpa_ans == "Equal":
                    print(f"{vpa_individual:12s}:VPA->{vpa_ans}")
                    sums -= 1
                else:
                    print(f"{vpa_individual:12s}:VPA->{vpa_ans}")
                    
                sums += 2


                print("-------------------------------------------------")
                

                # break
        # break
    acc = anss/sums
    print("weeks: ", tag)
    print(f"Acc: {acc:.2f} ({anss}/{sums}), weeks: {tag}")
    # sysprt = f"Acc: {acc:.2f}, ({anss}/{sums}), weeks: {tag}"
    os.system(f"end_report 上坂奏人 Acc={acc:.2f}_Nums={anss}/{sums}")
