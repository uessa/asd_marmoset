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

def del_nocall_data(data):
    data = data[~(data==0)]
    data = data-1

    return data

def calc_transition(ue_unpick, vpa_unpick, pick):
    # ue_unpick = del_nocall_data(ue_unpick)
    # vpa_unpick = del_nocall_data(vpa_unpick)
    pick = del_nocall_data(pick)
    array_ue = ue_unpick[pick[:-1], pick[1:]] # ue_unpickの確率モデルをあてはめたpickの配列
    array_vpa = vpa_unpick[pick[:-1], pick[1:]] # vpa_unpickの確率モデルをあてはめたpickの配列

    prod_array_ue = np.sum(np.log10(array_ue))
    prod_array_vpa = np.sum(np.log10(array_vpa))

    # print(f"pick {pick[0:10]}")
    # print(f"array_ue {array_ue[0:10]}")
    # print(f"array_vpa {array_vpa[0:10]}")
    # print("")
    # print(prod_array_ue)
    # print(prod_array_vpa)
    # print("")

    # pick が ue_unpick に近いか vpa_unpick に近いかを判定
    if prod_array_ue > prod_array_vpa:
        return "UE"
    elif prod_array_ue < prod_array_vpa:
        return "VPA"
    else:
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



    
        
target = "2022_vpa"

def tp(transition_probability, label):

    data = transition_probability
    zero = np.zeros((np.max(data)+1,np.max(data)+1))

    for i in range(len(data)-1):
        j = copy.deepcopy(i)
        j += 1
        for x, y in itertools.product(range(np.max(data)+1), range(np.max(data)+1)):
            if data[i] == x and data[j] == y:
                zero[x][y] += 1

    row_sum = np.sum(zero, axis=1).reshape((np.max(data)+1,1))
    prob    = zero / row_sum
    # print(prob)

    return prob

def del_nocall_tp(data, node_label):
    node_label.remove("NoCall")
    data = data[~(data==0)]
    data = data-1

    return tp(data, node_label)

if __name__ == "__main__":
    ######################## 2022年の名前 リスト ########################
    tests = ["Falco", "Haiji", "Kenshiro", "Kusukusu", "Shirushiru"] # UE
    vpas = ["Dior", "Diorella", "Francfranc", "Gabriela", "Galileo", "Marimo", "Sango"] # VPA
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    tag = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    lab = ["Phee", "Trill", "Twitter", "Other Calls"]
    node_label = ["NoCall", 'Phee', 'Trill', "Twitter", "Other"]
    name_sets = [[tests[0],tests[1],tests[2]], 
                 [tests[3],tests[4]],
                 [vpas[0],vpas[1],vpas[2],vpas[3]],
                 [vpas[4],vpas[5],vpas[6]]]

    ue_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_ue_48kHz/test/results/" # GroundTruth frame .txt Path
    vpa_labelpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_2022_vpa_48kHz/test/results/" # GroundTruth frame .txt Path

    ue_labels = [f for f in os.listdir(ue_labelpath) if os.path.isfile(os.path.join(ue_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    vpa_labels = [f for f in os.listdir(vpa_labelpath) if os.path.isfile(os.path.join(vpa_labelpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ
    results = ue_labels + vpa_labels

    ######################## 2022データのソート ########################
    results = sorted(results, key=lambda s: float(re.findall(r'[^_]*_alone_week([0-9]+).*', s)[0]))
    results = sorted(results, key=lambda s: re.findall(r'([^_]*)_alone_week[0-9]+.*', s)[0])

    ######################## 交差検証 ########################
    print("all_test:", tests)
    print("all_vpas:", vpas)
    print("-------------------------------------------------")

    anss = 0
    sums = 0
    for test_individual in tests: # ue個体ごと
        for vpa_individual in vpas: # vpa個体ごと

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

                ######################## 2022データ ########################
                pattern = '[^_]*_alone_week([0-9]+).*'
                date = re.findall(pattern ,result)[0]
                pattern = '([^_]*)_alone_week[0-9]+.*'
                name = re.findall(pattern ,result)[0]

                ######################## ue unpickのラベル列を抽出 ########################
                if name in remaining_tests:
                    data = np.loadtxt(ue_labelpath + result, dtype=int)
                    data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                    data_hstack_ue_unpick = np.hstack([data_hstack_ue_unpick, data])
            print("ue_unpick \t",len(data_hstack_ue_unpick))
            prob_ue_unpick = del_nocall_tp(data_hstack_ue_unpick, copy.deepcopy(node_label))

            ######################## unpick個体のみでvpaの遷移表作成 ########################
            data_hstack_vpa_unpick = np.empty(0, dtype=int) # vpa unpickのラベル列の結合用
            for i, result in enumerate(results):

                ######################## 2022データ ########################
                pattern = '[^_]*_alone_week([0-9]+).*'
                date = re.findall(pattern ,result)[0]
                pattern = '([^_]*)_alone_week[0-9]+.*'
                name = re.findall(pattern ,result)[0]

                ######################## vpa unpickのラベル列を抽出 ########################
                if name in remaining_vpas:
                    data = np.loadtxt(vpa_labelpath + result, dtype=int)
                    data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                    data_hstack_vpa_unpick = np.hstack([data_hstack_vpa_unpick, data])

            print("vpa_unpick \t", len(data_hstack_vpa_unpick))
            prob_vpa_unpick = del_nocall_tp(data_hstack_vpa_unpick, copy.deepcopy(node_label))

            ######################## pick個体のみでueの遷移表作成 ########################
            data_hstack_ue_pick = np.empty(0, dtype=int) # ue unpickのラベル列の結合用
            for i, result in enumerate(results):

                ######################## 2022データ ########################
                pattern = '[^_]*_alone_week([0-9]+).*'
                date = re.findall(pattern ,result)[0]
                pattern = '([^_]*)_alone_week[0-9]+.*'
                name = re.findall(pattern ,result)[0]

                ######################## ue unpickのラベル列を抽出 ########################
                if name == test_individual:
                    data = np.loadtxt(ue_labelpath + result, dtype=int)
                    data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                    data_hstack_ue_pick = np.hstack([data_hstack_ue_pick, data])

            print("ue_pick \t", len(data_hstack_ue_pick))
            prob_ue_pick = del_nocall_tp(data_hstack_ue_pick, copy.deepcopy(node_label))
            # prob_ue_pick = data_hstack_ue_pick

            ######################## pick個体のみでvpaの遷移表作成 ########################
            data_hstack_vpa_pick = np.empty(0, dtype=int) # vpa unpickのラベル列の結合用
            for i, result in enumerate(results):

                ######################## 2022データ ########################
                pattern = '[^_]*_alone_week([0-9]+).*'
                date = re.findall(pattern ,result)[0]
                pattern = '([^_]*)_alone_week[0-9]+.*'
                name = re.findall(pattern ,result)[0]

                ######################## vpa unpickのラベル列を抽出 ########################
                if name in vpa_individual:
                    data = np.loadtxt(vpa_labelpath + result, dtype=int)
                    data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
                    data_hstack_vpa_pick = np.hstack([data_hstack_vpa_pick, data])
            
            print("vpa_pick \t", len(data_hstack_vpa_pick))
            prob_vpa_pick = del_nocall_tp(data_hstack_vpa_pick, copy.deepcopy(node_label))
            # prob_vpa_pick = data_hstack_vpa_pick

            ######################## pick個体（ue, vpa）をUEかVPAか判定：cos類似度 か L2ノルム ########################
            print(prob_ue_unpick)
            print(prob_ue_pick)
            print(prob_vpa_unpick)
            print(prob_vpa_pick)

            test_ans = compare_matrices_oneline(prob_ue_unpick, prob_vpa_unpick, prob_ue_pick)
            vpa_ans = compare_matrices_oneline(prob_ue_unpick, prob_vpa_unpick, prob_vpa_pick)

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

            # test_ans = calc_transition(prob_ue_unpick, prob_vpa_unpick, prob_ue_pick)
            # vpa_ans = calc_transition(prob_ue_unpick, prob_vpa_unpick, prob_vpa_pick)


            if test_ans == "UE":
                print(f"{test_individual:12s}: UE/{test_ans}")
                anss += 1
            elif test_ans == "Equal":
                print(f"{test_individual:12s}: UE/{test_ans}")
                sums -= 1
            else:
                print(f"{test_individual:12s}: UE/{test_ans}")

            if vpa_ans == "VPA":
                print(f"{vpa_individual:12s}: VPA/{vpa_ans}")
                anss += 1
            elif vpa_ans == "Equal":
                print(f"{vpa_individual:12s}: VPA/{test_ans}")
                sums -= 1
            else:
                print(f"{vpa_individual:12s}: VPA/{vpa_ans}")
                
            sums += 2


            print("-------------------------------------------------")
            

        #     break
        # break
    acc = anss/sums
    print(f"Acc:\t {acc}")
