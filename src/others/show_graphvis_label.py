# -*- coding: utf-8 -*-
#-------------------------------------#
# 時間フレームごとのラベルから発声の遷移確率をpdf保存するスクリプト．2022データに対応
#
#-------------------------------------#
import numpy as np
import copy
import itertools
import seaborn as sns
import os
from graphviz import Digraph
import pprint
import matplotlib.pyplot as plt

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
    print(prob)
    
    plt.figure()
    sns.heatmap(prob, cmap="Blues", annot=True, xticklabels=label, yticklabels=label)
    plt.yticks(rotation=0)
    plt.savefig("./LabelRatio/graph/{}.pdf".format(target))


    return prob

def Graphviz(data, node_label):
    states = np.max(data)+1
    g = Digraph()

    for i in range(states):
        g.node(str(i), label=node_label[i])

    edges = np.array([np.sort(np.array([np.arange(states)]*states).flatten()),
                      np.array([np.arange(states)]*states).flatten()]).T

    edge_labels = np.round(tp(data, node_label), 2).flatten().astype('str')

    for i, e in enumerate(edges):
        if edge_labels[i] != '0.0':
            g.edge(str(e[0]), str(e[1]), label=edge_labels[i])

    return g

def del_nocall_Graphviz(data, node_label):
    node_label.remove("NoCall")
    data = data[~(data == 0)]
    data = data - 1

    return Graphviz(data, node_label)


if __name__ == "__main__":
    resultpath = "/home/muesaka/projects/marmoset/datasets/subset_marmoset_{}_48kHz/test/results/".format(target) # GroundTruth frame .txt Path
    results = [f for f in os.listdir(resultpath) if os.path.isfile(os.path.join(resultpath, f)) and f[-3:] == "txt"] # 末尾3文字まで（.txt）マッチ

    savepath = resultpath + "graph/"
    data_hstack = np.empty(0, dtype=int)

    for i, result in enumerate(results):
        print(result)

        data = np.loadtxt(resultpath + result,dtype=int)
        data = np.array([k for k,g in itertools.groupby(data)], dtype=int)
        data_hstack = np.hstack([data_hstack, data])

    node_label = ["NoCall", 'Phee', 'Trill', "Twitter", "Other"]

    g = del_nocall_Graphviz(data_hstack, node_label)
    g.render(savepath + target)