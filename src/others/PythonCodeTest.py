import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph
import itertools
import copy

# def process_matrix(matrix, array):
#     result_array = []
#     for i in range(len(array)-1):
#         result_array.append(matrix[array[i]-1, array[i+1]-1])
#     return np.array(result_array)

def process_matrix(matrix, array):
    result_array = matrix[array[:-1] - 1, array[1:] - 1]
    return result_array

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])
array = np.array([1, 2, 3, 4])

result = process_matrix(matrix, array)
print(result)  # 結果の





