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
     # Marmosetの音声ディレクトリ（/あやぴょん, /あさぴょん, ...）
     dir_path = pathlib.Path("/datanet/users/muesaka/Marmoset/CallData2021/Recorder") 

     # ファイル名を保存するテキストファイルのパスを指定
     file_path = os.path.join(os.path.dirname(dir_path), "recorder_file_list.txt")

     # ファイル一覧を取得しテキストファイルに保存
     with open(file_path, "w") as f:
          for dirpath, dirnames, filenames in os.walk(dir_path):
               for filename in filenames:
                    # 現在見ているファイル名
                    print("read:{}".format(filename))

                    # ファイル名のみを取得
                    base_filename = os.path.basename(filename)

                    # 「.」で始まるファイルは削除
                    if base_filename.startswith('.'):
                         os.remove(os.path.join(dirpath, filename))

                    # ファイル名をテキストファイルに書き込み
                    else:
                         f.write(base_filename + "\n")
