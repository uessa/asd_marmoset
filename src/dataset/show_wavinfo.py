# -*- coding: utf-8 -*-
import pathlib
import argparse
from multiprocessing import Pool
import numpy as np
import librosa
import sys
import datetime
import time
import wav2spc

# ファイルの読み込み
data1, fs1 = librosa.load('/home/muesaka/projects/marmoset/src/dataset/VOC_181214-0319_あやぴょん_1.4W.wav',sr=None,mono=None)
data2, fs2 = librosa.load('/home/muesaka/projects/marmoset/src/dataset/Beatrice_alone_week1.wav',sr=None,mono=None)


# ファイルの情報
print("fs1",fs1)
print("fs2",fs2)
print("data1",data1[0].min()) #librosaの自動正規化範囲の確認
print("data1",data1[1].min())
print("data2",data2.min())
