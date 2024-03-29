# -*- coding: utf-8 -*-
#-------------------------------------#
# 
#
#-------------------------------------#
import numpy as np
import librosa
import pathlib
from multiprocessing import Pool
import glob


def write_label(list_file):
    # parameters
    fftlen = 2048
    fftsft = fftlen // 2
    fs = 96000
    sig, fs = librosa.load(list_file[1], sr=None, mono=False)
    n_frame = (np.ceil((len(sig[0]) - fftlen) / fftsft) + 5).astype(int)
    # print(n_frame)

    # Load
    label = np.loadtxt(list_file[0], dtype=str, delimiter="\t")
    Marmoset = ''
    ed_frame_marmoset = 0

    # 先頭zero padding
    Marmoset += '0\n' * 2

    for t in label:
        if t[1] == 'Call' or t[1] == 'Calls':
            # 音声区間開始フレームを取得
            st_time = np.round(float(t[0])*fs)
            st_frame = np.ceil(st_time/fftsft).astype(int)
            
            # 前回の音声区間終了フレームから，開始フレームまでを0に設定
            Marmoset += '0\n' * (st_frame - ed_frame_marmoset - 1)

            # 音声区間終了フレームを取得
            ed_time = np.round(float(t[3])*fs)
            ed_frame_marmoset = np.floor(ed_time / fftsft).astype(int)

            # 開始から終了までを1に設定
            Marmoset += '1\n' * (ed_frame_marmoset - st_frame + 1)

    # 音声終了までのフレームを0で埋める, 末尾zero padding込
    if len(Marmoset)//2 < n_frame:
        Marmoset += '0\n' * (n_frame - len(Marmoset)//2)

    # print(n_frame)
    # print(len(Marmoset)//2)

    fname = str(path_text) + "/label/" + list_file[1].stem
    print("Write: " + fname + ".txt")
    with open(fname + ".txt", "w") as f:
        f.write(Marmoset)



def pool_initializer():
    import mkl
    mkl.set_num_threads(1)


def write_label_para(list_data):
    # Main: feature extraction
    with Pool(processes=None, initializer=pool_initializer) as p:
        p.map(write_label, list_data)


if __name__ == "__main__":
    # path
    path_text = pathlib.Path()
    path_wav = pathlib.Path('/home/muesaka/projects/marmoset/raw/marmoset_23ue_text')
    (path_text / 'label').mkdir(exist_ok=True)

    # Get file list
    list_text = path_text.glob("./*.txt")
    list_wav = path_wav.glob("*.wav")
    list_data = np.c_[list(list_text), list(list_wav)]
    write_label_para(list_data)
