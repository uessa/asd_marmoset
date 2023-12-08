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

calls = {
    "Twitter",
    "Trill",
    "Trill-Phee",
    "Phee",
    "Cough",
    "Tsik",
    "Cry",
    "Unknown",
    "Ek",
    "Phee-Trill",
    "Breath",
    "Sneeze",
    "Chatter",
    "Ek-Tsik",
    "Intermittent Phee",
    "Chirp",
    }

def write_label(list_file):
    # parameters
    fftlen = 2048
    fftsft = fftlen // 2
    fs = 48000
    sig, fs = librosa.load(list_file[1], sr=fs, mono=False)
    n_frame = (np.ceil((len(sig[0]) - fftlen) / fftsft) + 5).astype(int)
    print(n_frame)

    # Load
    label = np.loadtxt(list_file[0], dtype=str, delimiter="\t")
    Marmoset = []
    ed_frame_marmoset = 0

    # 先頭zero padding
    Marmoset += ['0'] * 2

    for t in label:
        if t[1] == 'Call' or t[1] == 'Calls':
            # 音声区間開始フレームを取得
            st_time = np.round(float(t[0])*fs)
            st_frame = np.ceil(st_time/fftsft).astype(int)
            
            # 前回の音声区間終了フレームから，開始フレームまでを0に設定
            Marmoset += ['0'] * (st_frame - ed_frame_marmoset - 1)

            # 音声区間終了フレームを取得
            ed_time = np.round(float(t[3])*fs)
            ed_frame_marmoset = np.floor(ed_time / fftsft).astype(int)

            # ##### 8クラス #####
            
            # if 'phee' in t[2].lower() and len(t[2]) < 6:
            #     Marmoset += ['1'] * (ed_frame_marmoset - st_frame + 1)
                
            # elif 'trill' in t[2].lower() and len(t[2]) < 7:
            #     Marmoset += ['2'] * (ed_frame_marmoset - st_frame + 1)

            # elif 'twitter' in t[2].lower():
            #     Marmoset += ['3'] * (ed_frame_marmoset - st_frame + 1)

            # elif 'phee-trill' in t[2].lower() and len(t[2]):
            #     Marmoset += ['5'] * (ed_frame_marmoset - st_frame + 1)

            # elif 'trill-phee' in t[2].lower():
            #     Marmoset += ['6'] * (ed_frame_marmoset - st_frame + 1)

            # elif 'unknown' in t[2].lower():
            #     Marmoset += ['7'] * (ed_frame_marmoset - st_frame + 1)

            # else:
            #     Marmoset += ['4'] * (ed_frame_marmoset - st_frame + 1) # 想定ラベル以外は”others”としてに集約
            
            
            ##### 13クラス #####
            if 'phee' in t[2].lower() and len(t[2]) < 6: # 6未満としphee-trillと区別
                Marmoset += ['1'] * (ed_frame_marmoset - st_frame + 1)
                
            elif 'trill' in t[2].lower() and len(t[2]) < 7: # 7未満としtrill-pheeと区別
                Marmoset += ['2'] * (ed_frame_marmoset - st_frame + 1)

            elif 'twitter' in t[2].lower():
                Marmoset += ['3'] * (ed_frame_marmoset - st_frame + 1)

            elif 'tsik' in t[2].lower() and len(t[2]) < 6: # 6未満としek-tsikと区別
                Marmoset += ['4'] * (ed_frame_marmoset - st_frame + 1)

            elif 'ek' in t[2].lower() and len(t[2]) < 4: # 4未満としek-tsikと区別
                Marmoset += ['5'] * (ed_frame_marmoset - st_frame + 1)

            elif 'ek-tsik' in t[2].lower():
                Marmoset += ['6'] * (ed_frame_marmoset - st_frame + 1)

            elif 'cough' in t[2].lower():
                Marmoset += ['7'] * (ed_frame_marmoset - st_frame + 1)

            elif 'cry' in t[2].lower():
                Marmoset += ['8'] * (ed_frame_marmoset - st_frame + 1)

            elif 'chatter' in t[2].lower():
                Marmoset += ['9'] * (ed_frame_marmoset - st_frame + 1)

            elif 'breath' in t[2].lower():
                Marmoset += ['10'] * (ed_frame_marmoset - st_frame + 1)

            elif 'unknown' in t[2].lower():
                Marmoset += ['11'] * (ed_frame_marmoset - st_frame + 1)

            elif 'phee-trill' in t[2].lower():
                Marmoset += ['12'] * (ed_frame_marmoset - st_frame + 1)

            elif 'trill-phee' in t[2].lower():
                Marmoset += ['13'] * (ed_frame_marmoset - st_frame + 1)

            else:
                Marmoset += ['11'] * (ed_frame_marmoset - st_frame + 1) # 想定ラベル以外は”unknown”としてに集約
            

    # 音声終了までのフレームを0で埋める, 末尾zero padding込
    if len(Marmoset) < n_frame:
        Marmoset += ['0'] * (n_frame - len(Marmoset))

    Marmoset = '\n'.join(Marmoset) + '\n'

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
    path_text = pathlib.Path("/home/muesaka/projects/marmoset/raw/marmoset_11vpa_text/nog_remake")
    path_wav = pathlib.Path("/home/muesaka/projects/marmoset/raw/marmoset_11vpa_wav")
    (path_text / 'label').mkdir(exist_ok=True)

    # Get file list
    list_text = path_text.glob("./*.txt")
    list_wav = path_wav.glob("*.wav")
    sort_list_text = sorted(list_text)
    sort_list_wav = sorted(list_wav)
    list_data = np.c_[list(sort_list_text), list(sort_list_wav)]
    write_label_para(list_data)