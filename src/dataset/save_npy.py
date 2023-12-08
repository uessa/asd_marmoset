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

sys.path.append("../")
import path

# def featureExtraction(file):
#     # Load
#     sig, fs = librosa.load(file, sr=args.fs, mono=False)
#     # Feature extraction
#     spc = wav2spc.wav2spc(sig, args.fftlen, args.fftsft, wnd)
#     # Save
#     fname = str(file).replace("/" + args.in_dir, args.out_dir).replace(".wav", "")
#     np.save(fname, spc)

if __name__ == "__main__":
    p = path.Path("Twitter_あやぴょん_3.wav")
    # Load
    sig, fs = librosa.load(p, sr=96000, mono=True)
    # Feature extraction
    spc = wav2spc.wav2spc(sig, 2048, 1024, np.hamming(2048))
    # Save
    fname = str(p).replace(".wav", "")
    np.save(fname, spc)
