# -*- coding: utf-8 -*-
import sys
import numpy as np
import librosa
import soundfile as sf

sys.path.append("../")
from functions import STFT


def wav2spc(sig, fftlen, fftsft, wnd):
    ampspec = np.abs(STFT.mSTFT(sig, fftlen, fftsft, wnd))
    print("ampspec",ampspec.shape)
    return  20*np.log(ampspec + 10**(-10))


if __name__ == "__main__":
    sig, fs = librosa.load("01aa0101.wav", sr=16000, mono=False)
    SIG = wav2spc(sig, 1024, 512, np.hamming(1024))
    rsig = STFT.miSTFT(SIG, 512, np.hamming(1024))
    rsig = rsig[0 : len(sig)]
    sf.write("test.wav", rsig, fs)
    print(np.sum(sig - rsig))
