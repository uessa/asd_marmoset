# -*- coding: utf-8 -*-
"STFT: Package for short-time Fourier Transform"
import numpy as np


def STFT(sig, fftlen, fftsft, wnd):
    "sig: time domain signal (vector, sig.shape -> (len, ))"
    "fftlen: fft frame length (even number, must be dividable by fftsft)"
    "fftsft: fft frame shift"
    "wnd: window function"

    # zero padding
    zero_pad = fftlen
    tmp = np.zeros(zero_pad)
    sig = np.concatenate([tmp, sig, tmp])
    l_sig = np.size(sig)

    # set number of time frames and frequency bins
    n_frame = int(np.ceil((l_sig - fftlen) / fftsft) + 1)
    n_freq = fftlen // 2 + 1

    # zero padding
    tmp = np.zeros([(n_frame - 1) * fftsft + fftlen - l_sig])
    sig = np.concatenate([sig, tmp])

    buf = np.zeros([fftlen], dtype=np.complex)
    SIG = np.zeros([n_freq, n_frame], dtype=np.complex)

    # main
    for i in range(0, n_frame):
        head = i * fftsft
        buf = np.fft.fft(sig[head : (head + fftlen)] * wnd)
        SIG[:, i] = buf[0:n_freq]

    return SIG


def mSTFT(sig, fftlen, fftsft, wnd):
    "sig: time domain signal (sig.shape -> (len, ch))"
    "fftlen: fft frame length (even number, must be dividable by fftsft)"
    "fftsft: fft frame shift"
    "wnd: window function"

    if sig.ndim == 1:
        l_sig = sig.size
        n_ch = 1
        sig = sig[:, np.newaxis]
    else:
        l_sig, n_ch = sig.shape
        if l_sig < n_ch:
            sig = sig.T
            l_sig, n_ch = sig.shape

    zero_pad = fftlen

    # main
    n_frame = int(np.ceil((l_sig + 2 * zero_pad - fftlen) / fftsft) + 1)
    n_freq = fftlen // 2 + 1
    SIG = np.zeros([n_freq, n_frame], dtype=complex)
    mSIG = np.zeros([n_freq, n_frame, n_ch], dtype=complex)

    for k in range(0, n_ch):
        # if sig.shape = (len, ) , then sig[:,k] reports error
        SIG = STFT(sig[:, k], fftlen, fftsft, wnd)
        mSIG[:, :, k] = SIG

    return np.squeeze(mSIG)


def iSTFT(SIG, fftsft, wnd):
    "SIG: STFT domain signal"
    "fftsft: fft frame shift"
    "wnd: window function"

    # set default values
    n_freq, n_frame = SIG.shape
    fftlen = (n_freq - 1) * 2
    zero_pad = fftlen

    # main
    buf = np.zeros([fftlen], dtype=complex)
    tmpsig = np.zeros([(n_frame - 1) * fftsft + fftlen])

    SIG[0, :] /= 2
    SIG[n_freq - 1, :] /= 2

    for t in range(0, n_frame):
        head = t * fftsft
        buf[0:n_freq] = SIG[:, t]
        tmpsig[head : head + fftlen] = (
            tmpsig[head : head + fftlen] + np.real(np.fft.ifft(buf, fftlen) * wnd) * 2
        )

    sig = tmpsig[zero_pad : (-1 * zero_pad)]

    return sig


def miSTFT(mSIG, fftsft, wnd):
    "mSIG: mult-channel STFT domain signal"
    "fftsft: fft frame shift"
    "wnd: window function"

    # set default values
    if mSIG.ndim == 2:
        n_freq, n_frame = mSIG.shape
        mSIG = mSIG[:, :, np.newaxis]
    else:
        n_freq, n_frame, n_ch = mSIG.shape
    fftlen = (n_freq - 1) * 2

    # main
    sig = np.zeros([(n_frame + 1) * fftsft - fftlen, n_ch])

    for k in range(0, n_ch):
        sig[:, k] = iSTFT(mSIG[:, :, k], fftsft, wnd)

    return np.squeeze(sig, axis=(1,))
