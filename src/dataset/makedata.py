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

parser = argparse.ArgumentParser(
    description="Converting waveform data to desired representation"
)

parser.add_argument("dataset", type=str, help="Directory name of your dataset")
parser.add_argument(
    "-i", "--in_dir", default="raw", type=str, help="Directory name of input dataset"
)
parser.add_argument(
    "-o", "--out_dir", default="", type=str, help="Directory name for output dataset"
)
parser.add_argument(
    "-r", "--fs", default=16000, type=np.int32, help="Desired sampling frequency"
)
parser.add_argument(
    "-l", "--fftlen", default=1024, type=np.int32, help="FFT frame length"
)
parser.add_argument(
    "-s", "--fftsft", default=512, type=np.int32, help="FFT frame shift"
)
parser.add_argument(
    "-w",
    "--window",
    default="hamming",
    choices=["bartlett", "blackman", "hamming", "hanning", "kaiser"],
    help="Name of window function",
)
parser.add_argument(
    "-n", "--n_worker", default=None, type=int, help="Number of parallel workers"
)


args = parser.parse_args()
exec("wnd = np." + args.window + "(" + str(args.fftlen) + ")")


def featureExtraction_init():
    import mkl

    mkl.set_num_threads(1)


def featureExtraction(file):
    # Load
    sig, fs = librosa.load(file, sr=args.fs, mono=False)
    # Feature extraction
    spc = wav2spc.wav2spc(sig, args.fftlen, args.fftsft, wnd)
    # Save
    fname = str(file).replace("/" + args.in_dir, args.out_dir).replace(".wav", "")
    np.save(fname, spc)


def makedata(p):
    # Set directory path
    in_path = p.dataset / args.in_dir
    out_path = p.dataset / args.out_dir

    # Make directory
    pathlib.Path(out_path / "train").mkdir(exist_ok=True)
    pathlib.Path(out_path / "test").mkdir(exist_ok=True)
    pathlib.Path(out_path / "valid").mkdir(exist_ok=True)

    # Get file list in your dataset
    dataset = in_path.glob("**/*.wav")

    # Main: feature extraction
    with Pool(args.n_worker, initializer=featureExtraction_init) as p:
        p.map(featureExtraction, dataset)


def report(p, etime, method):
    log_file = p.dataset / "log"
    in_path = p.dataset / args.in_dir
    out_path = p.dataset / args.out_dir
    with open(log_file, "w") as f:
        f.write("Date: " + datetime.datetime.today().isoformat() + "\n")
        f.write("Feature extraction: " + method + "\n")
        f.write("Input data: " + str(in_path) + "/{train,valid,test}\n")
        f.write("Output data: " + str(out_path) + "/{train,valid,test}\n")
        f.write("Elapsed time: " + str(etime) + " sec.\n")


if __name__ == "__main__":
    p = path.path(args.dataset)
    start = time.time()
    makedata(p)
    report(p, time.time() - start, "wav2spc.py")
