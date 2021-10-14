import pathlib
import sys
import soundfile as sf
from scipy import signal

# process wav files
def wavread(fn, resr=None):
    if resr is None:
        data, sr = sf.read(fn)
    else:
        data, sr = sf.read(fn)
        data = signal.resample(data, int(resr * len(data) / sr))
        sr = resr
    f = sf.info(fn)
    return data, sr, f.subtype

def calc_length_of_signal():

    path = pathlib.Path("/datanet/users/hkawauchi/vad_marmoset/datasets/subset_marmoset_24UE/raw/test")  # ここを変更する．
    file = list(path.glob("*.wav"))
    total_sec = 0
    for f in file:
        data, sr, subtype = wavread(f)
        print(sr)
        if data.ndim == 2:  # stereo
            total_sec += len(data[:, 0]) / sr
        else:  # mono
            total_sec += len(data) / sr

    m, s = divmod(total_sec, 60)
    print(total_sec)
    print(m)
    print(s)

if __name__ == "__main__":
    calc_length_of_signal()