import argparse
import os
import pathlib
import time

import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix
from torchsummary import summary


# スペクトログラム
waveforms = np.load("Twitter_あやぴょん_3.npy")
print(waveforms.shape)
ref = np.median(np.abs(waveforms))

plt.rcParams["font.family"] = "Times New Roman" # フォントファミリー
powspec = librosa.amplitude_to_db(np.abs(waveforms), ref=ref)
powspec = np.squeeze(powspec)
fig, ax = plt.subplots(figsize=(5, 5), dpi=512)
librosa.display.specshow(
    powspec,
    sr=96000,
    hop_length=1024,
    cmap="rainbow_r",
    # cmap="RdYlBu",
    x_axis="time",
    y_axis="hz",
    norm=Normalize(vmin=-10, vmax=2),
    ax=ax,
)



ax.set_yticks([0, 20000, 40000])
ax.set_yticklabels([0 ,20, 40])
# ax.set_xlim(0, 10)
# ax.set_xticks([0.0, 0.2, 0.4, 0.6])
ax.set_xticks([0.0, 0.5, 1.0])

ax.set_ylabel("Frequency [kHz]",fontsize=30)
ax.set_xlabel("Time [s]",fontsize=30)
ax.tick_params(labelsize=30)
# ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
# ax.axis("off")
plt.tight_layout()
fig.savefig("./spec_twitter.pdf")
plt.close()

