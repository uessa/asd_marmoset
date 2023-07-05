#-------------------------------------#
# 
#
#-------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import librosa.display

# start = 1
# stop = 33103
# time = np.linspace(0, 353, stop)

spec_pass = '/home/muesaka/projects/marmoset/src/dataset'
spec = np.load(spec_pass + '/phee.npy')
spec = spec[:, :]
ref = np.median(np.abs(spec))
powspec = librosa.amplitude_to_db(np.abs(spec), ref=ref)

fig = plt.figure(figsize=(10, 10))
librosa.display.specshow(
    powspec,
    sr=96000,
    hop_length=1024,
    # cmap="rainbow_r",
    cmap="gray",
    x_axis="time",
    y_axis="hz",
    norm=Normalize(vmin=-10, vmax=2),
)


# plt.xlim(100, 150)
# plt.xticks([100, 110, 120, 130, 140, 150])
# plt.colorbar(format="%+2.0fdB")
plt.show()
fig.savefig("/home/muesaka/projects/marmoset/src/others/LabelRatio/spec.pdf")