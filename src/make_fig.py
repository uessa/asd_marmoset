import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import librosa.display

start = 1
stop = 33103
time = np.linspace(0, 353, stop)

spec_pass = '/datanet/users/hkawauchi/vad_marmoset/datasets/subset_ayapyon_calltype/test'
spec = np.load(spec_pass + '/VOC_190315-0443_あやぴょん_14W.npy')
spec = spec[:, :, 0]
ref = np.median(np.abs(spec))
powspec = librosa.amplitude_to_db(np.abs(spec), ref=ref)

fig = plt.figure(figsize=(10, 10))
librosa.display.specshow(
    powspec,
    sr=96000,
    hop_length=1024,
    cmap="rainbow",
    # cmap="RdYlBu",
    x_axis="time",
    y_axis="hz",
    norm=Normalize(vmin=-20, vmax=20),
)
plt.xlim(100, 150)
plt.colorbar(format="%+2.0fdB")
plt.show()
fig.savefig("spec.pdf")