# -*- coding: utf-8 -*-
import torch
import numpy as np
from pathlib import Path
import path


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, path, arch, transform=None):
        self.transform = transform

        self.list_spec = sorted(list(path.glob("*.npy")))
        self.list_label = sorted(list(path.glob("*.txt")))
        self.datanum = len(self.list_spec)
        self.arch = arch

    def __len__(self):
        if self.arch == "cnn":
            return self.datanum

    def __getitem__(self, idx):
        if self.arch == "cnn":
            self.outdata_spec, self.outdata_label = self.getonesample(idx)
            self.outdata_spec = self.outdata_spec[np.newaxis, :, :]
            self.outdata_spec = torch.from_numpy(self.outdata_spec).float()
            self.outdata_label = torch.from_numpy(self.outdata_label).long()
            return self.outdata_spec, self.outdata_label

    def getonesample(self, idx):
        path_spec = self.list_spec[idx]
        path_label = self.list_label[idx]
        
        # if スペクトログラムのnpyがモノラルであったら then
        outdata_spec = np.load(path_spec)
        if len(outdata_spec.shape) == 2:
            outdata_spec = outdata_spec
        else:
            outdata_spec = outdata_spec[:, :, 0]

        outdata_label = np.loadtxt(path_label)

        if self.transform:
            out_data = self.transform(out_data)

        return outdata_spec, outdata_label


if __name__ == "__main__":
    p = path.path("subset_marmoset_new_vpa")
    test = p.test
    mydataset = Mydatasets(test, "cnn")
    testloader = torch.utils.data.DataLoader(
        mydataset, batch_size=2, shuffle=False, num_workers=1
    )
    print(mydataset.__getitem__(1)[0].shape) # スペクトログラム（.npy）のshape
    print(mydataset.__getitem__(0)[0].shape) # スペクトログラム（.npy）のshape

    # print(mydataset.__getitem__(0)[1].shape) # 正解ラベル（.txt）のshape
    # print(mydataset.__getitem__(0)[1]) # 正解ラベル表示
