# -*- coding: utf-8 -*-
import pathlib
import random
import itertools
import argparse
import numpy as np
from os import chmod

parser = argparse.ArgumentParser(
    description="Make {train,valid,test} set randomly and create symbolic links"
)
parser.add_argument("in_dir", type=str, help="Absolute path for raw dataset")
parser.add_argument("out_dir", type=str, help="Absolute path for raw sub-dataset")
parser.add_argument(
    "proportion",
    nargs=3,
    type=float,
    help="Ratio of the number of train, valid and test data",
)
parser.add_argument(
    "-l",
    "--flag_label",
    default=0,
    help="If 1, then copy labels from in_dir to out_dir.parent",
)
args = parser.parse_args()


def randomize(in_dir, out_dir, proportion, flag_label):
    # Paths
    in_path = pathlib.Path(in_dir)
    out_path = pathlib.Path(out_dir)
    file_path = out_path.parent / "makelink.sh"

    # make dir
    out_path.mkdir(exist_ok=True)
    pathlib.Path(out_path / "train").mkdir(exist_ok=True)
    pathlib.Path(out_path / "valid").mkdir(exist_ok=True)
    pathlib.Path(out_path / "test").mkdir(exist_ok=True)
    pathlib.Path(out_path.parent / "train").mkdir(exist_ok=True)
    pathlib.Path(out_path.parent / "valid").mkdir(exist_ok=True)
    pathlib.Path(out_path.parent / "test").mkdir(exist_ok=True)

    # Get file list
    # datalist = in_path.glob("*.wav")

    # Calc size of each dataset
    # datalist, tmp = itertools.tee(datalist)
    # n_data = len(list(tmp))
    # n_train, n_valid, n_test = prop2num(n_data, proportion)
    # print(n_train)
    # print(n_valid)
    # print(n_test)

    # randomize
    dic_marmoset = {
        0: ["あいぴょん"],
        1: ["あやぴょん"],
        2: ["イカ玉"],
        3: ["エバート"],
        4: ["カルビ"],
        5: ["スカイドン"],
        6: ["テレスドン"],
        7: ["ドラコ"],
        8: ["ビスコッテイー", "ビスコッティー"],
        9: ["ぶた玉"],
        10: ["ブラウニー"],
        11: ["マティアス"],
        12: ["マルチナ"],
        13: ["ミコノス"],
        14: ["阿伏兎"],
        15: ["黄金"],
        16: ["花月"],
        17: ["会津"],
        18: ["三春"],
        19: ["信成"],
        20: ["真央"],
        21: ["鶴ヶ城"],
        22: ["梨花"]
    }
    n_data = len(dic_marmoset)
    n_train, n_valid, n_test = prop2num(n_data, proportion)
    setdic = {"0": "train", "1": "valid", "2": "test"}
    idx = [0] * n_data
    idx[n_train : n_train + n_valid] = [1] * n_valid
    idx[n_train + n_valid :] = [2] * (n_data - n_train - n_valid)
    random.shuffle(idx)
    random.shuffle(idx)
    random.shuffle(idx)

    # make shellscript
    with open(file_path, "w") as f:
        f.write("#!/bin/sh\n\n")
        f.write("INDIR=" + str(in_path) + "\n")
        f.write("DATAOUTDIR=" + str(out_path) + "\n")
        f.write("LABELOUTDIR=" + str(out_path.parent) + "\n\n")
        # make link
        count = 0
        for dm in range(n_data):
            for l in dic_marmoset[dm]:
                datalist = in_path.glob("*" + l + "*.wav")
                datalist, tmp = itertools.tee(datalist)
                for d in datalist:
                    # write *.wav
                    fname = setdic[str(idx[dm])] + "/" + d.name
                    command = 'ln -s $INDIR/"' + d.name + '" $DATAOUTDIR/"' + fname + '"\n'
                    f.write(command)
                    if flag_label:
                        command = 'cp $INDIR/"' + d.name + '" $LABELOUTDIR/"' + fname + '"\n'
                        f.write(command.replace("wav", "txt"))

    chmod(file_path, 0o744)


def prop2num(n, proportion):
    "Compute the number of data corresponding to proportion"
    if not sum(proportion) == 1:
        raise Exception("The sum of second argument must be 1")

    # main
    retval = [0] * len(proportion)
    for i, p in enumerate(proportion):
        retval[i] = np.floor(n * p).astype(np.int32)

    # Surplus is divided equally
    tmp = (n - sum(retval)) / len(proportion)
    retval[0] += np.ceil(tmp).astype(np.int32)
    retval[1:] += np.floor(tmp).astype(np.int32)

    return retval


if __name__ == "__main__":
    # in_dir = "/home/kyamaoka/chainer_template/raw/VAD"
    # out_dir = "/home/kyamaoka/chainer_template/datasets/VAD_test/raw"
    # proportion = (0.7, 0.1, 0.2)
    # flag_label = 1
    randomize(args.in_dir, args.out_dir, tuple(args.proportion), args.flag_label)
