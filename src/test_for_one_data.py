#-------------------------------------#
# 
#
#-------------------------------------#
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

import dataset
import logger
import util
import model
import path
import trainer
from functions import chk

plt.switch_backend("agg")


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("set", help="Directory name of dataset")
    parser.add_argument("test_dir", help="Directory name for test data")
    parser.add_argument(
        "--model", help="Model path", type=str, default="../model/trained_model.pth"
    )
    parser.add_argument("--batch_size", help="Batch size", type=int, default=5)
    return parser.parse_args()


def print_cmd_line_arguments(args, log):
    log("-----Parameters-----")
    for key, item in args.__dict__.items():
        log(str(key) + ": " + str(item))
    log("--------------------")


def get_data_loaders(path_to_test_dir, arch, batch_size=1):
    classes = ("0", "1", "2", "3", "4")
    num_classes = 5
    testset = dataset.Mydatasets(path_to_test_dir, arch)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return testloader, num_classes, classes


def collate_fn(batch):
    images, labels = [], []

    n_ch, n_freq, _n_frame = batch[0][0].shape
    sig_len = 0
    for image, label in batch:
        if sig_len < image.shape[2]:
            sig_len = image.shape[2]

    for image, label in batch:
        padded_image = np.zeros([n_ch, n_freq, sig_len])
        padded_label = np.zeros([1, sig_len])

        padded_image[:, :, 0 : image.shape[2]] = image
        padded_label[:, 0 : image.shape[2]] = label

        images.append(torch.from_numpy(padded_image).float())
        labels.append(torch.from_numpy(padded_label).long())

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels


def calculate_accuracy(loader, net, num_classes, classes, log, out_dir):
    class_correct = list(0.0 for i in range(num_classes))
    class_total = list(0.0 for i in range(num_classes))
    cm = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for data in loader:
            waveforms, labels = data
            outputs = net(waveforms)
            _, predicted = torch.max(outputs, 1)
            np.savetxt("test.txt", np.squeeze(predicted)) #####
            # np.savetxt("test.txt", predicted)

            for n in range(len(labels)):

                print("label:",labels[n][0])
                # print("predict:",predicted[n][0])
                print("predict:",predicted[n])
                print("list_num_classes:", [i for i in range(num_classes)])
                print("")

                cm += confusion_matrix(
                    # labels[n][0], predicted[n][0], labels=[i for i in range(num_classes)]
                    labels[n][0], predicted[n], labels=[i for i in range(num_classes)]
                )

    accuracy = np.diag(cm) / np.sum(cm, axis=1)
    total_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    f_score = (2 * accuracy * precision) / (accuracy + precision)
    for i in range(num_classes):
        log(
            "Class %5s) Recall : %2f %%, Precision : %2f %%, F1-score : %2f %%"
            % (classes[i], 100 * accuracy[i], 100 * precision[i], 100 * f_score[i])
        )
    log("Total accuracy : %2f %%" % (100 * total_accuracy))

    # make_confusion_matrix(cm, out_dir)
    # make_fig(waveforms, predicted, labels, out_dir)


def save_output(loader, out_dir, net, num_classes, classes, log):
    with torch.no_grad():
        flist = loader.dataset.list_spec
        for i, data in enumerate(loader):
            print("check net(waveforms)")
            waveforms, labels = data
            outputs = net(waveforms)
            _, predicted = torch.max(outputs, 1)

            # save
            fpath = out_dir / str(flist[i]).split("/")[-1].replace(".npy", ".txt")
            np.savetxt(fpath, np.squeeze(predicted), fmt="%.0f")
            # np.savetxt(fpath, predicted, fmt="%.0f")
            print("save: " + str(fpath))

'''
def make_confusion_matrix(cm, out_dir):
    cm_label = ["No Call", "Phee", "Trill", "Twitter", "Other Calls"]
    # cm_label_estimate = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Tsik', 'Ek', 'Ek-Tsik', 'Cough', 'Cry', 'Chatter', 'Breath', 'Unknown']
    # cm_label = ['No Call', 'Phee', 'Trill', 'Twitter', 'Phee-Trill', 'Trill-Phee', 'Unknown', 'Other Calls']
    # 行毎に確率値を出して色分け
    cm_prob = cm / np.sum(cm, axis=1, keepdims=True)
    
    fig = plt.figure(figsize=(8, 4))
    plt.rcParams["font.size"] = 15
    sns.heatmap(
        cm_prob,
        annot=cm,
        # cmap="GnBu",
        cmap="Greys",
        xticklabels=cm_label,
        yticklabels=cm_label,
        fmt=".10g",
        cbar_kws=dict(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    )
    plt.ylim(cm.shape[0], 0)
    plt.xlabel("Estimated Label")
    plt.ylabel("Target Label")
    plt.yticks(rotation=0,rotation_mode="anchor",ha="right",)
    plt.xticks(rotation=30,)
    plt.tight_layout()
    plt.show()
    fig.savefig(out_dir / "confusion_matrix.pdf")



def make_fig(waveforms, predicted, labels, out_dir):
    # Spectrogram_GroundTruthLabel_EstimateLabel
    predicted = np.squeeze(predicted)
    labels = np.squeeze(labels)
    start = 1
    stop = labels.size()[0]
    end_time = stop * 1024 / 96000
    time = np.linspace(0, end_time, stop)

    plt.rcParams["font.family"] = "Times New Roman" # フォントファミリー

    # スペクトログラム
    ref = np.median(np.abs(waveforms))
    powspec = librosa.amplitude_to_db(np.abs(waveforms), ref=ref)
    powspec = np.squeeze(powspec)
    fig, ax = plt.subplots(
        3, 1, figsize=(40, 10), gridspec_kw={"height_ratios": [3, 2, 2]}, dpi=512
    )
    ax[0].set_title("Spectrogram", fontsize=24)
    librosa.display.specshow(
        powspec,
        sr=96000,
        hop_length=1024,
        cmap="rainbow_r",
        # cmap="RdYlBu",
        x_axis="time",
        y_axis="hz",
        norm=Normalize(vmin=-10, vmax=2),
        ax=ax[0],
    )
    
    LEN = [200, 220]
    # LEN = [197.2, 206.8]
    # LEN = [200, 202]

    ax[0].set_xlim(LEN)
    # ax[0].set_xticks([198,200,202,204,206])
    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    # ax[0].set_xticks([200, 210, 220, 230, 240, 250], [200, 210, 220, 230, 240, 250])
    # ax[0].set_xticklabels([200, "", 210, "", 220, "", 230, "", 240, "", 250])
    # ax[0].set_xticklabels([115, "", 120, "", 125])
    ax[0].set_yticklabels([0, 20, 40])
    # ax[0].set_yticklabels([0, 10, 20])
    ax[0].set_ylabel("Frequency [kHz]", fontsize=24)
    ax[0].tick_params(labelsize=22)
    # plt.colorbar(format="%+2.0fdB")

    # 正解ラベル
    ax[1].set_title("Annotated Label", fontsize=24)
    color_label = {0: "black", 1: "crimson", 2: "darkgreen", 3: "mediumblue", 4: "gold"}
    call_label = {0: "No Call", 1: "Phee", 2: "Trill", 3: "Twitter", 4: "Other Calls"}
    # line_label = {0: "dashdot", 1: "dashed", 2: "dashed", 3: "dotted", 4: "solid"}
    line_label = {0: "solid", 1: "dashed", 2: "dashdot", 3: "dotted", 4: "solid"}

    for i in range(max(labels)):
        first_plot = True
        labels_convert = labels == (i + 1)
        j = 0
        while j < len(labels):
            if labels_convert[j]:
                start_call = j - 1
                while labels_convert[j]:
                    j = j + 1
                end_call = j + 1
                if first_plot:
                    ax[1].plot(
                        time[start_call:end_call],
                        labels_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=1.5,
                        color=color_label[i + 1],
                        label=call_label[i + 1],
                    )
                    first_plot = False
                else:
                    ax[1].plot(
                        time[start_call:end_call],
                        labels_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=1.5,
                        color=color_label[i + 1],
                    )
            else:
                j = j + 1

    k = 0
    first_plot = True
    while k < len(labels):
        if labels[k] == 0:
            start_no_call = k
            while labels[k] == 0:
                k = k + 1
                if k >= len(labels):
                    break
            end_no_call = k - 1
            if first_plot:
                ax[1].plot(
                    time[start_no_call:end_no_call],
                    labels[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=1.5,
                    color=color_label[0],
                    label=call_label[0],
                )
                first_plot = False
            else:
                ax[1].plot(
                    time[start_no_call:end_no_call],
                    labels[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=1.5,
                    color=color_label[0],
                )
        else:
            k = k + 1
            if k >= len(labels):
                break
    # plt.plot(time, labels/4)
    # plt.plot(time, predicted*0.8, linewidth=1)
    ax[1].set_xlim(LEN)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls'])
    ax[1].set_yticks([0.0, 1.0])
    ax[1].set_yticklabels(["No Call", "Call"])
    # ax[1].legend(loc="center right", fontsize="18")
    ax[1].tick_params(labelsize=22)

    # 推定ラベル
    ax[2].set_title("Estimated Label", fontsize=24)
    for i in range(max(predicted)):
        first_plot = True
        predicted_convert = predicted == (i + 1)
        j = 0
        while j < len(predicted):
            if predicted_convert[j]:
                start_call = j - 1
                while predicted_convert[j]:
                    j = j + 1
                end_call = j + 1
                if first_plot:
                    ax[2].plot(
                        time[start_call:end_call],
                        predicted_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=1.5,
                        color=color_label[i + 1],
                        label=call_label[i + 1],
                    )
                    first_plot = False
                else:
                    ax[2].plot(
                        time[start_call:end_call],
                        predicted_convert[start_call:end_call],
                        linestyle=line_label[i + 1],
                        linewidth=1.5,
                        color=color_label[i + 1],
                    )
            else:
                j = j + 1

    k = 0
    first_plot = True
    while k < len(predicted):
        if predicted[k] == 0:
            start_no_call = k
            while predicted[k] == 0:
                k = k + 1
                if k >= len(labels):
                    break
            end_no_call = k - 1
            if first_plot:
                ax[2].plot(
                    time[start_no_call:end_no_call],
                    predicted[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=1.5,
                    color=color_label[0],
                    label=call_label[0],
                )
                first_plot = False
            else:
                ax[2].plot(
                    time[start_no_call:end_no_call],
                    predicted[start_no_call:end_no_call],
                    linestyle=line_label[0],
                    linewidth=1.5,
                    color=color_label[0],
                )
        else:
            k = k + 1
            if k >= len(labels):
                break
    # plt.plot(time, predicted/4)
    ax[2].set_xlim(LEN)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['No Call', 'Phee', 'Trill', 'Twitter', 'Other Calls'])
    ax[2].set_yticks([0.0, 1.0])
    ax[2].set_yticklabels(["No Call", "Call"])
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize="25", ncol=4)
    ax[2].set_xlabel("Time [s]", fontsize=24)
    ax[2].tick_params(labelsize=22)
    plt.tight_layout()
    plt.savefig(out_dir / "spec_labels.pdf")
    plt.show()
'''

if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    p = path.path(args.set)
    arch = "cnn"
    p_model = pathlib.Path(args.model)
    p_model_parent = p_model.parent
    log = logger.Logger(p_model_parent / "log_accuracy")
    print_cmd_line_arguments(args, log)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dir = p.dataset / args.test_dir
    testloader, num_classes, classes = get_data_loaders(test_dir, arch, args.batch_size)

    # net = model.NetworkCNN()
    net = model.ConvMixer(
        # input_shape=input_shape,
        # frame_length=1,
        frequency_length=1025,
        hidden_dims=529,
        # num_classes=len(cfg.dataset.target_labels),
        depth=21,
        kernel_size=20,
        is_dilated=False,
        # gap_direction=time
        )
    net.load_state_dict(torch.load(args.model))
    net.eval()

    calculate_accuracy(testloader, net, num_classes, classes, log, p_model.parent)

    out_dir = test_dir / "results"
    out_dir.mkdir(exist_ok=True)
    subset_name = "marmoset"
    os.system("end_report 上坂奏人 test={}".format(subset_name))
    save_output(testloader, out_dir, net, num_classes, classes, log)