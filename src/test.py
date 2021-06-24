import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchsummary import summary
import trainer
import model
import dataset
import path
import logger
import dataset
from functions import chk
from sklearn.metrics import confusion_matrix
plt.switch_backend("agg")

def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("set", help="Directory name of dataset")
    parser.add_argument('--model', help='Model path',
                        type=str,
                        default='../model/trained_model.pth')
    parser.add_argument('--batch_size', help='Batch size',
                        type=int,
                        default=5)
    return parser.parse_args()


def print_cmd_line_arguments(args, log):
    log('-----Parameters-----')
    for key, item in args.__dict__.items():
        log(str(key) + ': ' + str(item))
    log('--------------------')


def get_data_loaders(path, batch_size, arch):
    # classes = ('0', '1', '2', '3')
    # num_classes = 4
    # classes = ('0', '1')
    # num_classes = 2
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11')
    num_classes = 12
    trainset = dataset.Mydatasets(p.train, arch)
    testset = dataset.Mydatasets(p.test, arch)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, collate_fn=collate_fn)
    return trainloader, testloader, num_classes, classes

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

        padded_image[:, :, 0:image.shape[2]] = image
        padded_label[:, 0:image.shape[2]] = label

        images.append(torch.from_numpy(padded_image).float())
        labels.append(torch.from_numpy(padded_label).long())

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels

def calculate_accuracy(loader, net, num_classes, classes, log):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    cm = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for data in loader:
            waveforms, labels = data
            outputs = net(waveforms)
            _, predicted = torch.max(outputs, 1)
            # c = (predicted == labels).squeeze()
            c = (predicted == labels)
            for n in range(len(labels)):
                cm += confusion_matrix(labels[n][0], predicted[n][0], [i for i in range(num_classes)])
            # for i in range(len(labels)):
                # for j in range(labels.size(2)):
                    # label = labels[i, :, j]
                    # class_correct[label] += c[i, :, j].item()
                    # class_total[label] += 1
    accuracy = np.diag(cm) / np.sum(cm, axis=1)
    total_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    f_score = (2 * accuracy * precision) / (accuracy + precision)

    for i in range(num_classes):
        # if not np.isnan(accuracy[i]):
            # log('Accuracy(Recall) of %5s : %2f %%, Precision : %2f %%, F1-score : %2f %%' 
            # % classes[i], 100 * accuracy[i], 100 * precision[i], f_score[i])
        # else:
            # log('Accuracy of %5s : %2f %%' % (classes[i], 0.0))
        log('Accuracy(Recall) of %5s : %2f %%, Precision : %2f %%, F1-score : %2f %%' 
            % (classes[i], 100 * accuracy[i], 100 * precision[i], 100 * f_score[i]))
    
    log('Total accuracy : %2f %%' % (100 * total_accuracy))


if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    p = path.path(args.set)
    arch = 'cnn'
    p_model = pathlib.Path(args.model)
    p_model_parent = p_model.parent
    log = logger.Logger(p_model_parent / 'log_accuracy')
    print_cmd_line_arguments(args, log)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader, num_classes, classes = get_data_loaders(p, args.batch_size, arch)

    net = model.NetworkCNN()
    net.load_state_dict(torch.load(args.model))
    net.eval()

    calculate_accuracy(trainloader, net, num_classes, classes, log)
    calculate_accuracy(testloader, net, num_classes, classes, log)