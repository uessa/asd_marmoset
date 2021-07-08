import time
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
plt.switch_backend("agg")

def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("set", help="Directory name of dataset")
    parser.add_argument('--batch_size', help='Batch size',
                        type=int,
                        default=5)
    parser.add_argument('--seed', help='Random seed',
                        type=int,
                        default=None)
    parser.add_argument('--epoch', help='Number of epochs',
                        type=int,
                        default=100)
    parser.add_argument('--lr', help='Initial learning rate',
                        type=float,
                        default=0.1)
    return parser.parse_args()

def print_cmd_line_arguments(args, log):
    log('-----Parameters-----')
    for key, item in args.__dict__.items():
        log(str(key) + ': ' + str(item))
    log('--------------------')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(path, batch_size, arch):
    # p.mkdir(parents=True, exist_ok=True)

    num_classes = 12

    # trainset = AudioFolder(p / 'train', transform=transform)
    trainset = dataset.Mydatasets(p.train, arch)
    # valset = AudioFolder(p / 'evaluate', transform=transform)
    valset = dataset.Mydatasets(p.valid, arch)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2, collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, collate_fn=collate_fn)
    return trainloader, valloader, num_classes

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

def save_model(model, path):
    torch.save(model.state_dict(), path.model / 'model.pth')

def show_history(train_accuracy, val_accuracy):
    plt.plot(range(len(train_accuracy)), train_accuracy,
             label='Accuracy for training data')
    plt.plot(range(len(val_accuracy)), val_accuracy,
             label='Accuracy for val data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    p = path.path(args.set)
    arch = 'cnn'
    p.model.mkdir(parents=True, exist_ok=True)
    log = logger.Logger(p.model / 'log')
    print_cmd_line_arguments(args, log)
    if args.seed is not None:
        set_random_seed(args.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    trainloader, valloader, num_classes = get_data_loaders(p, args.batch_size, arch)

    # net = model.ResNet('ResNet18', num_classes=num_classes)
    net = model.NetworkCNN()
    net = net.to(device)
    # if arch == 'dnn':
        # summary(net, input_size=(1026,))
    # elif arch == 'cnn':
        # summary(net, input_size=(2, 513, 1000))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, dampening=0,
                              weight_decay=0.0001, nesterov=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [args.epoch//2, 3*args.epoch//4],
                                               0.1)

    start_time = time.time()
    trainer = trainer.CNNTrainer(net, optimizer, criterion,
                                        trainloader, device)
    costs = []
    train_accuracy = []
    val_accuracy = []
    log('-----Training Started-----')
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        loss = trainer.train()
        train_acc = trainer.eval(trainloader)
        val_acc = trainer.eval(valloader)
        log('Epoch: %03d/%03d | Loss: %.4f | Time: %.2f min | Acc: %.4f/%.4f'
              % (epoch+1, args.epoch, loss,
                 (time.time() - start_time)/60,
                 train_acc, val_acc))
        costs.append(loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        scheduler.step()

    log('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    log('-----Training Finished-----')

    save_model(net, p)
    os.system('end_report "ひで(河内)" Training')

    show_history(train_accuracy, val_accuracy)
