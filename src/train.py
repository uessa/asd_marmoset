import time
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchaudio
from torchsummary import summary
import trainer
import model
import dataset
import path
import logger
import dataset
from focalloss import *
from functions import chk
from util import masked_cross_entropy
torch.backends.cudnn.enabled = False
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
    num_classes = 8
    trainset = dataset.Mydatasets(p.train, arch)
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
    plt.ylim(0,1)
    plt.savefig("train_val_carve.pdf")
    plt.show()
    plt.clf()
    plt.close()

if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    p = path.path(args.set)
    arch = 'cnn'
    p.model.mkdir(parents=True, exist_ok=True)
    log = logger.Logger(p.model / 'log')
    
    print_cmd_line_arguments(args, log) # Parameters表示
    
    if args.seed is not None:
        set_random_seed(args.seed)

    trainloader, valloader, num_classes = get_data_loaders(p, args.batch_size, arch)
    
    print("dataloader finished") # print
    
    ###################### network type ######################
    # net = model.NetworkCNN() ######################
    net = model.ConvMixer(
        # input_shape=input_shape,
        # frame_length=1,
        frequency_length=1025,
        hidden_dims=529,
        # num_classes=len(cfg.dataset.target_labels),
        depth=5,
        kernel_size=5,
        is_dilated=False,
        # gap_direction=time
        )
    print("model finished") # print
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("device finished") # print
    
    net = net.to(device)
    print("net finished") # print

    # criterion = nn.CrossEntropyLoss()
    criterion = masked_cross_entropy
    # criterion = criterion.to(device)
    print("criterion finished") # print
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, dampening=0,
                              weight_decay=0.0001, nesterov=False)
    print("optimizer finished") # print

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [args.epoch//2, 3*args.epoch//4],
                                               0.1)
    print("scheduler finished") # print

    start_time = time.time()
    trainer = trainer.CNNTrainer(net, optimizer, criterion,
                                        trainloader, device)
    print("trainer finished") # print
    
    ###################### train start ######################
    costs = []
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    
    log('-----Training Started-----')
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        # torch.cuda.empty_cache() # cuda キャッシュクリア
        
        loss = trainer.train()
        # print("loss", loss) # print
        train_acc, train_l = trainer.eval(trainloader)
        # print("train_acc", train_acc) # print
        val_acc, val_l = trainer.eval(valloader)
        # print("val_acc",val_acc) # print
        
        log('---Epoch: %03d/%03d | Loss: %.4f | Time: %.2f min | Acc: %.4f/%.4f---'
              % (epoch+1, args.epoch, loss,
                 (time.time() - start_time)/60,
                 train_acc, val_acc))
        
        costs.append(loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        train_loss.append(train_l)
        val_loss.append(val_l)
        
        scheduler.step()
        print("") # print
        
    log('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    log('-----Training Finished-----')

    ###################### save model ######################
    save_model(net, p)
    subset_name = "marmoset"
    os.system("end_report 上坂奏人 train:{}".format(subset_name))
    show_history(train_accuracy, val_accuracy)
    # print("t   ", train_loss)
    # print("v   ", val_loss)
    logger.plot_loss(p, train_loss, val_loss)
