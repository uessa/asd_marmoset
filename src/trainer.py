import os
from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from util import AverageMeter
import util


class Trainer(metaclass = ABCMeta):
    @abstractmethod
    def train(self):
        pass


class ClassifierTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 device):
        super(ClassifierTrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def train(self):
        self.net.train()

        loss_meter = AverageMeter()

        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        return loss_meter.average

    def eval(self, dataloader):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total


class CNNTrainer(ClassifierTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, dataloader):
        self.net.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs) # ネットワークの出力
                loss = util.masked_cross_entropy(outputs, labels) # loss計算
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) # ソフトマックス処理

                total += labels.size(0) * labels.size(-1) # 全体のフレームの総数
                correct += (predicted == labels).sum().item() # 正解したフレームの総数

        return correct / total, total_loss / len(dataloader.dataset)


class RegressorTrainer(Trainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 device):
        super(RegressorTrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def train(self):
        self.net.train()

        loss_meter = AverageMeter()

        for inputs, targets in self.dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), number=inputs.size(0))

        return loss_meter.average

    def eval(self, dataloader):
        self.net.eval()

        loss_meter = AverageMeter()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss_meter.update(loss.item(), number=inputs.size(0))

        return loss_meter.average
