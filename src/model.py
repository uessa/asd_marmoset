import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class NetworkDNN(nn.Module):
    def __init__(self):
        super(NetworkDNN, self).__init__()
        ch_in = 1026
        ch_out = 4
        ch1, ch2, ch3, ch4 = 513, 256, 128, 64
        self.net = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(ch_in),
            nn.Linear(ch_in, ch1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(ch1),
            nn.Linear(ch1, ch2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(ch2),
            nn.Linear(ch2, ch3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(ch3),
            nn.Linear(ch3, ch4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(ch4),
            nn.Linear(ch4, ch_out),
        )


    def forward(self, x):
        return self.net(x)


class NetworkCNN(nn.Module):
    def __init__(self):
        super(NetworkCNN, self).__init__()
        ch_in = 1
        ch_out = 5
        ch_1, ch_2, ch_3, ch_4 = 10, 20, 20, 10
        self.net = nn.Sequential(
            nn.Conv2d(
                ch_in, ch_1, kernel_size=3, stride=1, padding=1, dilation=1
            ),
            nn.MaxPool2d(
                kernel_size=5, stride=(5, 1), padding=(0, 2), dilation=1, ceil_mode=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(ch_1),
            nn.Conv2d(
                ch_1, ch_2, kernel_size=3, stride=1, padding=2, dilation=2
            ),
            nn.MaxPool2d(
                kernel_size=5, stride=(5, 1), padding=(0, 2), dilation=1, ceil_mode=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(ch_2),
            nn.Conv2d(
                ch_2, ch_3, kernel_size=3, stride=1, padding=3, dilation=3
            ),
            nn.MaxPool2d(
                kernel_size=5, stride=(5, 1), padding=(0, 2), dilation=1, ceil_mode=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(ch_3),
            nn.Conv2d(
                ch_3, ch_4, kernel_size=3, stride=1, padding=4, dilation=4
            ),
            nn.MaxPool2d(
                kernel_size=3, stride=(3, 1), padding=(0, 1), dilation=1, ceil_mode=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(ch_4),
            nn.Conv2d(
                ch_4, ch_out, kernel_size=3, stride=1, padding=1, dilation=1
            ),
            nn.AvgPool2d(
                kernel_size=(3, 3), stride=(3, 1), padding=(0, 1), ceil_mode=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(ch_out),
        )


    def forward(self, x):
        return self.net(x)
    
