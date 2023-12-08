import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvMixer(nn.Module):
    class ConvMixerLayer(nn.Module):
        def __init__(self, hidden_dims, kernel_size, dilation):
            super().__init__()
            self.hidden_dims = hidden_dims
            self.kernel_size = kernel_size
            self.dilation = dilation

            self.depthwise_block = nn.Sequential(
                nn.Conv1d(
                    hidden_dims, hidden_dims, kernel_size=kernel_size, groups=hidden_dims, padding="same", dilation=dilation
                ),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dims)
            )

            self.pointwise_block = nn.Sequential(
                nn.Conv1d(
                    hidden_dims, hidden_dims, kernel_size=1
                ),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dims),
            )
        
        def forward(self, x):
            print('debug4: ', x.shape)
            x = self.depthwise_block(x) + x
            print('debug5: ', x.shape)
            # print("")
            return self.pointwise_block(x)

    def __init__(
        self,
        # input_shape, ######################
        # frame_length,######################
        frequency_length,
        hidden_dims,
        # num_classes,######################
        depth,
        kernel_size,
        is_dilated=False,
        # gap_direction="time" ######################
        ):
        super().__init__()
        # self.input_shape = input_shape # (C, T) or (C, F, T) #######################
        
        # self.output_shape = self.calc_embedding_shape( ######################
        #     # input_shape, #####################
        #     # frame_length, #####################
        #     hidden_dims
        # )
        # length = self.output_shape[-1]
        
        # self.frame_length = frame_length ######################
        self.frequency_length = frequency_length
        self.hidden_dims = hidden_dims
        # self.num_classes = num_classes ######################
        self.depth = depth
        self.kernel_size = kernel_size
        self.is_dilated = is_dilated
        
        # if gap_direction in ["time", "channel"]: ######################
        #     self.gap_direction = gap_direction
        # else:
        #     raise NotImplementedError()

        # if len(input_shape) == 2: ######################
        #     in_channels = input_shape[0]
        # elif len(input_shape) > 2:
        # in_channels = 1
        # for s in input_shape[:-1]:
        #     in_channels *= s
        # else:
        #     raise ValueError("Dimension of input_shape must be >= 2")
        in_channels = self.frequency_length
        
        self.frame_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels, hidden_dims, kernel_size=1
            ),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dims)
            # nn.LayerNorm([hidden_dims, length])
        )

        if is_dilated:
            self.blocks = nn.Sequential(
                *[ConvMixer.ConvMixerLayer(hidden_dims, kernel_size, dilation=2**i) for i in range(depth)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConvMixer.ConvMixerLayer(hidden_dims, kernel_size, dilation=1) for _ in range(depth)]
            )
            
        # tmp_dims = hidden_dims if self.gap_direction == "time" else length
        tmp_dims = hidden_dims # 固定
        num_classes = 5 # 固定
        
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            # nn.Flatten(),
            # nn.Linear(tmp_dims, num_classes),
            nn.Conv1d(hidden_dims, out_channels=5, kernel_size=1),
        )

        return
    
    def forward(self, x):
        # hidden_dims = 500
        # tmp_dims = hidden_dims if True else 16633
        # print(tmp_dims)
        # exit()
        
        print('debug1: ', x.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])
        print('debug2: ', x.shape)
        x = self.frame_embedding(x)
        print('debug3: ', x.shape)
        
        x = self.blocks(x)
        print('debug6: ', x.shape)
        # if self.gap_direction == "channel":
        #     x = x.transpose(1, 2)
        x = self.classifier(x)
        print('debug7: ', x.shape)
        print("")
        # exit()
        return x

    # def calc_embedding_shape(self, input_shape, frame_length, hidden_dims): 
    # def calc_embedding_shape(self, hidden_dims):
    #     stride = (frame_length+1) // 2
    #     length = math.floor((input_shape[-1] - frame_length) / stride + 1)
    #     return (hidden_dims, length)

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
        print("x.shape: ", x.shape)
        print("")
        # exit()
        return self.net(x)
    
