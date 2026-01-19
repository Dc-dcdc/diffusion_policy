import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #       kernel_size=3, stride=2, padding=1
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1) #刚好能让输出长度精确减半 [batch,channel,horizon] -> [batch,channel,horizon/2]

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #             kernel_size=, stride=2表示滑动步长, padding=1表示对进行裁剪去掉一个      A*w1 A*w2 A*w3 A*w4 
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1) #刚好能让输出长度精确翻倍    AB->            B*w1 B*w2 B*w3 B*w4    后进行叠加裁剪 -> A*w2 A*w3+B*w1 A*w4+B*w3 B*w3

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))
    o = cb(x)
