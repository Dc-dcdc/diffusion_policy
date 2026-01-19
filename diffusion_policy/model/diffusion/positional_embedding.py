import math
import torch
import torch.nn as nn
# 正弦位置编码，将一个简单的数字（如时间步 $t=50$）转换成一个高维的特征向量
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  #保证了当索引i走到最后时，频率刚好衰减到 $1/10000$
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) #生成[0, 1, 2, ..., half_dim] 的序列并乘上衰减系数
        emb = x[:, None] * emb[None, :]  #形状变成 [Batch_Size, half_dim]  每一行代表一个样本在不同频率下的相位角度。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)#正弦+余弦  形状 [Batch_Size, dim]
        return emb
