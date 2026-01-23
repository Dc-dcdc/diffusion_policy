from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)
# 1D带条件注入的残差块
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),  #负责维度变换（例如从 2 变 512）
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups), #负责特征加深（维度不变，比如 512 到 512）
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels  #(默认)：只学习偏移量Bias 缩放系数固定为1
        if cond_predict_scale: ## 如果要预测缩放系数
            cond_channels = out_channels * 2  #则还需要学习缩放系数scale，所以通道数翻倍
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),  # 1. 激活函数
            nn.Linear(cond_dim, cond_channels), # 2. 线性投影：将条件维数维 对齐 模型输出的维输66*2->256，以便后续对模型输出x进行进一步调节 y = scale*x + bisa
            Rearrange('batch t -> batch t 1'),  # 3. 重排形状：适配 1D 卷积格式   [16, 512]->[16, 512, 1]
        )

        # make sure dimensions compatible
        # 确保维度对齐，以使用残差链接                            一维卷积核
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity() #如果输入输出维度一样则无需处理，否则将输入通道数对齐输出通道数

    def forward(self, x, cond):# x = [16, 64, 16]=[不同演示数据的片段数量batch，通道维数channel，序列长度horizon] 
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) #进入第一个 Conv1dBlock
        embed = self.cond_encoder(cond)  #假设为[16, 256, 1]
        if self.cond_predict_scale:  
            embed = embed.reshape( #形状变化: [16, 256, 1] -> [16, 2, 128, 1]
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...] #[16, 128, 1] 缩放系数
            bias = embed[:,1,...]  #[16, 128, 1] 偏移量
            out = scale * out + bias  #对每条数据有不一样的系数和偏差，对同一条数据的不同序列是一样的
        else:
            out = out + embed #只做加法，不缩放
        out = self.blocks[1](out) #进入第一个 Conv1dBlock
        out = out + self.residual_conv(x) #残差连接，防止网络太深导致梯度消失。
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,  #输入维度  动作  或者 动作+观测
        local_cond_dim=None,
        global_cond_dim=None, #全局条件维数（正弦波时间编码 + ResNet图像特征）
        diffusion_step_embed_dim=256,  #时间步编码维数
        down_dims=[256,512,1024], #下采样每一层通道数，具体数值以配置文件中policy为主
        kernel_size=3, #卷积核，每次获取前后多少输入
        n_groups=8, #归一化分组
        cond_predict_scale=False #控制条件注入（FiLM）层的初始化权重
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)  #构建完整维度列表all_dims = [2, 512, 1024, 2048]
        start_dim = down_dims[0] #512

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),      # 第一步：数学变换  输出是相应维度的正弦/余弦固定向量
            nn.Linear(dsed, dsed * 4),   # 第二步：升维
            nn.Mish(),                   # 第三步：激活函数 
            nn.Linear(dsed * 4, dsed),   # 第四步：降维
        )
        cond_dim = dsed  #条件维度设定为时间特征长度
        if global_cond_dim is not None: #如果观测作为条件输入
            cond_dim += global_cond_dim  #则加上  这里global_cond_dim = 观测维度 x 观测帧数
        #all_dims[:-1] [2, 256, 512](去尾)    all_dims[1:] [256, 512, 1024] (去头)
        in_out = list(zip(all_dims[:-1], all_dims[1:])) #配对后 [(2, 512), (512, 1024), (1024, 2048)]

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])
        #U-Net中间层
        mid_dim = all_dims[-1]  #最后一个元素  2048
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D( #1D 条件残差块
                #输入维度  输出维度  条件维度（正弦波时间编码 + ResNet图像特征）
                mid_dim, mid_dim, cond_dim=cond_dim, 
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale #条件注入层的初始化权重
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])
        #U-Net编码器部分，即下采样
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out): #[(2, 512), (512, 1024), (1024, 2048)]
            is_last = ind >= (len(in_out) - 1) #检查当前是否是 最后一层
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(  # 升维
                    dim_in, dim_out, cond_dim=cond_dim, #2 -> 512   512 -> 1024   1024 -> 2048
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(  # 在当前维度进行深层特征提取，增加非线性拟合能力
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity() #如果不是最后一层,则进行压缩，维度不变，动作序列长度减半16->8->4
            ]))
        #U-Net解码器部分，即上采样
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): #去掉第一层并且倒着遍历,为[(1024, 2048)，(512, 1024)]
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim, #4096 -> 1024   2048 -> 512  会用旧特征进行补齐
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,     #特征深度提取
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()  #增加序列长度 4->8->16
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1), #最后一步 映射回动作维度  512 -> 2，且不加激活函数
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps) #128

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1) #时间步特征 + 观测特征
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample  #[64, 2, 16]
        # print(f"Shape of x: {x.shape}")
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules): #下采样 [64,2,16]->[64,512,16]->[64,1024,8]->[64,2048,4]
            x = resnet(x, global_feature) #升维
            # print(f"Shape of x: {x.shape}")
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            # print(f"Shape of x: {x.shape}")
            h.append(x)   #保留旧特征
            x = downsample(x)
            # print(f"Shape of x: {x.shape}")
        for mid_module in self.mid_modules:  #[64,2048,4]->[64,2048,4]->[64,2048,4]
            x = mid_module(x, global_feature)
            # print(f"Shape of x: {x.shape}")

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  #[64,2048,4]->[64,1024,4]->[64,512,16]
            x = torch.cat((x, h.pop()), dim=1) #取出旧特征，与新特征进行连接
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)  #[64, 512, 16] -> [64, 2, 16]

        x = einops.rearrange(x, 'b t h -> b h t') #[64, 2, 16] -> [64, 16, 2]
        return x

